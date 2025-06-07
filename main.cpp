#include <cmath>
#include <cstdio>
#include <math.h>
#include <new>
#include <raylib.h>

// MUSIC Stuff
constexpr int FONTSIZE = 75;
constexpr float MAX_SAMPLES = 512;
constexpr float MAX_SAMPLES_PER_UPDATE = 4096;
constexpr float AST_INTERVAL = 2.0f;
constexpr float NOTE_C = 261.63;
constexpr float NOTE_CS = 277.18;
constexpr float NOTE_D = 293.66;
constexpr float NOTE_DS = 311.13;
constexpr float NOTE_E = 329.63;
constexpr float NOTE_Eb = 311.13f;
constexpr float NOTE_F = 349.23;
constexpr float NOTE_FS = 369.99;
constexpr float NOTE_G = 392.00;
constexpr float NOTE_GS = 415.30;
constexpr float NOTE_A = 440.00;
constexpr float NOTE_AS = 466.16;
constexpr float NOTE_B = 493.88;
constexpr float SEMITONE = 1.05946;
constexpr float SR = 48000.0f;

constexpr float alpha(float cutoff_freq) {
  const float rc = 1.0f / (2.0f * M_PI * cutoff_freq);
  const float dt = 1.0f / SR;
  return dt / (rc + dt);
}

inline float noise() { return 2.0f * ((rand() / (float)RAND_MAX) - 0.5f); }
static constexpr float osc_sine(float t, float fundamental) {
  return sinf(2.0f * M_PI * fundamental * t);
}

// https://en.wikipedia.org/wiki/Square_wave_(waveform)
static constexpr float osc_square(float t, float fundamental) {
  const float nyquist = SR * 0.5f;
  const int max_harmonic = (int)(nyquist / fundamental);
  float value = 0.0f;
  for (int k = 1; k <= max_harmonic; ++k) {
    value += sinf(2.0f * M_PI * (2.0f * k - 1.0f) * fundamental * t) /
             (2.0f * k - 1.0f);
  }
  value *= 4.0f / M_PI;
  return value;
}

// https://en.wikipedia.org/wiki/Sawtooth_wave
static constexpr float osc_sawtooth(float t, float fundamental) {
  const float nyquist = SR * 0.5f;
  const int max_harmonic = (int)(nyquist / fundamental);
  float value = 0.0f;
  for (int k = 1; k <= max_harmonic; ++k) {
    value += sinf(2.0f * M_PI * fundamental * k * t) / k;
  }
  value *= -2.0f / M_PI;
  return value;
}

static constexpr float LPF(float input, float &prev_output, float alpha) {
  prev_output = prev_output + alpha * (input - prev_output);
  return prev_output;
}

struct ADSR {
  ADSR(float o, float l, float a, float d, float s, float r)
      : onset(o), note_length(l), A(a), D(d), S(s), R(r) {}
  float onset;
  float note_length;
  float A;
  float D;
  float S;
  float R;
};

static constexpr float adsr(float t, const ADSR &env) {
  float dt = t - env.onset;
  if (dt < 0.0f)
    return 0.0f;
  if (dt < env.A)
    return dt / env.A;
  if (dt < env.A + env.D)
    return 1.0f - (1.0f - env.S) * ((dt - env.A) / env.D);
  if (dt < env.note_length)
    return env.S;
  if (dt < env.note_length + env.R)
    return env.S * (1.0f - (dt - env.note_length) / env.R);
  return 0.0f;
}

static Vector3 Vector3Scale(Vector3 a, float s) {
  return Vector3{a.x * s, a.y * s, a.z * s};
}

static inline float Vector3Length(Vector3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

constexpr static void *memcpy(void *dest, const void *src, unsigned int n) {
  char *d = (char *)dest;
  const char *s = (const char *)src;
  for (unsigned int i = 0; i < n; i++) {
    d[i] = s[i];
  }
  return dest;
}

constexpr static char *strncpy(char *dest, const char *src, unsigned int n) {
  unsigned int i = 0;
  for (; i < n && src[i] != '\0'; i++) {
    dest[i] = src[i];
  }
  for (; i < n; i++) {
    dest[i] = '\0';
  }
  return dest;
}

float static clamp(const float &value, const float min, const float max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

static float wrap(float x, float minVal, float maxVal) {
  float range = maxVal - minVal;
  return minVal + fmodf((x - minVal + range), range);
}

// IVAN
static void callback(void *buffer, unsigned int frames) {
  static float demo_time = 0.0f;
  static float dt = 1.0f / SR;
  static float prev = 0;
  float *const d = reinterpret_cast<float *>(buffer);

  for (unsigned int i = 0; i < frames; ++i) {
    float sample = 0.0f;
    const float bar_time = fmodf(demo_time, 4.0f);

    // KICK
    for (int b = 0; b < 4; ++b) {
      const float onset = b * 1.0f;
      const float env =
          adsr(bar_time, ADSR(onset, 0.12f, 0.01f, 0.08f, 0.3f, 0.08f));
      sample += 0.9f * env * osc_sine(demo_time, 60.0f);
    }

    // SNARE
    for (int b = 0; b < 2; ++b) {
      const float onset = 1.0f + b * 2.0f;
      const float pitch_mod = 1.0f + 0.5f * (demo_time * 0.25f);
      const float env =
          adsr(bar_time, ADSR(onset, 0.08f, 0.01f, 0.05f, 0.25f, 0.05f));
      sample += 0.5f * env * noise() * pitch_mod;
    }

    // HIGH HAT
    for (int b = 0; b < 16; ++b) {
      const float onset = b * 0.25f;
      const float env =
          adsr(bar_time, ADSR(onset, 0.015f, 0.002f, 0.01f, 0.06f, 0.01f));
      sample += 0.12f * env * noise();
    }

    // Output
    const float a = alpha(850.0f); 
    sample = LPF(sample, prev, a);

    sample = clamp(sample, -1.0f, 1.0f);
    d[2 * i] = sample;
    d[2 * i + 1] = sample;
    demo_time += dt;
  }
}
//~Music stuff

// AST and Memory Pool
struct BumpAllocator {
  void *_memory = nullptr;
  size_t _sp = 0;
  size_t _totalBytes = 0;
  size_t _allocBytes = 0;
  size_t _freeBytes = 0;

  BumpAllocator(const BumpAllocator &) = delete;
  BumpAllocator &operator=(const BumpAllocator &) = delete;
  BumpAllocator(BumpAllocator &&) = delete;
  BumpAllocator &operator=(BumpAllocator &&) = delete;

  BumpAllocator(void *buffer, size_t bytes)
      : _memory(buffer), _sp(0), _totalBytes(bytes), _allocBytes(0),
        _freeBytes(bytes) {}

  template <typename T> T *allocate(size_t count) {
    if (count == 0) {
      return nullptr;
    }
    const size_t bytesToAllocate = count * sizeof(T);
    const size_t alignment = fmax(alignof(T), size_t(8));

    size_t baseAddress = reinterpret_cast<size_t>((char *)_memory + _sp);
    size_t padding = (baseAddress % alignment == 0)
                         ? 0
                         : (alignment - baseAddress % alignment);
    size_t totalBytes = padding + bytesToAllocate;

    if (_sp + totalBytes > _totalBytes) {
      abort();
      return nullptr;
    }

    void *ptr = (char *)_memory + _sp + padding;
    _sp += totalBytes;
    _allocBytes += totalBytes;
    _freeBytes -= totalBytes;
    return reinterpret_cast<T *>(ptr);
  }

  void release() {
    _sp = 0;
    _allocBytes = 0;
    _freeBytes = _totalBytes;
  }
};

struct Scene {
  Image img;
  Texture2D tex;
  float time = {0.0f};
};

enum class BINOPS { ADD, MUL, N_BINOPS };

enum class UNOPS { SIN, COS, SQRT, POW2, N_UNOPS };

enum class AST_NODE_TYPE : u_int8_t {
  AST_NODE_NUMBER,
  AST_NODE_VARIABLE,
  AST_NODE_BINARY,
  AST_NODE_UNARY
};

// Fwd declare these guys
struct NodeBinary;
struct NodeUnary;
struct NodeNumber;
struct NodeVariable;

struct GenericNode {
  AST_NODE_TYPE type;
  union {
    NodeBinary *binary;
    NodeUnary *unary;
    NodeNumber *number;
    NodeVariable *variable;
  } data;
};

struct NodeBinary {
  BINOPS op;
  GenericNode lhs;
  GenericNode rhs;
};

struct NodeUnary {
  UNOPS op;
  GenericNode lhs;
};

struct NodeNumber {
  float v;
};

struct NodeVariable {
  char v[8];
};

static const char *to_string(BINOPS op) {
  switch (op) {
  case BINOPS::ADD:
    return "+";
  case BINOPS::MUL:
    return "*";
  default:
    return nullptr;
  }
}

static const char *to_string(UNOPS op) {
  switch (op) {
  case UNOPS::SIN:
    return "sin";
  case UNOPS::COS:
    return "cos";
  case UNOPS::POW2:
    return "pow";
  case UNOPS::SQRT:
    return "sqrt";
  default:
    return nullptr;
  }
}

static char *write_expr(const GenericNode &node, char *out) {
  switch (node.type) {
  case AST_NODE_TYPE::AST_NODE_NUMBER:
    out += sprintf(out, "%f", node.data.number->v);
    break;
  case AST_NODE_TYPE::AST_NODE_VARIABLE:
    out += sprintf(out, "%s", node.data.variable->v);
    break;
  case AST_NODE_TYPE::AST_NODE_UNARY: {
    auto *n = node.data.unary;
    if (n->op == UNOPS::POW2) {
      out += sprintf(out, "(");
      out = write_expr(n->lhs, out);
      out += sprintf(out, ")*(");
      out = write_expr(n->lhs, out);
      out += sprintf(out, ")");
    } else if (n->op == UNOPS::SQRT) {
      out += sprintf(out, "sqrt(abs(");
      out = write_expr(n->lhs, out);
      out += sprintf(out, "))");
    } else {
      out += sprintf(out, "%s(2.0*3.14*", to_string(n->op));
      out = write_expr(n->lhs, out);
      out += sprintf(out, ")");
    }
    break;
  }
  case AST_NODE_TYPE::AST_NODE_BINARY: {
    auto *n = node.data.binary;
    out += sprintf(out, "(");
    out = write_expr(n->lhs, out);
    out += sprintf(out, " %s ", to_string(n->op));
    out = write_expr(n->rhs, out);
    out += sprintf(out, ")");
    break;
  }
  }
  return out;
}

static const char *codegen_glsl_sawtooth(const GenericNode &node) {
  static char buffer[65536];
  char *out = buffer;
  out += sprintf(out, "#version 330 core\n"
                      "out vec4 FragColor;\n"
                      "uniform vec2 resolution;\n"
                      "uniform float time;\n\n"
                      "float saw(float x) {\n"
                      "    return fract(x);\n"
                      "}\n\n"
                      "void main() {\n"
                      "    vec2 uv = gl_FragCoord.xy / resolution;\n"
                      "    float x = uv.x * 2.0 - 1.0;\n"
                      "    float y = uv.y * 2.0 - 1.0;\n"
                      "    float t = time / 5.0;\n"
                      "    float pulse = 1.0 + 0.5 * sin(6.2831 * t);\n"
                      "    float base = ");
  out = write_expr(node, out);
  out += sprintf(out,
                 ";\n"
                 "    float r = saw(base * 2.0 + 0.0) * pulse;\n"
                 "    float g = saw(base * 1.5 + 0.33) * pulse;\n"
                 "    float b = saw(base * -3.0 + 0.67) * pulse;\n"
                 "    FragColor = vec4(clamp(vec3(r, g, b), 0.0, 1.0), 1.0);\n"
                 "}\n");
  return buffer;
}

static const char *codegen_glsl_sawtooth_grayscale(const GenericNode &node) {
  static char buffer[65536];
  char *out = buffer;
  out += sprintf(out, "#version 330 core\n"
                      "out vec4 FragColor;\n"
                      "uniform vec2 resolution;\n"
                      "uniform float time;\n\n"
                      "float saw(float x) {\n"
                      "    return fract(x);\n"
                      "}\n\n"
                      "void main() {\n"
                      "    vec2 uv = gl_FragCoord.xy / resolution;\n"
                      "    float x = uv.x * 2.0 - 1.0;\n"
                      "    float y = uv.y * 2.0 - 1.0;\n"
                      "    float t = time / 5.0;\n"
                      "    float pulse = 1.0 + 0.5 * sin(6.2831 * t);\n"
                      "    float base = ");
  out = write_expr(node, out);
  out +=
      sprintf(out, ";\n"
                   "    float gray = saw(base * 2.0) * pulse;\n"
                   "    FragColor = vec4(vec3(clamp(gray, 0.0, 1.0)), 1.0);\n"
                   "}\n");
  return buffer;
}

static float rand_float(float min = -1.0f, float max = 1.0f) {
  const float scale = rand() / (float)RAND_MAX;
  return min + scale * (max - min);
}

static int rand_int(int min, int max) { return min + rand() % (max - min + 1); }

static BINOPS random_binop() {
  return static_cast<BINOPS>(
      rand_int(0, -1 + static_cast<int>(BINOPS::N_BINOPS)));
}

static UNOPS random_unop() {
  return static_cast<UNOPS>(rand_int(0, -1 + static_cast<int>(UNOPS::N_UNOPS)));
}

static GenericNode generate_random_ast_arena(int depth, BumpAllocator *arena) {
  if (depth <= 0) {
    if (rand_int(0, 1)) {
      auto *number =
          new (arena->allocate<NodeNumber>(1)) NodeNumber{.v = rand_float()};
      return GenericNode{.type = AST_NODE_TYPE::AST_NODE_NUMBER,
                         .data = {.number = number}};
    } else {
      int choice = rand_int(0, 2);
      const char *name = (choice == 0) ? "x" : (choice == 1) ? "y" : "t";
      auto *var = arena->allocate<NodeVariable>(1);
      strncpy(var->v, name, sizeof(var->v));
      var->v[sizeof(var->v) - 1] = '\0';
      return GenericNode{.type = AST_NODE_TYPE::AST_NODE_VARIABLE,
                         .data = {.variable = var}};
    }
  }

  int choice = rand_int(0, 3);
  switch (choice) {
  case 0:
  case 1:
  case 3: {
    auto *binary = new (arena->allocate<NodeBinary>(1))
        NodeBinary{.op = random_binop(),
                   .lhs = generate_random_ast_arena(depth - 1, arena),
                   .rhs = generate_random_ast_arena(depth - 1, arena)};
    return GenericNode{.type = AST_NODE_TYPE::AST_NODE_BINARY,
                       .data = {.binary = binary}};
  }
  case 2: {
    auto *unary = new (arena->allocate<NodeUnary>(1))
        NodeUnary{.op = random_unop(),
                  .lhs = generate_random_ast_arena(depth - 1, arena)};
    return GenericNode{.type = AST_NODE_TYPE::AST_NODE_UNARY,
                       .data = {.unary = unary}};
  }
  default:
    return generate_random_ast_arena(depth, arena);
  }
}
//~AST and Memory Pool

// Randoms stuff
template <typename T, typename F>
static constexpr auto rk4(T yn, T tn, T h, F &&f) -> T {
  const auto k1 = f(tn, yn);
  const auto k2 = f(tn + h / T(2.0), yn + h * k1 / T(2.0));
  const auto k3 = f(tn + h / T(2.0), yn + h * k2 / T(2.0));
  const auto k4 = f(tn + h, yn + h * k3);
  const auto yn1 = yn + (h / T(6.0)) * (k1 + T(2.0) * k2 + T(2.0) * k3 + k4);
  return yn1;
}

static Vector3 getAttractor(Vector3 r0, float dt) {
  static constexpr float SIGMA = 10.0;
  static constexpr float BETA = 8.0 / 3.0;
  static constexpr float RHO = 28.0;
  Vector3 r1 = r0;
  float t = 0.0;
  auto f1 = [&](float t, float x) {
    (void)t;
    return SIGMA * (r0.y - x);
  };
  auto f2 = [&](float t, float y) {
    (void)t;
    return r0.x * (RHO - r0.z) - y;
  };
  auto f3 = [&](float t, float z) {
    (void)t;
    (void)z;
    return r0.x * r0.y - BETA * r0.z;
  };
  r1.x = rk4(r0.x, t, dt, f1);
  r1.y = rk4(r0.y, t, dt, f2);
  r1.z = rk4(r0.z, t, dt, f3);
  return r1;
}
//~Random stuff

// Scene specific stuff and main demos
static const char *intro_texts(float t) {
  if (t < 2.0) {
    return "Graffathon 2025!";
  }
  if (t >= 2.0 && t < 5.0) {
    return "A.S.T. Picasso";
  }
  if (t >= 5.0 && t < 8.0) {
    return "By GreenHouse!";
  }
  if (t >= 8.0 && t < 10) {
    return "..in under 16 KB!";
  }
  if (t >= 10.0 && t < 12) {
    return " 16 KB? LMFAO! <3";
  }
  return nullptr;
}

static size_t intro(Scene *sc, BumpAllocator *arena, float dur) {
  (void)arena;
  (void)sc;
  const float W = GetScreenWidth();
  const float H = GetScreenHeight();
  float actual_time = 0.0;
  constexpr float dt = 1 * 1e-2;
  Vector3 p1{1.0f, 0.0f, 0.0};
  Vector3 p2{0.8f, 0.0f, 0.0};
  size_t point_counter = 0;
  Vector2 *points = arena->allocate<Vector2>(1 < 20);
  while (!WindowShouldClose() && actual_time < dur) {
    ClearBackground(BLACK);
    auto msg = intro_texts(actual_time);
    if (!msg) {
      return point_counter;
    }
    int text_width = MeasureText(msg, 100);
    BeginDrawing();
    for (size_t i = 0; i < 16 / 2; ++i) {
      auto ps1 = Vector3Scale(p1, 24.0f);
      auto ps2 = Vector3Scale(p2, 24.0f);
      Vector2 cand1 = Vector2{ps1.x + (W / 2) + (W / 8), ps1.y + (H / 2)};
      Vector2 cand2 = Vector2{ps2.x + (W / 2) + (W / 8), ps2.y + (H / 2)};
      points[point_counter++] = cand1;
      points[point_counter++] = cand2;
      p1 = getAttractor(p1, dt / 2);
      p2 = getAttractor(p2, dt / 2);
    }
    for (size_t i = 0; i < point_counter; ++i) {
      DrawCircleV(points[i], 1, i % 2 == 0 ? RED : BLUE);
    }
    DrawText(msg, W / 4.0f - 0.5 * text_width, H / 2.0f, FONTSIZE, GOLD);
    actual_time += GetFrameTime();
    EndDrawing();
  }
  return point_counter;
}

static const char *intro_post1_texts(float t) {
  if (t < 3.0) {
    return "Today we produce \n abstract graphics using \n random ASTs.";
  }
  if (t >= 3.0 && t < 8.0) {
    return "An AST is an \n Asbtract Synatx Tree!";
  }
  if (t >= 8.0 && t < 12.0) {
    return "Today each AST node \n is an operator.";
  }
  if (t >= 12.0 && t < 15) {
    return "Our vocabulary for today: ";
  }
  if (t >= 15.0 && t < 19) {
    return "add, mut, mod, exp, sin,\n sqrt, pow";
  }
  if (t >= 19.0 && t < 23) {
    return "Here is an AST: \n\n f=sin(cos(x)+\npow(sqrt(2*x+sin(y)))";
  }
  if (t >= 23.0 && t < 26) {
    return "Let's see what this \nlooks like:";
  }
  if (t >= 26.0 && t < 28) {
    return "All that below 16 KB!\n\nLet's go!";
  }
  return nullptr;
}

static int demo(Scene *sc, BumpAllocator *arena, BumpAllocator *music_arena,
                float dur, int depth = 8);

static void post_intro(Scene *sc, BumpAllocator *arena, float dur) {
  (void)arena;
  (void)sc;
  const float W = GetScreenWidth();
  const float H = GetScreenHeight();
  float actual_time = 0.0;
  constexpr float dt = 1 * 1e-2;
  Vector3 p1{1.3f, 0.0f, 0.0};
  Vector3 p2{0.4f, 0.0f, 0.0};
  size_t point_counter = 0;
  Vector2 *points = arena->allocate<Vector2>(1 < 20);
  bool flag = true;

  while (!WindowShouldClose() && actual_time < dur) {
    ClearBackground(BLACK);
    // Spawn an AST and then revert actual time back
    if (actual_time > 26 && flag) {
      float store = actual_time;
      // Create a Pool from my Pool -> Insane!
      BumpAllocator tmp1(arena->allocate<float>(1024 * 1024),
                         1024 * sizeof(float));
      flag = false;
      demo(sc, arena, &tmp1, 2, 8);
      ClearBackground(BLACK);
      actual_time = store + GetFrameTime();
      continue;
    }
    auto msg = intro_post1_texts(actual_time);
    if (!msg) {
      return;
    }
    int text_width = MeasureText(msg, 100);
    BeginDrawing();
    ClearBackground(BLACK);
    for (size_t i = 0; i < 16 / 2; ++i) {
      auto ps1 = Vector3Scale(p1, 20.0f);
      auto ps2 = Vector3Scale(p2, 20.0f);
      Vector2 cand1 = Vector2{ps1.z + (W / 2), ps1.x + (H / 2)};
      Vector2 cand2 = Vector2{ps2.z + (W / 2), ps2.x + (H / 2)};
      points[point_counter++] = cand1;
      points[point_counter++] = cand2;
      p1 = getAttractor(p1, dt / 2);
      p2 = getAttractor(p2, dt / 2);
    }
    for (size_t i = 0; i < point_counter; ++i) {
      DrawCircleV(points[i], 1, i % 2 == 0 ? GOLD : PINK);
    }

    if (actual_time > 20.0 && actual_time <= 26) {
      const float anim_dt = 0.25f;
      float fontSize = 32;
      float rootX = 0.25f * W;
      float rootY = 0.45f * H;
      DrawEllipse(rootX, rootY, 64, 32, RED);
      const char *text = "sin";
      int textWidth = MeasureText(text, fontSize);
      DrawText(text, rootX - textWidth / 2.0f, rootY - fontSize / 2.0f,
               fontSize, RAYWHITE);
      if (actual_time < 19.5)
        goto checkpoint;

      float plusX = rootX;
      float plusY = rootY + 96;
      DrawEllipse(plusX, plusY, 64, 32, ORANGE);
      text = "+";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, plusX - textWidth / 2.0f, plusY - fontSize / 2.0f,
               fontSize, RAYWHITE);
      DrawLine(rootX, rootY + 32, plusX, plusY - 32, LIGHTGRAY);
      if (actual_time < 19.5 + anim_dt)
        goto checkpoint;

      float cosX = plusX - 160;
      float cosY = plusY + 96;
      DrawEllipse(cosX, cosY, 64, 32, GREEN);
      text = "cos";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, cosX - textWidth / 2.0f, cosY - fontSize / 2.0f, fontSize,
               RAYWHITE);
      DrawLine(plusX, plusY + 32, cosX, cosY - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 2 * anim_dt)
        goto checkpoint;

      float x1X = cosX;
      float x1Y = cosY + 96;
      DrawEllipse(x1X, x1Y, 64, 32, DARKGREEN);
      text = "x";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, x1X - textWidth / 2.0f, x1Y - fontSize / 2.0f, fontSize,
               RAYWHITE);
      DrawLine(cosX, cosY + 32, x1X, x1Y - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 3 * anim_dt)
        goto checkpoint;

      float powX = plusX + 160;
      float powY = plusY + 96;
      DrawEllipse(powX, powY, 64, 32, BLUE);
      text = "pow";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, powX - textWidth / 2.0f, powY - fontSize / 2.0f, fontSize,
               RAYWHITE);
      DrawLine(plusX, plusY + 32, powX, powY - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 4 * anim_dt)
        goto checkpoint;

      float sqrtX = powX;
      float sqrtY = powY + 96;
      DrawEllipse(sqrtX, sqrtY, 64, 32, PURPLE);
      text = "sqrt";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, sqrtX - textWidth / 2.0f, sqrtY - fontSize / 2.0f,
               fontSize, RAYWHITE);
      DrawLine(powX, powY + 32, sqrtX, sqrtY - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 5 * anim_dt)
        goto checkpoint;

      float plus2X = sqrtX;
      float plus2Y = sqrtY + 96;
      DrawEllipse(plus2X, plus2Y, 64, 32, ORANGE);
      text = "+";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, plus2X - textWidth / 2.0f, plus2Y - fontSize / 2.0f,
               fontSize, RAYWHITE);
      DrawLine(sqrtX, sqrtY + 32, plus2X, plus2Y - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 6 * anim_dt)
        goto checkpoint;

      float mulX = plus2X - 120;
      float mulY = plus2Y + 96;
      DrawEllipse(mulX, mulY, 64, 32, GRAY);
      text = "*";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, mulX - textWidth / 2.0f, mulY - fontSize / 2.0f, fontSize,
               RAYWHITE);
      DrawLine(plus2X, plus2Y + 32, mulX, mulY - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 7 * anim_dt)
        goto checkpoint;

      float num2X = mulX - 64;
      float num2Y = mulY + 96;
      DrawEllipse(num2X, num2Y, 64, 32, DARKGRAY);
      text = "2";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, num2X - textWidth / 2.0f, num2Y - fontSize / 2.0f,
               fontSize, RAYWHITE);
      DrawLine(mulX, mulY + 32, num2X, num2Y - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 8 * anim_dt)
        goto checkpoint;

      float x2X = mulX + 64;
      float x2Y = mulY + 96;
      DrawEllipse(x2X, x2Y, 64, 32, DARKGREEN);
      text = "x";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, x2X - textWidth / 2.0f, x2Y - fontSize / 2.0f, fontSize,
               RAYWHITE);
      DrawLine(mulX, mulY + 32, x2X, x2Y - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 9 * anim_dt)
        goto checkpoint;

      float sinY_X = plus2X + 120;
      float sinY_Y = plus2Y + 96;
      DrawEllipse(sinY_X, sinY_Y, 64, 32, RED);
      text = "sin";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, sinY_X - textWidth / 2.0f, sinY_Y - fontSize / 2.0f,
               fontSize, RAYWHITE);
      DrawLine(plus2X, plus2Y + 32, sinY_X, sinY_Y - 32, LIGHTGRAY);
      if (actual_time < 19.5 + 10 * anim_dt)
        goto checkpoint;

      float yX = sinY_X;
      float yY = sinY_Y + 96;
      DrawEllipse(yX, yY, 64, 32, DARKGREEN);
      text = "y";
      textWidth = MeasureText(text, fontSize);
      DrawText(text, yX - textWidth / 2.0f, yY - fontSize / 2.0f, fontSize,
               RAYWHITE);
      DrawLine(sinY_X, sinY_Y + 32, yX, yY - 32, LIGHTGRAY);
    }
  checkpoint:
    DrawText(msg, W / 4.0f - 0.4 * text_width, 0.175 * H, FONTSIZE, GOLD);
    actual_time += GetFrameTime();
    EndDrawing();
  }
  return;
}

static int demo(Scene *sc, BumpAllocator *arena, BumpAllocator *music_arena,
                float dur, int depth) {
  (void)music_arena;
  const int screenWidth = GetScreenWidth();
  const int screenHeight = GetScreenHeight();
  arena->release();
  auto ast = generate_random_ast_arena(depth, arena);
  auto glsl_str = codegen_glsl_sawtooth_grayscale(ast);
  float actual_time = 0.0;
  Shader shader = LoadShaderFromMemory(0, glsl_str);
  int locTime, locRes;
  locRes = GetShaderLocation(shader, "resolution");
  locTime = GetShaderLocation(shader, "time");
  float resolution[2] = {(float)screenWidth, (float)screenHeight};
  SetShaderValue(shader, locRes, resolution, SHADER_UNIFORM_VEC2);
  while (!WindowShouldClose() && actual_time < dur) {
    if (sc->time > AST_INTERVAL) {
      sc->time = 0.0;
      arena->release();
      ast = generate_random_ast_arena(depth, arena);
      glsl_str = (actual_time >= dur / 2.0f)
                     ? codegen_glsl_sawtooth(ast)
                     : codegen_glsl_sawtooth_grayscale(ast);
      UnloadShader(shader);
      shader = LoadShaderFromMemory(0, glsl_str);
      locRes = GetShaderLocation(shader, "resolution");
      locTime = GetShaderLocation(shader, "time");
      SetShaderValue(shader, locRes, resolution, SHADER_UNIFORM_VEC2);
    }
    SetShaderValue(shader, locTime, &sc->time, SHADER_UNIFORM_FLOAT);
    BeginDrawing();
    ClearBackground(WHITE);
    BeginShaderMode(shader);
    DrawRectangle(0, 0, screenWidth, screenHeight, WHITE);
    EndShaderMode();
    DrawFPS(0, 0);
    sc->time += GetFrameTime() / 2;
    actual_time += GetFrameTime();
    EndDrawing();
  }
  UnloadShader(shader);
  return 0;
}

static void outro(Scene *sc, BumpAllocator *arena, float dur, int n) {
  (void)arena;
  (void)sc;
  const float W = GetScreenWidth();
  const float H = GetScreenHeight();
  float actual_time = 0.0;
  Vector2 *points = reinterpret_cast<Vector2 *>(arena->_memory);
  const char *msg = "See you next year!";
  while (!WindowShouldClose() && actual_time < dur) {
    ClearBackground(BLACK);
    int text_width = MeasureText(msg, 100);
    BeginDrawing();
    for (int i = n; i > 0; --i) {
      DrawCircleV(points[i], 1, i % 2 == 0 ? RED : BLUE);
    }
    n -= 16;
    DrawText(msg, W / 4.0f - 0.5 * text_width, H / 2.0f, FONTSIZE, GOLD);
    actual_time += GetFrameTime();
    EndDrawing();
  }
  arena->release();
}
//~Scene specific stuff and main demos

/*
NOTES
+Due to memory pool usage waves do not need to be unloaded
*/
extern "C" {
int jump_start() {
  // Setttings
  constexpr int seed = 512; // Adam no touch! (0:142) (1:512)
  constexpr size_t SCENE_MEMORY_POOL = 1024ul * 1024ul * 1024ul;
  constexpr size_t MUSIC_MEMORY_POOL = 1024ul * 1024ul * 1024ul;
  constexpr int screenWidth = 2 * 72 * 16;
  constexpr int screenHeight = 2 * 72 * 9;
  constexpr float intro_dur = 3.0f;
  constexpr float post_intro_1_dur = 31.0f;
  constexpr float demo_dur = 60.0f;
  constexpr float outro_dur = intro_dur;
  constexpr int FPS = 30;
  //~Settings
  srand(seed);

  // Pool
  void *scene_memory = RL_MALLOC(SCENE_MEMORY_POOL);
  void *scratch_memory = RL_MALLOC(SCENE_MEMORY_POOL);
  void *music_memory = RL_MALLOC(MUSIC_MEMORY_POOL);
  if (!scene_memory || !music_memory || !scratch_memory) {
    return 1;
  }

  // Init pools
  BumpAllocator scene_arena(scene_memory, SCENE_MEMORY_POOL);
  BumpAllocator scratch_arena(scratch_memory, SCENE_MEMORY_POOL);
  BumpAllocator music_arena(music_memory, MUSIC_MEMORY_POOL);

  // clang-format off
  SetTraceLogLevel(TraceLogLevel::LOG_NONE);
  InitWindow(screenWidth, screenHeight, "");
  // DisableCursor();
  SetExitKey(KEY_ESCAPE);
  InitAudioDevice();
  SetAudioStreamBufferSizeDefault(1<<10);
  AudioStream stream = LoadAudioStream(SR, 32, 2);
  AttachAudioStreamProcessor(stream, callback);
  PlayAudioStream(stream);  
  SetTargetFPS(FPS);
   
          // Setup Scene
          Scene scene{.img = GenImageColor(screenWidth, screenHeight, BLACK),
                      .tex = {},
                      .time = 0.0};
          scene.tex = LoadTextureFromImage(scene.img);

          // Intro
          auto n = intro(&scene, &scene_arena, intro_dur);
          memcpy(scratch_memory, scene_memory, n * sizeof(Vector2));
          scene_arena.release();

          // Post Intro 1
          post_intro(&scene, &scene_arena, post_intro_1_dur);
          scene_arena.release();

          // Main demo with AST
          demo(&scene, &scene_arena, &music_arena, demo_dur,6);

          // Outro
          outro(&scene, &scratch_arena, outro_dur, n);
        
  UnloadImage(scene.img);
  UnloadTexture(scene.tex);
  UnloadAudioStream(stream);
  CloseAudioDevice(); 
  CloseWindow();
  // clang-format on
  if (scene_memory) {
    free(scene_memory);
  }
  if (music_memory) {
    free(music_memory);
  }
  if (scratch_memory) {
    free(scratch_memory);
  }
  return 0;
}
}
#ifdef WITH_MAIN
int main() { return jump_start(); }
#endif
