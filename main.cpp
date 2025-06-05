#include <cstdio>
#include <raylib.h>
#include <math.h>
#define FONTSIZE 75

inline Vector3 Vector3Scale(Vector3 a, float s) {
  return Vector3{a.x * s, a.y * s, a.z * s};
}

inline float Vector3Length(Vector3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

static void *memcpy(void *dest, const void *src, unsigned int n) {
  char *d = (char *)dest;
  const char *s = (const char *)src;
  for (unsigned int i = 0; i < n; i++) {
    d[i] = s[i];
  }
  return dest;
}

static char *strncpy(char *dest, const char *src, unsigned int n) {
  unsigned int i = 0;
  for (; i < n && src[i] != '\0'; i++) {
    dest[i] = src[i];
  }
  for (; i < n; i++) {
    dest[i] = '\0';
  }
  return dest;
}

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
        _freeBytes(bytes) {
  }

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
  template <typename F> void destroy_with(F &&f) {
    if (_memory) {
      f(_memory);
    }
  }
};

struct Scene {
  Image img;
  Texture2D tex;
  float time = {0.0f};
  Wave wave;
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
    return generate_random_ast_arena(depth,arena);
  }
}

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

struct MusicLorenzOscillator {
  Vector3 pos = {2.0f, 0.0f, 0.0f};
  float dt = 1e-5;
  float step() {
    pos = getAttractor(pos, dt);
    return Vector3Length(pos);
  }

  Vector3 stepv() {
    pos = getAttractor(pos, dt);
    return pos;
  }
};

MusicLorenzOscillator music_osc{};

static const char *intro_texts(float t) {
  if (t < 4.0) {
    return "Graffathon 2025!";
  }
  if (t >= 4.0 && t < 8.0) {
    return "LSD Picasso";
  }
  if (t >= 8.0 && t < 12.0) {
    return "By GreenHouse!";
  }
  if (t >= 12.0 && t < 16) {
    return "..in under 16 KB!";
  }
  if (t >= 16.0 && t < 20) {
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
  Sound snd = LoadSoundFromWave(sc->wave);
  PlaySound(snd);
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
  UnloadSound(snd);
  return point_counter;
}

static int demo(Scene *sc, BumpAllocator *arena, BumpAllocator *music_arena,
                float dur) {
  (void)music_arena;
  const int screenWidth = GetScreenWidth();
  const int screenHeight = GetScreenHeight();
  int depth = 8;
  arena->release();
  auto ast = generate_random_ast_arena(depth, arena);
  auto glsl_str = codegen_glsl_sawtooth(ast);
  float actual_time = 0.0;

  Sound snd = LoadSoundFromWave(sc->wave);
  PlaySound(snd);
  Shader shader = LoadShaderFromMemory(0, glsl_str);
  int locTime, locRes;

  locRes = GetShaderLocation(shader, "resolution");
  locTime = GetShaderLocation(shader, "time");
  float resolution[2] = {(float)screenWidth, (float)screenHeight};
  SetShaderValue(shader, locRes, resolution, SHADER_UNIFORM_VEC2);
  while (!WindowShouldClose() && actual_time < dur) {
    if (sc->time > 2) {
      // depth++;
      sc->time = 0.0;
      arena->release();
      ast = generate_random_ast_arena(depth, arena);
      glsl_str = codegen_glsl_sawtooth(ast);
      UnloadShader(shader);
      shader = LoadShaderFromMemory(0, glsl_str);
      locRes = GetShaderLocation(shader, "resolution");
      locTime = GetShaderLocation(shader, "time");
      SetShaderValue(shader, locRes, resolution, SHADER_UNIFORM_VEC2);
      SetShaderValue(shader, locTime, &sc->time, SHADER_UNIFORM_FLOAT);
    }
    SetShaderValue(shader, GetShaderLocation(shader, "time"), &sc->time,
                   SHADER_UNIFORM_FLOAT);
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
  UnloadSound(snd);
  UnloadShader(shader);
  return 0;
}

static void outro(Scene *sc, BumpAllocator *arena, float dur, int n) {
  (void)arena;
  (void)sc;
  const float W = GetScreenWidth();
  const float H = GetScreenHeight();
  float actual_time = 0.0;
  Sound snd = LoadSoundFromWave(sc->wave);
  PlaySound(snd);
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
  UnloadSound(snd);
  arena->release();
}

// Singature has to be (float,....)
static float custom_track_1(float t) {
  static float freqs[6] = {220.0f, 261.63f, 293.66f, 329.63f, 392.0f, 440.0f};
  static size_t n_freqs = sizeof(freqs) / sizeof(freqs[0]);
  float bpm = 3 * 120.0f;
  float beat_period = 90.0f / bpm;
  float beat_phase = fmodf(t, beat_period);
  float envelope = sinf(M_PI * beat_phase / beat_period);
  static float last_switch = -1000.0f;
  static int base_idx = 0;
  static float phases[4] = {};
  static float amps[4] = {};
  if (t - last_switch > 2.0f) {
    base_idx = rand() % (n_freqs - 4);
    for (int i = 0; i < 4; ++i) {
      phases[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * M_PI;
      amps[i] = 0.8f + ((float)rand() / (float)RAND_MAX) * 0.4f;
    }
    last_switch = t;
  }
  float sum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    float freq = freqs[base_idx + i];
    sum += amps[i] * sinf(2.0f * M_PI * freq * t + phases[i]);
  }
  return envelope * (sum / 4.0f);
}

static float lorenz_track(float t) {
  float freq = music_osc.step();
  auto beat = [](float t) {
    float beat_freq = 4.0f;
    float base_freq = 32.0f + 16.0f * sinf(M_PI * t);
    return sinf(2.0f * M_PI * beat_freq * t + 90.0f * (M_PI / 180.0f)) *
           sinf(2.0f * M_PI * base_freq * t);
  };
  return 0.7 * sinf(2.0f * M_PI * freq * t) + 0.3 * beat(t);
}

template <typename T> T static clamp(const T &value, const T min, const T max) {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

template <typename F, typename... Args>
static Wave generate_music(float duration, BumpAllocator *arena, F &&f,
                           Args &&...args) {
  using music_type_t = short;
  constexpr size_t sampleRate = 48000;
  const size_t sampleCount = static_cast<size_t>(duration * sampleRate);
  music_type_t *samples = arena->allocate<music_type_t>(sampleCount);

  for (size_t i = 0; i < sampleCount; i++) {
    float t = static_cast<float>(i) / sampleRate;
    float val = f(t, std::forward<Args>(args)...);
    val = clamp(val, -1.0f, 1.0f);
    samples[i] = static_cast<music_type_t>(val * 31000);
  }

  Wave wave = {
      .frameCount = static_cast<unsigned int>(sampleCount),
      .sampleRate = sampleRate,
      .sampleSize = 16,
      .channels = 1,
      .data = samples,
  };
  return wave;
}

/*
NOTES
+Due to memory pool usage waves do not need to be unloaded
*/
extern "C" {
int jump_start() {
  // Setttings
  constexpr int seed = 142; // Adam no touch! (0:142) (1:512)
  constexpr size_t SCENE_MEMORY_POOL = 512 * 1024ul * 1024ul;
  constexpr size_t MUSIC_MEMORY_POOL = 1024ul * 1024ul * 1024ul;
  constexpr int screenWidth = 2 * 72 * 16;
  constexpr int screenHeight = 2 * 72 * 9;
  constexpr float intro_dur = 20.0f;
  constexpr float demo_dur = 90.0f;
  constexpr float outro_dur = intro_dur;
  constexpr int FPS = 60;
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
  SetTargetFPS(FPS);
  InitAudioDevice();
  
        //Setup Scene
        Scene scene{.img=GenImageColor(screenWidth, screenHeight, BLACK),.tex={},.time=0.0,.wave={}};
        scene.tex = LoadTextureFromImage(scene.img);
       
        //Intro 
        scene.wave = generate_music(intro_dur,&music_arena,lorenz_track);
        auto n=intro(&scene, &scene_arena, intro_dur);
        memcpy(scratch_memory,scene_memory,n*sizeof(Vector2));
                               
        scene_arena.release();
        music_arena.release();
        scene.wave={};

        // Main demo with AST
        scene.wave = generate_music(demo_dur,&music_arena,custom_track_1);
        demo(&scene, &scene_arena,&music_arena,demo_dur);
        music_arena.release();

        //Outro
        scene.wave = generate_music(outro_dur,&music_arena,lorenz_track);
        outro(&scene, &scratch_arena, outro_dur, n);
        music_arena.release();
        
  UnloadImage(scene.img);
  UnloadTexture(scene.tex);
  CloseWindow();
  // clang-format on
  return 0;
}
}

#ifdef WITH_MAIN
int main() { return jump_start(); }
#endif
