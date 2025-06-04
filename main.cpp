#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <omp.h>
#include <raylib.h>
#include <raymath.h>
#include <stdio.h>
#include <string>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <variant>

class BumpAllocator {
private:
  void *_memory = nullptr;
  std::size_t _sp = 0;
  std::size_t _totalBytes = 0;
  std::size_t _allocBytes = 0;
  std::size_t _freeBytes = 0;

public:
  BumpAllocator(const BumpAllocator &) = delete;
  BumpAllocator &operator=(const BumpAllocator &) = delete;
  BumpAllocator(BumpAllocator &&) = delete;
  BumpAllocator &operator=(BumpAllocator &&) = delete;

  BumpAllocator(void *buffer, std::size_t bytes)
      : _memory(buffer), _sp(0), _totalBytes(bytes), _allocBytes(0),
        _freeBytes(bytes) {
    if (!buffer) {
      std::abort();
    }
  }

  template <typename T> T *allocate(std::size_t count) {
    if (count == 0) {
      return nullptr;
    }
    const size_t bytesToAllocate = count * sizeof(T);
    const size_t alignment = std::max(alignof(T), size_t(8));

    std::size_t baseAddress =
        reinterpret_cast<std::size_t>((char *)_memory + _sp);
    std::size_t padding = (baseAddress % alignment == 0)
                              ? 0
                              : (alignment - baseAddress % alignment);
    std::size_t totalBytes = padding + bytesToAllocate;

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
  template <typename F> void destroy_with(F &&f) { f(_memory); }
};

struct Scene {
  Image img;
  Texture2D tex;
  float time = {0.0f};
  Wave wave;
};

enum class BINOPS : uint8_t { ADD, MUL, N_BINOPS };

enum class UNOPS : uint8_t { SIN, COS, SQRT, POW2, N_UNOPS };

float eval(BINOPS op, float lhs, float rhs) {
  float ret;
  switch (op) {
  case BINOPS::ADD:
    ret = (lhs + rhs) / 2.0f;
    break;
  case BINOPS::MUL:
    ret = (lhs * rhs);
    break;
  default:
    abort();
    break;
  }
  return ret;
}

float eval(UNOPS op, float lhs) {
  switch (op) {
  case UNOPS::SIN:
    return std::sin(2.0 * M_PI * (lhs * M_PI / 180.0f));
  case UNOPS::COS:
    return std::cos(2.0 * M_PI * (lhs * M_PI / 180.0f));
  case UNOPS::POW2:
    return std::pow(lhs, 2);
  case UNOPS::SQRT:
    return std::sqrt(std::abs(lhs));
  default:
    abort();
  }
}

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
using GenericNode =
    std::variant<NodeNumber *, NodeVariable *, NodeUnary *, NodeBinary *>;

struct NodeBinary {
  BINOPS op;
  std::variant<NodeNumber *, NodeVariable *, NodeUnary *, NodeBinary *> lhs;
  std::variant<NodeNumber *, NodeVariable *, NodeUnary *, NodeBinary *> rhs;
};

struct NodeUnary {
  UNOPS op;
  std::variant<NodeNumber *, NodeVariable *, NodeUnary *, NodeBinary *> lhs;
};

struct NodeNumber {
  float v;
};

struct NodeVariable {
  std::string v;
};

float evaluate_node(const GenericNode &node, float x, float y, float t) {
  return std::visit(
      [&](auto &&n) -> float {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, NodeNumber *>) {
          return n->v;
        } else if constexpr (std::is_same_v<T, NodeBinary *>) {
          return eval(n->op, evaluate_node(n->lhs, x, y, t),
                      evaluate_node(n->rhs, x, y, t));
        } else if constexpr (std::is_same_v<T, NodeUnary *>) {
          return eval(n->op, evaluate_node(n->lhs, x, y, t));
        } else if constexpr (std::is_same_v<T, NodeVariable *>) {
          if (n->v == "x")
            return x;
          if (n->v == "y")
            return y;
          if (n->v == "t")
            return t;
          abort();
        } else {
          abort();
          return -1;
        }
      },
      node);
}

float rand_float(float min = -1.0f, float max = 1.0f) {
  float scale = rand() / (float)RAND_MAX;
  return min + scale * (max - min);
}

int rand_int(int min, int max) { return min + rand() % (max - min + 1); }

BINOPS random_binop() {
  return static_cast<BINOPS>(
      rand_int(0, -1 + static_cast<int>(BINOPS::N_BINOPS)));
}

UNOPS random_unop() {
  return static_cast<UNOPS>(rand_int(0, -1 + static_cast<int>(UNOPS::N_UNOPS)));
}

GenericNode generate_random_ast_arena(int depth, BumpAllocator *arena) {
  if (depth <= 0) {
    if (rand_int(0, 1)) {
      return new (arena->allocate<NodeNumber>(1)) NodeNumber{.v = rand_float()};
    } else {
      int choice = rand_int(0, 2);
      if (choice == 0)
        return new (arena->allocate<NodeVariable>(1)) NodeVariable{"x"};
      if (choice == 1)
        return new (arena->allocate<NodeVariable>(1)) NodeVariable{"y"};
      return new (arena->allocate<NodeVariable>(1)) NodeVariable{"t"};
    }
  }

  int choice = rand_int(0, 3);
  switch (choice) {
  case 0:
  case 1:
  case 3:
    return new (arena->allocate<NodeBinary>(1)) NodeBinary{
        .op = random_binop(),
        .lhs = generate_random_ast_arena(depth - 1, arena),
        .rhs = generate_random_ast_arena(depth - 1, arena),
    };
  case 2: {
    return new (arena->allocate<NodeUnary>(1)) NodeUnary{
        .op = random_unop(),
        .lhs = generate_random_ast_arena(depth - 1, arena),
    };
  }
  default:
    fprintf(stderr, "ERROR: Reached unreachable state in BINOP switch ");
    abort();
  }
}

std::string to_string(BINOPS op) {
  switch (op) {
  case BINOPS::ADD:
    return "+";
  case BINOPS::MUL:
    return "*";
  default:
    fprintf(stderr, "ERROR: Reached unreachable state in BINOP switch ");
    abort();
  }
}

std::string to_string(UNOPS op) {
  switch (op) {
  case UNOPS::SIN:
    return "sinf";
  case UNOPS::COS:
    return "cosf";
  case UNOPS::POW2:
    return "pow";
  case UNOPS::SQRT:
    return "sqrt";
  default:
    fprintf(stderr, "ERROR: Reached unreachable state in UNOP switch ");
    abort();
  }
}

// void print_ast(const GenericNode &node, std::ostream &out = std::cout) {
//   std::visit(
//       [&](auto &&n) {
//         using T = std::decay_t<decltype(n)>;
//         if constexpr (std::is_same_v<T, NodeNumber *>) {
//           out << n->v;
//         } else if constexpr (std::is_same_v<T, NodeVariable *>) {
//           out << n->v;
//         } else if constexpr (std::is_same_v<T, NodeUnary *>) {
//           out << to_string(n->op) << "(";
//           print_ast(n->lhs, out);
//           out << ")";
//         } else if constexpr (std::is_same_v<T, NodeBinary *>) {
//           out << "(";
//           print_ast(n->lhs, out);
//           out << " " << to_string(n->op) << " ";
//           print_ast(n->rhs, out);
//           out << ")";
//         } else {
//           out << "?";
//         }
//       },
//       node);
// }

void write_expr(const GenericNode &node, FILE *f) {
  std::visit(
      [&](auto &&n) {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, NodeNumber *>) {
          fprintf(f, "%f", n->v);
        } else if constexpr (std::is_same_v<T, NodeVariable *>) {
          fprintf(f, "%s", n->v.c_str());
        } else if constexpr (std::is_same_v<T, NodeUnary *>) {
          if (n->op == UNOPS::POW2) {
            fprintf(f, "(");
            write_expr(n->lhs, f);
            fprintf(f, ")*(");
            write_expr(n->lhs, f);
            fprintf(f, ")");
          } else if (n->op == UNOPS::SQRT) {
            fprintf(f, "sqrt(fabs(");
            write_expr(n->lhs, f);
            fprintf(f, "))");
          } else {
            fprintf(f, "%s(2.0*M_PI*", to_string(n->op).c_str());
            write_expr(n->lhs, f);
            fprintf(f, ")");
          }
        } else if constexpr (std::is_same_v<T, NodeBinary *>) {
          fprintf(f, "(");
          write_expr(n->lhs, f);
          fprintf(f, " %s ", to_string(n->op).c_str());
          write_expr(n->rhs, f);
          fprintf(f, ")");
        }
      },
      node);
}

float (*codegen(const GenericNode &node))(float, float, float) {
  static std::string last_so = "";
  int rand_id = rand();
  std::string base = ".gen_" + std::to_string(rand_id);
  std::string c_file = ".code.c";
  std::string so_file = base + ".so";
  if (!last_so.empty()) {
    std::remove(last_so.c_str());
    std::remove(c_file.c_str());
  }

  FILE *f = fopen(c_file.c_str(), "w");
  if (!f) {
    perror("fopen");
    return nullptr;
  }
  fprintf(f, "#include <math.h>\n");
  fprintf(f, "float func(float x, float y, float t) {\n");
  fprintf(f, "    return ");
  write_expr(node, f);
  fprintf(f, ";\n}\n");
  fclose(f);

  std::string cmd =
      "gcc -fPIC -O3 -shared -march=native " + c_file + " -o " + so_file;
  int ret = system(cmd.c_str());
  if (ret != 0) {
    fprintf(stderr, "Compilation failed\n");
    return nullptr;
  }
  void *handle = dlopen(so_file.c_str(), RTLD_NOW);
  if (!handle) {
    fprintf(stderr, "dlopen error: %s\n", dlerror());
    return nullptr;
  }

  void *sym = dlsym(handle, "func");
  if (!sym) {
    fprintf(stderr, "dlsym error: %s\n", dlerror());
    dlclose(handle);
    return nullptr;
  }
  last_so = so_file;
  return reinterpret_cast<float (*)(float, float, float)>(sym);
}

float wrap(float x, float min, float max) {
  const float range = max - min;
  return min + std::fmod((x - min + range), range);
}

template <std::floating_point T, typename F>
static constexpr auto rk4(T yn, T tn, T h, F &&f) -> T {
  const auto k1 = f(tn, yn);
  const auto k2 = f(tn + h / T(2.0), yn + h * k1 / T(2.0));
  const auto k3 = f(tn + h / T(2.0), yn + h * k2 / T(2.0));
  const auto k4 = f(tn + h, yn + h * k3);
  const auto yn1 = yn + (h / T(6.0)) * (k1 + T(2.0) * k2 + T(2.0) * k3 + k4);
  return yn1;
}

Vector3 getAttractor(Vector3 r0, float dt) {
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

struct MusicLorentzOscillator {
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
MusicLorentzOscillator music_osc{};

int demo(Scene *sc, BumpAllocator *arena, BumpAllocator *music_arena,
         float dur) {
  (void)music_arena;
  const int screenWidth = GetScreenWidth();
  const int screenHeight = GetScreenHeight();
  constexpr int depth = 8;
  arena->release();
  auto ast = generate_random_ast_arena(depth, arena);
  auto paint_fn = codegen(ast);
  float actual_time = 0.0;
  
  Sound snd = LoadSoundFromWave(sc->wave);
  PlaySound(snd);
  Color *const colors = ((Color *)sc->img.data);
  while (!WindowShouldClose() && actual_time < dur) {
#pragma omp parallel for
    for (int y = 0; y < screenHeight; ++y) {
#pragma omp simd
      for (int x = 0; x < screenWidth; ++x) {
        const float xx = 2.0f * (x / (float)screenWidth) - 1.0f;
        const float yy = 2.0f * (y / (float)screenHeight) - 1.0f;
        const auto r =
            wrap(paint_fn(xx, yy, sc->time / 5.0f) + 1.0f, 0.0f, 1.0f);
        const auto cc = (unsigned char)(r * 255.0f);
        // ImageDrawPixel(&sc->img, x, y, Color{cc, cc, cc, cc});
        colors[x + y * screenWidth] = Color{cc, cc, cc, 255};
      }
    }
    if (sc->time > 5) {
      sc->time = 0.0;
      arena->release();
      ast = generate_random_ast_arena(depth, arena);
      paint_fn = codegen(ast);
    }
    sc->time += GetFrameTime() / 2;
    actual_time += GetFrameTime();
    UpdateTexture(sc->tex, sc->img.data);
    BeginDrawing();
    // ClearBackground(RAYWHITE);
    DrawTexture(sc->tex, 0, 0, WHITE);
    DrawFPS(0, 0);
    EndDrawing();
  }
  UnloadSound(snd);
  return 0;
}

const char *intro_texts(float t) {
  if (t < 4.0) {
    return "Graffathon 2025!";
  }
  if (t >= 4.0 && t < 8.0) {
    return "Abstract Math!";
  }
  if (t >= 8.0 && t < 12.0) {
    return "By GreenHouse!";
  }
  if (t >= 12.0 && t < 20.0) {
    return "..in only 22KB!";
  }
  return nullptr;
}


void intro(Scene *sc, BumpAllocator *arena, float dur) {
  (void)arena;
  (void)sc;
  const float W = GetScreenWidth();
  const float H = GetScreenHeight();
  float actual_time = 0.0;
  Sound snd = LoadSoundFromWave(sc->wave);
  PlaySound(snd);
  constexpr float dt = 1.5 * 1e-2;
  Vector3 p1{1.0f, 0.0f, 0.0};
  Vector3 p2{0.8f, 0.0f, 0.0};
  std::size_t point_counter = 0;
  Vector2 *points = arena->allocate<Vector2>(1 < 14);
  while (!WindowShouldClose() && actual_time < dur) {
    ClearBackground(BLACK);
    auto msg = intro_texts(actual_time);
    if (!msg) {
      return;
    }
    int text_width = MeasureText(msg, 100);
    BeginDrawing();
    auto ps1 = Vector3Scale(p1, 16.0f);
    auto ps2 = Vector3Scale(p2, 16.0f);
    Vector2 cand1 = Vector2{ps1.z + (W / 2), ps1.y + (H / 2)};
    Vector2 cand2 = Vector2{ps2.z + (W / 2), ps2.y + (H / 2)};
    points[++point_counter] = cand1;
    points[++point_counter] = cand2;
    for (std::size_t i = 0; i < point_counter; ++i) {
      DrawCircleV(points[i], 3, i % 2 == 0 ? RED : RAYWHITE);
    }
    p1 = getAttractor(p1, dt / 1);
    p2 = getAttractor(p2, dt / 1);
    DrawText(msg, W / 4.0f - 0.5 * text_width, H / 2.0f, 100, GOLD);
    actual_time += GetFrameTime();
    EndDrawing();
  }
  UnloadSound(snd);
  arena->release();
}

void outro(Scene *sc, BumpAllocator *arena, float dur) {
  (void)arena;
  (void)sc;
  const float W = GetScreenWidth();
  const float H = GetScreenHeight();
  float actual_time = 0.0;
  Sound snd = LoadSoundFromWave(sc->wave);
  PlaySound(snd);
  const char *msg = "See you next year!";
  int text_width = MeasureText(msg, 100);
  while (!WindowShouldClose() && actual_time < dur) {
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText(msg, W / 2.0f - 0.5 * text_width, H / 2.0f, 100, ORANGE);
    actual_time += GetFrameTime();
    EndDrawing();
  }
  UnloadSound(snd);
}


// Singature has to be (float,....)
float custom_track_0(float t, const std::array<float, 3> &freqs) {
  float out = 0.0f;
  for (auto f : freqs) {
    out += sinf(2.0f * M_PI * f * t);
  }
  return out / freqs.size();
}

float beat(float t) {
  float beat_freq = 4.0f;
  float base_freq = 32.0f + 16.0f * std::sin(M_PI * t);
  return std::cos(2.0f * M_PI * beat_freq * t) *
         std::sin(2.0f * M_PI * base_freq * t);
}

float lorenz_track(float t) {
  float freq = music_osc.step();
  return 0.7 * std::sin(2.0f * M_PI * freq * t) + 0.3 * beat(t);
}

template<typename F>
float ast_track(float time,F&&f) {
  return 0.7 * std::sin(2.0f * M_PI *100+f(time,time,time) * time) ;
}


template <typename F, typename... Args>
Wave generate_music(float duration, BumpAllocator *arena, F &&f,
                    Args &&...args) {
  using music_type_t = short;
  constexpr std::size_t sampleRate = 48000;
  const std::size_t sampleCount =
      static_cast<std::size_t>(duration * sampleRate);
  music_type_t *samples = arena->allocate<music_type_t>(sampleCount);

  for (std::size_t i = 0; i < sampleCount; i++) {
    float t = static_cast<float>(i) / sampleRate;
    float val = f(t, std::forward<Args>(args)...);
    val = std::clamp(val, -1.0f, 1.0f);
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
+ Due to memory pool usage waves do not need to be unloaded
*/

extern "C" {
int jump_start() {
  // Setttings
  constexpr int seed = 1444; // Adam no touch! (0:142)
  constexpr std::size_t SCENE_MEMORY_POOL =  1024ul * 1024ul;
  constexpr std::size_t MUSIC_MEMORY_POOL = 1024ul * 1024ul * 1024ul;
  constexpr int screenWidth = 2 * 72 * 16;
  constexpr int screenHeight = 2 * 72 * 9;
  constexpr int FPS = 60;
  constexpr float intro_dur = 20.0f;
  constexpr float demo_dur = 96.0f;
  constexpr float outro_dur = 4.0f;
  //~Settings
  srand(seed);

  // Pool
  void *scene_memory = malloc(SCENE_MEMORY_POOL);
  void *music_memory = malloc(MUSIC_MEMORY_POOL);
  if (!scene_memory || !music_memory) {
    fprintf(stderr,
            "ERROR: Could not allocate any memory for the damn pool!\n");
    return 1;
  }
  BumpAllocator scene_arena(scene_memory, SCENE_MEMORY_POOL);
  BumpAllocator music_arena(music_memory, MUSIC_MEMORY_POOL);

  // clang-format off
  SetTraceLogLevel(TraceLogLevel::LOG_NONE);
  InitWindow(screenWidth, screenHeight, "");
  SetExitKey(KEY_ESCAPE);
  InitAudioDevice();
        //Setup Scene
        Scene scene{.img=GenImageColor(screenWidth, screenHeight, BLACK),.tex={},.time=0.0,.wave={}};
        scene.tex = LoadTextureFromImage(scene.img);
       
        //Intro 
        scene.wave = generate_music(intro_dur,&music_arena,lorenz_track);
        SetTargetFPS(FPS);
        intro(&scene, &scene_arena, 1);
        music_arena.release();
        scene.wave={};

        // Main demo with AST
        scene.wave = generate_music(demo_dur,&music_arena,lorenz_track);
        demo(&scene, &scene_arena,&music_arena,demo_dur);
        music_arena.release();

        //Outro
        scene.wave = generate_music(outro_dur,&music_arena,lorenz_track);
        outro(&scene, &scene_arena, outro_dur);
        music_arena.release();
  UnloadImage(scene.img);
  UnloadTexture(scene.tex);
  CloseWindow();
  // clang-format on
  scene_arena.destroy_with(free);
  music_arena.destroy_with(free);
  return 0;
}
}

#ifdef WITH_MAIN
int main() { return jump_start(); }
#endif
