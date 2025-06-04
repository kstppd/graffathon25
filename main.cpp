#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <math.h>
#include <omp.h>
#include <raylib.h>
#include <stdio.h>
#include <string>
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

int demo(int screenWidth, int screenHeight, float &time, BumpAllocator *arena) {
  Image img = GenImageColor(screenWidth, screenHeight, BLACK);
  Texture2D texture = LoadTextureFromImage(img);
  constexpr int depth = 8;
  arena->release();
  auto ast = generate_random_ast_arena(depth, arena);
  auto paint_fn = codegen(ast);
  while (!WindowShouldClose()) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < screenHeight; ++y) {
      for (int x = 0; x < screenWidth; ++x) {
        const float xx = 2 * (x / (float)screenWidth) - 1.0;
        const float yy = 2 * (y / (float)screenHeight) - 1.0;
        const auto r = wrap(paint_fn(xx, yy, time / 5.0f) + 1.0f, 0.0f, 1.0f);
        const Color c = {(unsigned char)(r * 255.0f),
                         (unsigned char)(r * 255.0f),
                         (unsigned char)(r * 255.0f), 255};
        ImageDrawPixel(&img, x, y, c);
      }
    }
    if (time > 5) {
      time = 0.0;
      arena->release();
      ast = generate_random_ast_arena(depth, arena);
      paint_fn = codegen(ast);
    }
    time += GetFrameTime() / 2;
    UpdateTexture(texture, img.data);
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawTexture(texture, 0, 0, WHITE);
    DrawFPS(0, 0);
    EndDrawing();
  }
  UnloadImage(img);
  UnloadTexture(texture);
  return 0;
}

void intro() {}
void outro() {}

extern "C" {
int jump_start() {
  // Setttings
  constexpr std::size_t MEMORY_POOL = 4ul*1024ul*1024ul;
  const int screenWidth = 72 * 16;
  const int screenHeight = 72 * 9;
  constexpr int FPS = 60;

  const int seed = 142;
  srand(seed);

  // Pool
  void *memory = malloc(MEMORY_POOL);
  if (!memory) {
    fprintf(stderr,
            "ERROR: Could not allocate any memory for the damn pool!\n");
    return 1;
  }
  BumpAllocator arena(memory, MEMORY_POOL);

  float time = 0.0f;
  SetTraceLogLevel(TraceLogLevel::LOG_NONE);
  InitWindow(screenWidth, screenHeight, "");
  intro();
  SetTargetFPS(FPS);
  demo(screenWidth, screenHeight, time, &arena);
  outro();
  CloseWindow();
  if (memory) {
    free(memory);
  }
  return 0;
}
}

#ifdef WITH_MAIN
int main() { return jump_start(); }
#endif
