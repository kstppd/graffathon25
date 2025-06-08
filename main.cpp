#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>
#include <raylib.h>

// Global handle
float *fft_data;
size_t fft_size;
size_t ncallbacks = 0;
bool show_fft = false;
//! Global handle

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

BumpAllocator *global_arena;

// MUSIC Stuff
using fft_type_t = float;
struct ComplexNum {
  fft_type_t real() { return r; }
  fft_type_t img() { return i; }
  fft_type_t r;
  fft_type_t i;
};

static constexpr inline bool isPow2(unsigned int val) noexcept {
  return (val & (val - 1)) == 0;
}

static constexpr inline uint32_t nextPow2(uint32_t v) noexcept {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

constexpr int FONTSIZE = 100;
constexpr float AST_INTERVAL = 8.0f;
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
constexpr float SR = 44100.0f;
constexpr float NYQUIST = SR / 2;
constexpr float MAX_HARMONICS = 12;

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
  const int max_harmonic = fmin(NYQUIST / fundamental, MAX_HARMONICS);
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
  const int max_harmonic = fmin(NYQUIST / fundamental, MAX_HARMONICS);
  float value = 0.0f;
  for (int k = 1; k <= max_harmonic; ++k) {
    value += sinf(2.0f * M_PI * fundamental * k * t) / k;
  }
  value *= -2.0f / M_PI;
  return value;
}

struct LowPassFilter {
  float prev = 0.0f;
  float alpha = 0.1f;

  LowPassFilter(float cutoff) { setCutoff(cutoff); }

  void setCutoff(float cutoff) {
    float rc = 1.0f / (2.0f * M_PI * cutoff);
    float dt = 1.0f / SR;
    alpha = dt / (rc + dt);
  }

  float process(float input) {
    prev = prev + alpha * (input - prev);
    return prev;
  }
};

struct HighPassFilter {
  float prev_input = 0.0f;
  float prev_output = 0.0f;
  float alpha = 0.1f;

  HighPassFilter(float cutoff) { setCutoff(cutoff); }

  void setCutoff(float cutoff) {
    float rc = 1.0f / (2.0f * M_PI * cutoff);
    float dt = 1.0f / SR;
    alpha = rc / (rc + dt);
  }

  float process(float input) {
    float output = alpha * (prev_output + input - prev_input);
    prev_input = input;
    prev_output = output;
    return output;
  }
};

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

void tiny_fft(ComplexNum *signal, size_t N) {

  // We done
  if (N <= 1) {
    return;
  }

  // Split into even and odd samples
  ComplexNum *even = global_arena->allocate<ComplexNum>(N / 2);
  ComplexNum *odd = global_arena->allocate<ComplexNum>(N / 2);
  for (size_t i = 0; i < N / 2; ++i) {
    even[i] = signal[2 * i];
    odd[i] = signal[2 * i + 1];
  }

  // Recursion hell
  tiny_fft(even, N / 2);
  tiny_fft(odd, N / 2);

  for (size_t k = 0; k < N / 2; ++k) {
    const float angle = -2.0f * PI * k / N;
    const float wr = cosf(angle);
    const float wi = sinf(angle);
    const float t_real = wr * odd[k].r - wi * odd[k].i;
    const float t_imag = wr * odd[k].i + wi * odd[k].r;
    signal[k].r = even[k].r + t_real;
    signal[k].i = even[k].i + t_imag;
    signal[k + N / 2].r = even[k].r - t_real;
    signal[k + N / 2].i = even[k].i - t_imag;
  }
  return;
}

void apply_hann(ComplexNum *signal, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    signal[i].r *= (1.0f / 2.0f) * (1.0f - cosf(2.0f * PI * i / (N - 1)));
  }
}

static HighPassFilter hp_filter_lead(100.0f); // Set your cutoff in Hz
static LowPassFilter lp_filter_kick(400.0f);  // Set your cutoff in Hz

// IVAN
static void callback(void *buffer, unsigned int frames) {
  static float demo_time = 0.0f;
  static float dt = 1.0f / SR;
  static float prev = 0;
  float *const d = reinterpret_cast<float *>(buffer);
  bool do_fft = false;
  ComplexNum *fft_buffer = global_arena->allocate<ComplexNum>(frames);

  for (unsigned int i = 0; i < frames; ++i) {
    float sample = 0.0f;
    const float bar_time = fmodf(demo_time, 4.0f);
    if (demo_time < 8.0) {
      for (int b = 0; b < 8; ++b) {
        float onset1 = b * 0.5f;
        float env1 = adsr(bar_time, ADSR(onset1, 0.5f, 0.5f, 0.05f, 0.2f,
                                         0.05f)); // Lead Synth tune
        sample += 0.1f * env1 * osc_square(demo_time, NOTE_AS / 4 - 20);
      }
    }

    /// 2nd part // lead + kick // up to 24.0
    else if (demo_time < 31.0) {
      for (int b = 0; b < 8; ++b) {
        float onset1 = b * 0.5f;
        float onset2 = b * 0.25f;
        float env2 = adsr(bar_time, ADSR(onset2, 0.5f, 0.005f, 0.004f, 0.001f,
                                         0.003f)); // Kick tune
        sample += 2.0f * env2 * osc_sawtooth(demo_time, NOTE_A / 4 - 20);
        // sample = lp_filter_kick.process(sample);
        sample = LPF(sample, prev, alpha(400.0f));
        float env1 = adsr(bar_time, ADSR(onset1, 0.5f, 0.5f, 0.05f, 0.2f,
                                         0.05f)); // Lead Synth tune
        sample += 0.1f * env1 * osc_square(demo_time, NOTE_AS / 4 - 20);
      }
    } else if (demo_time < 42.0) {
      for (int b = 0; b < 8; ++b) {
        float onset1 = b * 0.5f;
        float onset2 = b * 0.25f;
        float env2 = adsr(bar_time, ADSR(onset2, 0.5f, 0.005f, 0.004f, 0.001f,
                                         0.003f)); // Kick tune
        sample += 2.0f * env2 * osc_sawtooth(demo_time, NOTE_A / 4 - 20);
        sample = LPF(sample, prev, alpha(400.0f));
        float env3 = adsr(bar_time, ADSR(onset2, 0.4f, 0.005f, 0.004f, 0.2f,
                                         0.003f)); // Mario 8bit pitch slide
        // sample += 0.3f * env3 * osc_square(demo_time,
        // NOTE_DS+0.03*onset2*NOTE_DS );
        sample += 0.8f * env3 * osc_sawtooth(demo_time, NOTE_A + 4 * b);
        float env1 = adsr(bar_time, ADSR(onset1, 0.5f, 0.5f, 0.05f, 0.2f,
                                         0.05f)); // Lead Synth tune
        sample += 0.1f * env1 * osc_square(demo_time, NOTE_AS / 4 - 20);
      }
      /// Cental part 1 // lead + kick  // after 42 and up to 90
    } else if (demo_time < 900.0) {

      for (int b = 0; b < 4; ++b) {
        float onset2 = b * 0.75f;
        float env_fat = adsr(bar_time, ADSR(onset2, 0.4f, 0.005f, 0.05f, 0.3f,
                                            0.2f)); // FAT synth tune
        sample += 0.4f * env_fat * osc_square(demo_time, NOTE_B / 4 - 62);
        if (demo_time > 50.0) { // if more than 50.0 !!!!!
          float onset_kick = b * 0.5f;
          float env_kick =
              adsr(bar_time, ADSR(onset_kick, 0.75f, 0.005f, 0.005f, 0.002f,
                                  0.005f)); // Kick tune
          sample += 1.7f * env_kick * osc_sawtooth(demo_time, NOTE_A / 4);
        }
        if (demo_time > 58.0) { // if more than 58.0 !!!!!
          float onset_noiz = b * 1.0f + 0.25;
          float env_kick = adsr(bar_time, ADSR(onset_noiz, 0.75f, 0.04f, 0.005f,
                                               0.003f, 0.005f)); // noise1
          sample += 1.0f * env_kick * noise();
        }
        if (demo_time > 66.0) { // if more than 66.0 !!!!!
          float onset_crush = b * 1.0f + 0.5;
          float env_crush =
              adsr(bar_time, ADSR(onset_crush, 0.075f, 0.04f, 0.05f, 0.03f,
                                  0.005f)); // melodic noise
          sample += 1.f * env_crush * noise();
          sample += 1.f * env_crush * osc_sawtooth(demo_time, NOTE_B / 4);
        }
        if (demo_time > 74.0) { // if more than 74.0 !!!!!
          float onset_crush2 = b * 1.0f + 0.6;
          float env_crush2 =
              adsr(bar_time, ADSR(onset_crush2, 0.075f, 0.04f, 0.05f, 0.03f,
                                  0.005f)); // melodic noise 2
          sample += 1.f * env_crush2 * noise();
          sample += 1.f * env_crush2 * osc_sawtooth(demo_time, NOTE_D / 4);
        }
      } // end of b loop
    }else if (demo_time < 98.0) { // FINAL PART 1
      for (int b = 0; b < 4; ++b) {
        float onset_crush1 = b * 1.0f + 0.5;
        float onset_crush2 = b * 1.0f + 0.6;
        // float onset_lead_central = b * 1.0f + 0.75;
        float onset_kick = b * 0.5f;
        float env_kick = adsr(bar_time, ADSR(onset_kick, 0.75f, 0.005f, 0.005f,
                                             0.002f, 0.005f)); // Kick tune
        sample += 1.7f * env_kick * osc_sawtooth(demo_time, NOTE_A / 4);
        float env_crush1 =
            adsr(bar_time, ADSR(onset_crush1, 0.075f, 0.04f, 0.05f, 0.03f,
                                0.005f)); // melodic noise 2
        sample += 1.f * env_crush1 * noise();
        sample += 1.f * env_crush1 * osc_sawtooth(demo_time, NOTE_B / 4);
        float env_crush2 =
            adsr(bar_time, ADSR(onset_crush2, 0.075f, 0.04f, 0.05f, 0.03f,
                                0.005f)); // melodic noise 2
        sample += 1.f * env_crush2 * noise();
        sample += 1.f * env_crush2 * osc_sawtooth(demo_time, NOTE_D / 4);
      }
    }else if (demo_time < 98.0) { // FINAL PART 2
      for (int b = 0; b < 4; ++b) {
        float onset_crush1 = b * 1.0f + 0.5;
        // float onset_crush2 = b * 1.0f + 0.6;
        // float onset_lead_central = b * 1.0f + 0.75;
        float onset_kick = b * 0.5f;
        float env_kick = adsr(bar_time, ADSR(onset_kick, 0.75f, 0.005f, 0.005f,
                                             0.002f, 0.005f)); // Kick tune
        sample += 1.7f * env_kick * osc_sawtooth(demo_time, NOTE_A / 4);
        float env_crush1 =
            adsr(bar_time, ADSR(onset_crush1, 0.075f, 0.04f, 0.05f, 0.03f,
                                0.005f)); // melodic noise 2
        sample += 1.f * env_crush1 * noise();
        sample += 1.f * env_crush1 * osc_sawtooth(demo_time, NOTE_B / 4);
      }
    }else if (demo_time < 106.0) { // FINAL PART 3
      for (int b = 0; b < 4; ++b) {
        // float onset_crush1 = b * 1.0f + 0.5;
        // float onset_crush2 = b * 1.0f + 0.6;
        // float onset_lead_central = b * 1.0f + 0.75;
        float onset_kick = b * 0.5f;
        float env_kick = adsr(bar_time, ADSR(onset_kick, 0.75f, 0.005f, 0.005f,
                                             0.002f, 0.005f)); // Kick tune
        sample += 1.7f * env_kick * osc_sawtooth(demo_time, NOTE_A / 4);
      }
    }else if (demo_time < 112.0) { // FINAL PART 4
      for (int b = 0; b < 4; ++b) {
        // float onset_crush1 = b * 1.0f + 0.5;
        // float onset_crush2 = b * 1.0f + 0.6;
        // float onset_lead_central = b * 1.0f + 0.75;
        float onset_kick = b * 0.5f;
        float env_kick = adsr(bar_time, ADSR(onset_kick, 0.75f, 0.005f, 0.005f,
                                             0.002f, 0.005f)); // Kick tune
        sample += 1.7f * env_kick * osc_sawtooth(demo_time, NOTE_A / 4);
      }
  }else if (demo_time < 120.0) { // FINAL PART 5
      for (int b = 0; b < 4; ++b) {
        float onset_kick = b * 0.5f;
        float onset_synth = 0.001f;
        float env_kick = adsr(bar_time, ADSR(onset_kick, 0.75f, 0.005f, 0.005f,
                                             0.002f, 0.005f)); // Kick tune
        sample += 1.7f * env_kick * osc_sawtooth(demo_time, NOTE_A / 4);
        float env1 = adsr(bar_time, ADSR(onset_synth, 0.01f, 0.01f, 0.05f, 0.2f,
                                         0.5f)); // Lead Synth tune
        sample += 0.1f * env1 * osc_square(demo_time, NOTE_AS / 4 - 20);
      }
    }
    sample = clamp(sample, -1.0f, 1.0f);
    d[2 * i] = sample;
    d[2 * i + 1] = sample;
    fft_buffer[i].r = sample;
    fft_buffer[i].i = 0;
    if (sample > 0.1 && !do_fft) {
      do_fft = true;
    }
    demo_time += dt;
  }

  if (do_fft) {
    apply_hann(fft_buffer, frames);
    tiny_fft(fft_buffer, frames);
    for (unsigned int i = 0; i < 1 + frames / 2; ++i) {
      fft_data[i] = fft_buffer[i].r;
    }
    fft_data[0] = 0.0f; // nuke dc
    fft_size = 1 + frames / 2;
  }
  global_arena->release();
  ncallbacks++;
}

//~Music stuff

// AST Stuff
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

static const char *codegen_glsl_ast_marcher(const GenericNode &node) {
  static char buffer[50 * 65536];
  char *out = buffer;
  out += sprintf(out, "#version 330 core\n"
                      " precision lowp float;\n"
                      "#define PI 3.14159265\n"
                      "out vec4 FragColor;"
                      "uniform float angleOffset;"
                      "uniform vec2 resolution;"
                      "uniform float mag;"
                      "uniform float time;"
                      "uniform float ast_time;"
                      "uniform int torus;"
                      "uniform int axis;"
                      "uniform int color;"
                      "uniform int wrap;"
                      "vec3 sample_ast(vec3 pos) {"
                      "    float x = pos.x;"
                      "    float y = pos.y;"
                      "    float t = ast_time/5.0;"
                      "    float v = ");
  out = write_expr(node, out);
  out += sprintf(
      out,
      ";"
      "    float r = v * 1.2 + 0.0;"
      "    float g = v * 1.1 + 0.33;"
      "    float b = v * -1.5 + 0.67;"
      "    vec3 retval = (color==1)?clamp(vec3(100*r, g, 100*b), 0.0, 1.0)  "
      ":clamp(vec3(128,128, 128), 0.0, 1.0); "
      "return retval;"
      // "return clamp(vec3(100*r, 100*r, 100*r), 0.0, 1.0);"
      "}"
      "vec3 rotateY(vec3 p, float a) {"
      "    float c = cos(a), s = sin(a);"
      "    return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);"
      "}"
      "vec3 rotateX(vec3 p, float a) {"
      "    float c = cos(a), s = sin(a);"
      "    return vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z);"
      "}"
      "vec3 rotateAroundAxis(vec3 v, vec3 k, float theta){"
      "return v * cos(theta) + cross(k, v) * sin(theta)+ k * dot(k, v) * (1.0 "
      "- cos(theta));"
      "}"
      "float sdfMandel(vec3 point0) {"
      "    const vec3 SCENE_CENTER = vec3(0.0, 0.0, 0.0);"
      "    const float  BLOB_FACTOR  = 4.0;               "
      "    const float  SPIKE_ANGLE  = 1.5707/8;     "
      "    const float  POWER         = 5.0;              "
      "    const float  BAILOUT       = 2.0;"
      "    vec3 point = point0 - SCENE_CENTER;"
      "    vec3  z    = point;"
      "    float dr   = 1.0;"
      "    float dist = 0.0;"
      "    for (int i = 0; i < 4; i++) {"
      "        dist = length(z);"
      "        if (dist > BAILOUT) {break;}"
      "        float theta = acos(z.z / dist) * POWER * BLOB_FACTOR;"
      "        float phi   = atan(z.y, z.x) * POWER;"
      "        float distPowM1 = pow(dist, POWER - 1.0);"
      "        float zr        = distPowM1 * dist;"
      "        dr = distPowM1 * POWER * dr + 1.0;"
      "        float sinTh = sin(theta);"
      "        z = zr * vec3(sinTh * cos(phi),sin(phi + SPIKE_ANGLE) * "
      "sinTh,cos(theta));"
      "        z += point;"
      "        }"
      "    return 0.5 * log(dist) * dist / dr;"
      "    }"
      "vec3 tile(vec3 p, float S) {"
      "    return mod(p + S*0.5, S) - S*0.5;"
      "}"
      "float sdBox(vec3 p, vec3 b) {"
      "    vec3 d = abs(p) - b;"
      "    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));"
      "}"
      "float sdTorus(vec3 p, vec2 t) {"
      "    vec2 q = vec2(length(p.xz) - t.x, p.y);"
      "    return length(q) - t.y;"
      "}"
      "float sdSphere(vec3 p, float r) {"
      "    return length(p) - r;"
      "}"
      "float smin(float a, float b, float k) {"
      "    float h = clamp(0.5 + 0.5*(b - a)/k, 0.0, 1.0);"
      "    return mix(b, a, h) - k*h*(1.0 - h);"
      "}"
      "float sceneSDF(vec3 p) {"
      "    const float tileSize = 2.5;"
      "    vec3 q =(wrap==1)? tile(p, tileSize):p;"
      "    q = rotateY(q, time * 1*0.8);"
      "    q = rotateX(q, time * 1*0.6);"
      "    float dBox = (torus==0)?sdBox(q, vec3(0.5)):sdTorus(q, "
      "vec2(0.5,0.25));"
      "    float dS1  = sdSphere(q + vec3(mag*sin(time*2.0), 0, 0), 0.25);"
      "    float dS2  = sdSphere(q + vec3(0, mag*cos(time*2.0), 0), 0.25);"
      "    float dS3  = sdSphere(q + vec3(0, 0, mag*sin(time*2.0)), 0.25);"
      "    float u1   = smin(dBox, dS1, 0.3);"
      "    float u2   = smin(dS2, u1, 0.3);"
      "    float u3   = smin(dS3, u2, 0.3);\n"
      "    float dM = sdfMandel(p);"
      "    return smin(dM, u3, 0.3);"
      "}"
      "vec3 getNormal(vec3 p) {"
      "    const float epsilon = 1e-4;"
      "    return normalize(vec3("
      "        sceneSDF(p + vec3(epsilon,0,0)) - sceneSDF(p - "
      "vec3(epsilon,0,0)),"
      "        sceneSDF(p + vec3(0,epsilon,0)) - sceneSDF(p - "
      "vec3(0,epsilon,0)),"
      "        sceneSDF(p + vec3(0,0,epsilon)) - sceneSDF(p - "
      "vec3(0,0,epsilon))"
      "    ));"
      "}"
      "float march(vec3 ro, vec3 rd) {"
      "    float t = 0.0;"
      "    for(int i=0;i<100;i++) {"
      "        vec3 p = ro + rd*t;"
      "        float d = sceneSDF(p);"
      "        if(d < 1e-3) break;"
      "        t += d;"
      "        if(t > 100.0) break;"
      "    }"
      "    return t;"
      "}"
      "void main() {"
      "    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution) / resolution.y;"
      "    float camAng = time * 0.3;"
      " float speed = 0.5;"
      "float theta = time * speed;"
      "vec3 baseRo = vec3(0.0, 0.0, 4.0);"
      "vec3 raxis=vec3(1,1,0);"
      "if (axis==0){raxis=vec3(1.0,0.0,0.0);}"
      "if (axis==1){raxis=vec3(0.0,1.0,0.0);}"
      "if (axis==2){raxis=vec3(0.0,0.0,1.0);}"
      "if (axis==3){raxis=vec3(1.0,1.0,0.0);}"
      "if (axis==4){raxis=vec3(1.0,0.0,1.0);}"
      "if (axis==5){raxis=vec3(0.0,1.0,1.0);}"
      "if (axis==6){raxis=vec3(1.0,1.0,1.0);}"
      "vec3 ro = rotateAroundAxis(baseRo, normalize(raxis), theta);"
      // "    vec3 ro = vec3(4* sin(camAng), 4* sin(camAng), 2.0 *
      // cos(camAng));"
      "    vec3 fwd   = normalize(vec3(0.0) - ro);"
      "    vec3 right = normalize(cross(fwd, vec3(0,1,0)));"
      "    vec3 camUp = cross(right, fwd);"
      "    vec3 rd    = normalize(fwd + uv.x*right + uv.y*camUp);"
      "    float t = march(ro, rd);"
      "    if(t > 80.0) {\n"
      "        FragColor = vec4(0,0,0,1);\n"
      "    }\n else {"
      "        vec3 p       = ro + rd * t;"
      "        vec3 n       = getNormal(p);"
      "        vec3 light   = normalize(vec3(1.0, 1.0, 0.7));"
      "        vec3 viewDir = normalize(ro - p);"
      "        float ka     = 0.1;"
      "        float kd     = max(dot(n, light), 0.0);"
      "        vec3 halfVec = normalize(light + viewDir);"
      "        float ks     = pow(max(dot(n, halfVec), 0.0), 32.0);"
      "        const float tileSize = 2.5;"
      "        vec3 q = (wrap==1)? tile(p, tileSize):p;"
      "        q = rotateY(q, time * 0.8);"
      "        q = rotateX(q, time * 0.6);"
      "        vec3 baseCol = sample_ast(q);"
      "        vec3 color = baseCol * (ka + kd) + ks;"
      "        vec3 rd2   = reflect(rd, n);"
      "        vec3 ro2   = p + n * 1e-3;"
      "        float t2   = march(ro2, rd2);"
      "        if(t2 < 50.0) {"
      "            vec3 p2    = ro2 + rd2 * t2;"
      "            vec3 n2    = getNormal(p2);"
      "            float kd2  = max(dot(n2, light), 0.0);"
      "            vec3 half2 = normalize(light + normalize(ro2 - p2));"
      "            float ks2  = pow(max(dot(n2, half2), 0.0), 32.0);"
      "            vec3 col2  = vec3(ka + kd2 + ks2);"
      "            color      = mix(color, col2, 0.2);"
      "        }"
      "        FragColor = vec4(color, 1.0);"
      "    }"
      "}");
  return buffer;
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
    return "..in under 39 KB!";
  }
  if (t >= 10.0 && t < 12) {
    return " 39 KB? LMFAO! <3";
  }
  return nullptr;
}

void draw_real_fft() {
  if (!fft_data || fft_size == 0) {
    return;
  }
  const float W = GetScreenWidth();
  const float H = GetScreenHeight();
  const char *msg = "Fastest Fourier Transform in the North";
  const int sz = MeasureText(msg, 20);
  const float pad = 20;
  const float plotW = fmax(1.2 * sz, W * 0.2f);
  const float plotH = plotW / 4.5;
  const float plotX = 2 * pad;
  const float plotY = H - plotH - 2 * pad;
  const int ticksX = 5;
  const int ticksY = 4;

  const float minFreq = 20.0f;
  const float maxFreq = NYQUIST;
  const float logMin = log10f(minFreq);
  const float logMax = log10f(maxFreq);

  // Normalization By max
  float maxPower = -100000.0f;
  for (size_t i = 0; i < fft_size; ++i) {
    float mag = fft_data[i] * fft_data[i];
    if (mag > maxPower)
      maxPower = mag;
  }

  DrawRectangleRounded(
      (Rectangle){0, plotY - pad, plotW + 4 * pad, plotH + 2 * pad}, 0.4f, 1000,
      BLACK);
  DrawRectangleLinesEx((Rectangle){plotX, plotY, plotW, plotH}, 2, RAYWHITE);

  for (int i = 1; i < ticksX - 1; ++i) {
    const float t = i / (float)(ticksX - 1);
    const float f = powf(10.0f, logMin + t * (logMax - logMin));
    const float x = plotX + ((log10f(f) - logMin) / (logMax - logMin)) * plotW;
    DrawLine((int)x, (int)plotY, (int)x, (int)(plotY + plotH), DARKGRAY);
  }

  for (int i = 1; i < ticksY; ++i) {
    const float y = plotY + (i / (float)ticksY) * plotH;
    DrawLine((int)plotX, (int)y, (int)(plotX + plotW), (int)y, DARKGRAY);
  }

  // Log bins with smoothening
  constexpr int log_bins = 256;
  float logPowers[log_bins];
  int binCounts[log_bins] = {0};

  for (int i = 0; i < log_bins; ++i) {
    logPowers[i] = 0;
  }

  for (size_t i = 0; i < fft_size; ++i) {
    const float freq = (i / (float)(fft_size - 1)) * NYQUIST;
    if (freq < minFreq)
      continue;
    const float power = fft_data[i] * fft_data[i];
    const float logf = log10f(freq);
    const int bin = (int)((logf - logMin) / (logMax - logMin) * (log_bins - 1));
    if (bin >= 0 && bin < log_bins) {
      logPowers[bin] += power;
      binCounts[bin]++;
    }
  }

  for (int i = 0; i < log_bins; ++i) {
    if (binCounts[i] > 0)
      logPowers[i] /= binCounts[i];
    // clamp to 0.9 otherwise it looks like shit
    logPowers[i] = clamp(logPowers[i] / maxPower, 0.0f, 0.9f);
  }

  for (int i = 0; i < log_bins - 1; ++i) {
    const float x0 = plotX + i / (float)(log_bins - 1) * plotW;
    const float x1 = plotX + (i + 1) / (float)(log_bins - 1) * plotW;
    const float y0 = plotY + plotH * (1.0f - logPowers[i]);
    const float y1 = plotY + plotH * (1.0f - logPowers[i + 1]);
    // DrawLineEx((Vector2){x0, y0}, (Vector2){x1, y1}, 2.0f, GREEN);
    if (logPowers[i] > 0.2) {
      DrawLineEx((Vector2){x0, (plotY + plotH)}, (Vector2){x0, y0}, 1.0f, LIME);
      DrawLineEx((Vector2){x1, (plotY + plotH)}, (Vector2){x0, y0}, 1.0f, LIME);
      DrawCircleV(Vector2{x0, y0}, 3, GOLD);
      DrawCircleV(Vector2{x1, y1}, 3, GOLD);
    }
    // DrawPixel(x0, y0, RED);
  }

  // Title
  DrawText(msg, plotX + plotW / 2.0 - sz / 2.0f, plotY - 20, 20, RED);

  for (int i = 0; i < ticksX; ++i) {
    float t = i / (float)(ticksX - 1);
    float f = powf(10.0f, logMin + t * (logMax - logMin));
    char label[16];
    snprintf(label, sizeof(label), "%.0fHz", f);
    float x = plotX + ((log10f(f) - logMin) / (logMax - logMin)) * plotW - 10;
    DrawText(label, (int)x, (int)(plotY + plotH + 4), 20, LIGHTGRAY);
  }

  for (int i = 0; i <= ticksY - 1; ++i) {
    float p = 1.0f - (i / (float)ticksY);
    char label[16];
    snprintf(label, sizeof(label), "%.1f", p);
    float y = plotY + i * (plotH / ticksY) - 6 + pad;
    DrawText(label, (int)(plotX - 32), (int)y, 20, LIGHTGRAY);
  }
}

static size_t intro(Scene *sc, BumpAllocator *arena, float dur) {
  (void)arena;
  (void)sc;
  float actual_time = 0.0;
  constexpr float dt = 1 * 1e-2;
  Vector3 p1{1.0f, 0.0f, 0.0};
  Vector3 p2{0.8f, 0.0f, 0.0};
  Vector2 *points = arena->allocate<Vector2>(1 << 20);
  size_t point_counter = 0;
  while (!WindowShouldClose() && actual_time < dur) {
    const float W = GetScreenWidth();
    const float H = GetScreenHeight();
    BeginDrawing();
    auto msg = intro_texts(actual_time);
    if (!msg) {
      return point_counter;
    }
    for (size_t i = 0; i < 16 / 2; ++i) {
      auto ps1 = Vector3Scale(p1, 20.0f);
      auto ps2 = Vector3Scale(p2, 20.0f);
      Vector2 cand1 = Vector2{ps1.x + (W / 2) + (W / 8), ps1.y + (H / 2)};
      Vector2 cand2 = Vector2{ps2.x + (W / 2) + (W / 8), ps2.y + (H / 2)};
      points[point_counter] = Vector2{cand1.x / W, cand1.y / H};
      points[point_counter + 1] = Vector2{cand2.x / W, cand2.y / H};
      point_counter += 2;
      p1 = getAttractor(p1, dt / 2);
      p2 = getAttractor(p2, dt / 2);
    }
    ClearBackground(BLACK);
    for (size_t i = 0; i < point_counter; ++i) {
      DrawCircle(W * points[i].x, H * points[i].y, 1, i % 2 == 0 ? RED : BLUE);
    }
    DrawText(msg, 0.25*W, 0.5 * H +osc_sine(GetTime(), 0.2)*32*sinf(2.0*M_PI*0.5*GetTime()), FONTSIZE, ORANGE);
    if (show_fft) {
      draw_real_fft();
    }
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
    return "Our grammar for today: ";
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
    return "All that below 39 KB!\n\nLet's go!";
  }
  return nullptr;
}

static int demo_ast(Scene *sc, BumpAllocator *arena, BumpAllocator *music_arena,
                    float dur, int depth = 8, bool color = false);

static void post_intro(Scene *sc, BumpAllocator *arena, float dur) {
  (void)arena;
  (void)sc;

  float actual_time = 0.0;
  constexpr float dt = 1 * 1e-2;
  Vector3 p1{1.3f, 0.0f, 0.0};
  Vector3 p2{0.4f, 0.0f, 0.0};
  Vector2 *points = arena->allocate<Vector2>(1 << 20);
  bool flag = true;

  size_t point_counter = 0;
  while (!WindowShouldClose() && actual_time < dur) {
    const float W = GetScreenWidth();
    const float H = GetScreenHeight();
    BeginDrawing();
    // Spawn an AST and then revert actual time back
    if (actual_time > 26 && flag) {
      float store = actual_time;
      // Create a Pool from my Pool -> Insane!
      BumpAllocator tmp1(arena->allocate<float>(1024 * 1024),
                         1024 * 1024 * sizeof(float));
      flag = false;
      demo_ast(sc, arena, &tmp1, 2, 8);
      point_counter = 0;
      actual_time = store + GetFrameTime();
      continue;
    }
    auto msg = intro_post1_texts(actual_time);
    if (!msg) {
      return;
    }
    if (point_counter < 10000) {
      for (size_t i = 0; i < 16 / 2; ++i) {
        auto ps1 = Vector3Scale(p1, 18.0f);
        auto ps2 = Vector3Scale(p2, 18.0f);
        Vector2 cand1 = Vector2{ps1.z + (W / 2), ps1.x + (H / 2)};
        Vector2 cand2 = Vector2{ps2.z + (W / 2), ps2.x + (H / 2)};
        points[point_counter++] = Vector2{cand1.x / W, cand1.y / H};
        points[point_counter++] = Vector2{cand2.x / W, cand2.y / H};
        p1 = getAttractor(p1, dt / 2);
        p2 = getAttractor(p2, dt / 2);
      }
    }
    ClearBackground(BLACK);
    // Skip this for upcoming AST graphic
    if (!(actual_time > 20.0 && actual_time <= 26)) {
      for (size_t i = 0; i < point_counter; ++i) {
        DrawCircle(W * points[i].x, H * points[i].y, 1,
                   i % 2 == 0 ? GOLD : PINK);
      }
    }


    if (actual_time > 20.0 && actual_time <= 26) {
      const float anim_dt = 0.25f;
      float fontSize = 32;
      float rootX = 0.75f * W;
      float rootY = 0.25f * H;
      DrawEllipse(rootX, rootY, 64, 32, RED);
      const char *text = "sin";
      int textWidth = MeasureText(text, fontSize);
      DrawText(text, rootX - textWidth / 2.0f, rootY - fontSize / 2.0f +osc_sine(GetTime(), 0.1)*16*sinf(2.0*M_PI*0.5*GetTime()),
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
      if (actual_time < 19.5 + 3 * anim_dt) {
        goto checkpoint;
      }
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
    DrawText(msg, 32, 0.2 * H +osc_sine(GetTime(), 0.1)*16*sinf(2.0*M_PI*0.5*GetTime()), FONTSIZE, ORANGE);
    actual_time += GetFrameTime();
    if (show_fft) {
      draw_real_fft();
    }
    EndDrawing();
  }
  return;
}

static int demo_ast(Scene *sc, BumpAllocator *arena, BumpAllocator *music_arena,
                    float dur, int depth, bool color) {
  (void)music_arena;
  const int screenWidth = GetScreenWidth();
  const int screenHeight = GetScreenHeight();
  arena->release();
  auto ast = generate_random_ast_arena(depth, arena);
  auto glsl_str = (color) ? codegen_glsl_sawtooth(ast)
                          : codegen_glsl_sawtooth_grayscale(ast);
  float actual_time = 0.0;
  Shader shader = LoadShaderFromMemory(0, glsl_str);
  int locTime, locRes;
  locRes = GetShaderLocation(shader, "resolution");
  locTime = GetShaderLocation(shader, "time");
  float resolution[2] = {(float)screenWidth, (float)screenHeight};
  SetShaderValue(shader, locRes, resolution, SHADER_UNIFORM_VEC2);
  while (!WindowShouldClose() && actual_time < dur) {
    float t = fmodf(actual_time, AST_INTERVAL);
    SetShaderValue(shader, locTime, &t, SHADER_UNIFORM_FLOAT);
    BeginDrawing();
    ClearBackground(WHITE);
    BeginShaderMode(shader);
    DrawRectangle(0, 0, screenWidth, screenHeight, WHITE);
    EndShaderMode();
    sc->time += GetFrameTime() / 2;
    actual_time += GetFrameTime();
    if (show_fft) {
      draw_real_fft();
    }
    EndDrawing();
  }
  UnloadShader(shader);
  return 0;
}

static void demo_march(Scene *sc, BumpAllocator *arena,
                       BumpAllocator *music_arena, float dur, float mag = 1.0,
                       int torus = 1, int color = 0, int wrap = 0) {
  (void)music_arena;
  const int depth = 8;
  const int screenWidth = GetScreenWidth();
  const int screenHeight = GetScreenHeight();
  arena->release();
  auto ast = generate_random_ast_arena(depth, arena);
  auto glsl_str = codegen_glsl_ast_marcher(ast);
  float actual_time = 0.0;
  Shader shader = LoadShaderFromMemory(0, glsl_str);
  int locRes = GetShaderLocation(shader, "resolution");
  int locTime = GetShaderLocation(shader, "time");
  int locAstTime = GetShaderLocation(shader, "ast_time");
  int locMag = GetShaderLocation(shader, "mag");
  int locTorus = GetShaderLocation(shader, "torus");
  int locColor = GetShaderLocation(shader, "color");
  int locWrap = GetShaderLocation(shader, "wrap");
  int locOffset = GetShaderLocation(shader, "angleOffset");
  int locAxis = GetShaderLocation(shader, "axis");
  float offset = fmodf(GetTime() * (2 * PI / 10.0f), 2 * PI);
  int axis = ncallbacks % 7;
  float resolution[2] = {(float)screenWidth, (float)screenHeight};
  SetShaderValue(shader, locRes, resolution, SHADER_UNIFORM_VEC2);
  SetShaderValue(shader, locTorus, &torus, SHADER_UNIFORM_INT);
  SetShaderValue(shader, locMag, &mag, SHADER_UNIFORM_FLOAT);
  SetShaderValue(shader, locColor, &color, SHADER_UNIFORM_INT);
  SetShaderValue(shader, locWrap, &wrap, SHADER_UNIFORM_INT);
  SetShaderValue(shader, locRes, resolution, SHADER_UNIFORM_VEC2);
  SetShaderValue(shader, locOffset, &offset, SHADER_UNIFORM_FLOAT);
  SetShaderValue(shader, locAxis, &axis, SHADER_UNIFORM_INT);
  while (!WindowShouldClose() && actual_time < dur) {
    float t = GetTime();
    float ast_t = fmodf(t, AST_INTERVAL);
    SetShaderValue(shader, locTime, &t, SHADER_UNIFORM_FLOAT);
    SetShaderValue(shader, locAstTime, &ast_t, SHADER_UNIFORM_FLOAT);
    BeginDrawing();
    ClearBackground(WHITE);
    BeginShaderMode(shader);
    DrawRectangle(0, 0, screenWidth, screenHeight, WHITE);
    EndShaderMode();
    sc->time += GetFrameTime() / 2;
    actual_time += GetFrameTime();
    if (show_fft) {
      draw_real_fft();
    }
    EndDrawing();
  }
  UnloadShader(shader);
}

static void outro(Scene *sc, BumpAllocator *arena, float dur, int n) {
  (void)arena;
  (void)sc;

  float actual_time = 0.0;
  Vector2 *points = reinterpret_cast<Vector2 *>(arena->_memory);
  while (!WindowShouldClose() && actual_time < dur) {
    const float W = GetScreenWidth();
    const float H = GetScreenHeight();
    BeginDrawing();
    ClearBackground(BLACK);
    const char *msg = "See you next year!";
    if (actual_time > 0.75 * dur) {
      msg = "GreenHouse";
    }

    for (int i = n; i > 0; --i) {
      DrawCircle(W * points[i].x, H * points[i].y, 1, i % 2 == 0 ? RED : BLUE);
    }
    n -= 16;
    // DrawText(msg, 32, 0.2 * H, FONTSIZE, GOLD);
    DrawText(msg, 0.2*W, 0.5 * H +osc_sine(GetTime(), 0.2)*32*sinf(2.0*M_PI*0.5*GetTime()), FONTSIZE, ORANGE);
    actual_time += GetFrameTime();
    if (show_fft) {
      draw_real_fft();
    }
    EndDrawing();
  }
  arena->release();
}
//~Scene specific stuff and main demos

static void wait() {
  const char* txt="Press [SPACE] To Win 42 BTC!";
  while (!IsKeyPressed(KEY_SPACE) && !WindowShouldClose()) {
    BeginDrawing();
    const float W = GetScreenWidth();
    const float H = GetScreenHeight();
    auto sz = MeasureText(txt,FONTSIZE);
    ClearBackground(ORANGE);
    DrawText(txt, W/2-sz/2.0f, 32.0f*sinf(2*M_PI*1*GetTime())+ H/2, FONTSIZE, BLACK);
    EndDrawing();
  }
}

static void wait_more() {
  const char* txt="Yeeaaah...You Wish...LOL";
  float t = GetTime();
  while (GetTime()-t<5.0f) {
    BeginDrawing();
    const float W = GetScreenWidth();
    const float H = GetScreenHeight();
    auto sz = MeasureText(txt,FONTSIZE);
    ClearBackground(ORANGE);
    DrawText(txt, 32.0f*sinf(2*M_PI*1*GetTime())+W/2-sz/2.0f, 32.0f*cos(2*M_PI*1*GetTime())+ H/2, FONTSIZE, BLACK);
    EndDrawing();
  }
}
/*
NOTES
+Due to memory pool usage waves do not need to be unloaded
*/
extern "C" {
int jump_start() {
  // Setttings
  constexpr int seed = 256; // Adam no touch! (0:142) (1:512)
  constexpr size_t SCENE_MEMORY_POOL = 1024ul * 1024ul * 1024ul;
  constexpr size_t MUSIC_MEMORY_POOL = 1024ul * 1024ul * 1024ul;
  constexpr float intro_dur = 12.0f;
  constexpr float post_intro_1_dur = 31.0f;
  constexpr float outro_dur = intro_dur;
  constexpr int FPS = 60;
  //~Settings
  srand(seed);

  // Pools
  void *scene_memory = RL_MALLOC(SCENE_MEMORY_POOL);
  void *scratch_memory = RL_MALLOC(SCENE_MEMORY_POOL);
  void *music_memory = RL_MALLOC(MUSIC_MEMORY_POOL);
  void *fft_memory = RL_MALLOC(MUSIC_MEMORY_POOL);
  fft_data = (float *)RL_MALLOC(1024ul * 1024ul * sizeof(float));

  if (!scene_memory || !music_memory || !scratch_memory || !scratch_memory ||
      !fft_data) {
    return 1;
  }

  // Init pools
  BumpAllocator scene_arena(scene_memory, SCENE_MEMORY_POOL);
  BumpAllocator scratch_arena(scratch_memory, SCENE_MEMORY_POOL);
  BumpAllocator music_arena(music_memory, MUSIC_MEMORY_POOL);
  BumpAllocator fft_arena(fft_memory, MUSIC_MEMORY_POOL);
  global_arena = &fft_arena;

  // clang-format off
  SetTraceLogLevel(TraceLogLevel::LOG_NONE);
  InitWindow(GetScreenWidth(), GetScreenHeight(), "A.S.T Picasso - GreenHouse");
  SetWindowState(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_UNDECORATED);
  ToggleFullscreen();
  const int screenWidth  = 2*GetScreenWidth();
  const int screenHeight = 2*GetScreenHeight();;
  wait();
  wait_more();
  DisableCursor();
  SetExitKey(KEY_ESCAPE);
  InitAudioDevice();
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
          constexpr  int depth = 8;
          demo_ast(&scene, &scene_arena, &music_arena, 2,depth,false);
          scene_arena.release();
          show_fft=true;
          demo_march(&scene, &scene_arena, &music_arena, 8,0.5,0,0,0);
          scene_arena.release();
          demo_ast(&scene, &scene_arena, &music_arena, 2,depth,false);
          scene_arena.release();
          demo_march(&scene, &scene_arena, &music_arena, 16,1.0,0,0,1);
          scene_arena.release();
          show_fft=true;
          demo_ast(&scene, &scene_arena, &music_arena, 2,depth,true);
          scene_arena.release();
          demo_march(&scene, &scene_arena, &music_arena, 4,1.0,0,1,1);
          scene_arena.release();
          demo_ast(&scene, &scene_arena, &music_arena, 2,depth,true);
          scene_arena.release();
          demo_march(&scene, &scene_arena, &music_arena, 4,1.0,1,1,0);
          scene_arena.release();
          demo_ast(&scene, &scene_arena, &music_arena, 2,depth,true);
          scene_arena.release();
          demo_march(&scene, &scene_arena, &music_arena, 4,1.0,1,1,1);
          scene_arena.release();
          for (int i=1; i< 4; ++i){
            demo_ast(&scene, &scene_arena, &music_arena, 2,depth,false);
            scene.time+=GetFrameTime();
            demo_march(&scene, &scene_arena, &music_arena, 2,1.0,1,1,1);
            scene_arena.release();
          }
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
  if (fft_memory) {
    free(fft_memory);
  }
  if (fft_data) {
    free(fft_data);
  }
  return 0;
}
}
#ifdef WITH_MAIN
int main() { return jump_start(); }
#endif
