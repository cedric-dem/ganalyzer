#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define INPUT_SIZE 49
#define OUTPUT_IMAGE_SIZE 100
#define MAX_PATH_LEN 1024

#define DIE(...) do { fprintf(stderr, "Error: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

typedef enum { false = 0, true = 1 } bool;

static size_t c_strlen(const char *s) { size_t n = 0; while (s[n]) n++; return n; }
static int c_isspace(char ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v'; }
static int c_isdigit(char ch) { return ch >= '0' && ch <= '9'; }
static void *c_memcpy(void *dst, const void *src, size_t n) { unsigned char *d = dst; const unsigned char *s = src; for (size_t i = 0; i < n; ++i) d[i] = s[i]; return dst; }
static void *c_memset(void *dst, int value, size_t n) { unsigned char *d = dst; for (size_t i = 0; i < n; ++i) d[i] = (unsigned char)value; return dst; }
static int c_strcmp(const char *a, const char *b) { while (*a && *a == *b) { a++; b++; } return (unsigned char)*a - (unsigned char)*b; }
static int c_strncmp(const char *a, const char *b, size_t n) { for (size_t i = 0; i < n; ++i) { if (a[i] != b[i] || !a[i] || !b[i]) return (unsigned char)a[i] - (unsigned char)b[i]; } return 0; }
static char *c_strchr(char *s, int ch) { while (*s) { if (*s == (char)ch) return s; s++; } return ch == '\0' ? s : NULL; }
static char *c_strrchr(char *s, int ch) { char *last = NULL; do { if (*s == (char)ch) last = s; } while (*s++); return last; }
static char *c_strcpy(char *dst, const char *src) { char *start = dst; while ((*dst++ = *src++)); return start; }
static float c_floorf(float x) { int i = (int)x; if ((float)i > x) i--; return (float)i; }
static float c_ceilf(float x) { int i = (int)x; if ((float)i < x) i++; return (float)i; }
static long c_lroundf(float x) { return (long)(x >= 0.0f ? x + 0.5f : x - 0.5f); }
static float c_sqrtf(float x) { if (x <= 0.0f) return 0.0f; float guess = x > 1.0f ? x : 1.0f; for (int i = 0; i < 20; ++i) guess = 0.5f * (guess + x / guess); return guess; }
static float c_expf(float x) {
    if (x < 0.0f) return 1.0f / c_expf(-x);
    int halvings = 0;
    while (x > 1.0f) { x *= 0.5f; halvings++; }
    float term = 1.0f, sum = 1.0f;
    for (int n = 1; n <= 18; ++n) { term *= x / (float)n; sum += term; }
    while (halvings-- > 0) sum *= sum;
    return sum;
}
static float c_tanhf(float x) { if (x > 10.0f) return 1.0f; if (x < -10.0f) return -1.0f; float e = c_expf(2.0f * x); return (e - 1.0f) / (e + 1.0f); }
static double c_pow10_int(int exp) { double value = 1.0; while (exp > 0) { value *= 10.0; exp--; } while (exp < 0) { value /= 10.0; exp++; } return value; }
static double c_parse_json_number(const char *s, size_t *used) {
    size_t i = 0; int sign = 1; if (s[i] == '-') { sign = -1; i++; }
    if (!c_isdigit(s[i])) DIE("Invalid JSON number");
    double value = 0.0;
    if (s[i] == '0') { i++; }
    else { while (c_isdigit(s[i])) { value = value * 10.0 + (double)(s[i++] - '0'); } }
    if (s[i] == '.') {
        i++; if (!c_isdigit(s[i])) DIE("Invalid JSON number fraction");
        double place = 0.1; while (c_isdigit(s[i])) { value += (double)(s[i++] - '0') * place; place *= 0.1; }
    }
    if (s[i] == 'e' || s[i] == 'E') {
        i++; int exp_sign = 1; if (s[i] == '+' || s[i] == '-') { if (s[i] == '-') exp_sign = -1; i++; }
        if (!c_isdigit(s[i])) DIE("Invalid JSON number exponent");
        int exp = 0; while (c_isdigit(s[i])) { exp = exp * 10 + (s[i++] - '0'); }
        value *= c_pow10_int(exp_sign * exp);
    }
    *used = i; return sign * value;
}

typedef enum { JV_NULL, JV_BOOL, JV_NUMBER, JV_STRING, JV_ARRAY, JV_OBJECT } JsonType;

typedef struct JsonValue JsonValue;
typedef struct { char *key; JsonValue *value; } JsonMember;
struct JsonValue {
    JsonType type;
    double number;
    bool boolean;
    char *string;
    JsonValue **items;
    size_t count;
    JsonMember *members;
};

typedef struct { const char *text; size_t pos; } Parser;

static void *xcalloc(size_t n, size_t s) { void *p = calloc(n, s); if (!p) DIE("Out of memory"); return p; }
static void *xrealloc(void *p, size_t s) { void *q = realloc(p, s); if (!q) DIE("Out of memory"); return q; }
static char *xstrdup(const char *s) { size_t n = c_strlen(s) + 1; char *p = xcalloc(n, 1); c_memcpy(p, s, n); return p; }

static JsonValue *json_new(JsonType type) { JsonValue *v = xcalloc(1, sizeof(*v)); v->type = type; return v; }
static void skip_ws(Parser *p) { while (c_isspace(p->text[p->pos])) p->pos++; }
static char peek(Parser *p) { skip_ws(p); return p->text[p->pos]; }
static bool consume(Parser *p, char ch) { skip_ws(p); if (p->text[p->pos] == ch) { p->pos++; return true; } return false; }

static char *parse_string_raw(Parser *p) {
    skip_ws(p);
    if (p->text[p->pos++] != '"') DIE("Expected JSON string at byte %zu", p->pos - 1);
    size_t cap = 32, len = 0;
    char *out = xcalloc(cap, 1);
    for (;;) {
        char ch = p->text[p->pos++];
        if (ch == '\0') DIE("Unterminated JSON string");
        if (ch == '"') break;
        if (ch == '\\') {
            ch = p->text[p->pos++];
            switch (ch) {
                case '"': case '\\': case '/': break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                case 'u': {
                    /* Artifacts use ASCII identifiers. Keep non-ASCII escapes as '?'. */
                    p->pos += 4;
                    ch = '?';
                    break;
                }
                default: DIE("Unsupported JSON escape at byte %zu", p->pos - 1);
            }
        }
        if (len + 2 > cap) { cap *= 2; out = xrealloc(out, cap); }
        out[len++] = ch;
    }
    out[len] = '\0';
    return out;
}

static JsonValue *parse_value(Parser *p);

static JsonValue *parse_array(Parser *p) {
    consume(p, '[');
    JsonValue *v = json_new(JV_ARRAY);
    if (consume(p, ']')) return v;
    for (;;) {
        v->items = xrealloc(v->items, sizeof(JsonValue*) * (v->count + 1));
        v->items[v->count++] = parse_value(p);
        if (consume(p, ']')) return v;
        if (!consume(p, ',')) DIE("Expected ',' or ']' in array at byte %zu", p->pos);
    }
}

static JsonValue *parse_object(Parser *p) {
    consume(p, '{');
    JsonValue *v = json_new(JV_OBJECT);
    if (consume(p, '}')) return v;
    for (;;) {
        char *key = parse_string_raw(p);
        if (!consume(p, ':')) DIE("Expected ':' after object key at byte %zu", p->pos);
        v->members = xrealloc(v->members, sizeof(JsonMember) * (v->count + 1));
        v->members[v->count].key = key;
        v->members[v->count].value = parse_value(p);
        v->count++;
        if (consume(p, '}')) return v;
        if (!consume(p, ',')) DIE("Expected ',' or '}' in object at byte %zu", p->pos);
    }
}

static JsonValue *parse_number(Parser *p) {
    skip_ws(p);
    size_t used = 0;
    double n = c_parse_json_number(p->text + p->pos, &used);
    if (used == 0) DIE("Invalid JSON number at byte %zu", p->pos);
    p->pos += used;
    JsonValue *v = json_new(JV_NUMBER);
    v->number = n;
    return v;
}

static JsonValue *parse_value(Parser *p) {
    char ch = peek(p);
    if (ch == '{') return parse_object(p);
    if (ch == '[') return parse_array(p);
    if (ch == '"') { JsonValue *v = json_new(JV_STRING); v->string = parse_string_raw(p); return v; }
    if (ch == '-' || c_isdigit(ch)) return parse_number(p);
    if (c_strncmp(p->text + p->pos, "true", 4) == 0) { p->pos += 4; JsonValue *v = json_new(JV_BOOL); v->boolean = true; return v; }
    if (c_strncmp(p->text + p->pos, "false", 5) == 0) { p->pos += 5; JsonValue *v = json_new(JV_BOOL); return v; }
    if (c_strncmp(p->text + p->pos, "null", 4) == 0) { p->pos += 4; return json_new(JV_NULL); }
    DIE("Unexpected JSON token at byte %zu", p->pos);
}

static JsonValue *json_parse_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) DIE("Could not open %s", path);
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    char *buf = xcalloc((size_t)len + 1, 1);
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) DIE("Could not read %s", path);
    fclose(f);
    Parser p = {buf, 0};
    JsonValue *root = parse_value(&p);
    free(buf);
    return root;
}

static JsonValue *jget(JsonValue *obj, const char *key) {
    if (!obj || obj->type != JV_OBJECT) DIE("Expected object while reading key %s", key);
    for (size_t i = 0; i < obj->count; ++i) if (c_strcmp(obj->members[i].key, key) == 0) return obj->members[i].value;
    return NULL;
}
static JsonValue *jreq(JsonValue *obj, const char *key) { JsonValue *v = jget(obj, key); if (!v) DIE("Missing JSON key: %s", key); return v; }
static const char *jstr(JsonValue *obj, const char *key, const char *def) { JsonValue *v = jget(obj, key); if (!v) return def; if (v->type != JV_STRING) DIE("Expected string key: %s", key); return v->string; }
static int jint(JsonValue *obj, const char *key, int def) { JsonValue *v = jget(obj, key); if (!v) return def; if (v->type != JV_NUMBER) DIE("Expected numeric key: %s", key); return (int)v->number; }
static float jfloat(JsonValue *obj, const char *key, float def) { JsonValue *v = jget(obj, key); if (!v) return def; if (v->type != JV_NUMBER) DIE("Expected numeric key: %s", key); return (float)v->number; }
static bool jbool(JsonValue *obj, const char *key, bool def) { JsonValue *v = jget(obj, key); if (!v) return def; if (v->type == JV_BOOL) return v->boolean; DIE("Expected boolean key: %s", key); }
static int jint_at(JsonValue *arr, size_t idx) { if (!arr || arr->type != JV_ARRAY || idx >= arr->count || arr->items[idx]->type != JV_NUMBER) DIE("Expected integer array item"); return (int)arr->items[idx]->number; }

typedef struct { int *shape; int rank; float *data; size_t size; } Tensor;

typedef struct { char *name; Tensor *items; int count; } LayerWeights;
typedef struct { JsonValue *layers; LayerWeights *weights; int weight_count; } ModelArtifacts;

typedef struct { char artifacts_dir[MAX_PATH_LEN]; char recreated_dir[MAX_PATH_LEN]; } AppConfig;

static const float CUSTOM_INPUT_VECTOR[INPUT_SIZE] = {
    -0.16f, -0.15f, +0.19f, -0.14f, +0.05f, +0.66f, -0.07f,
    -0.06f, -0.04f, -0.0f, +0.51f, -0.05f, +0.23f, -0.17f,
    -0.06f, +0.07f, +0.39f, +0.23f, -0.08f, +0.02f, -0.14f,
    -0.16f, +0.0f, +0.05f, +0.09f, +0.15f, +0.09f, -0.12f,
    -0.54f, +0.18f, -0.44f, -0.13f, -0.2f, +0.03f, -0.23f,
    -0.14f, -0.46f, +0.2f, +0.03f, -0.36f, +0.42f, -0.12f,
    -0.19f, -0.25f, -0.23f, -0.04f, -0.18f, +0.18f, -0.19f,
};

static Tensor tensor_create(int rank, const int *shape) {
    Tensor t = {0}; t.rank = rank; t.shape = xcalloc((size_t)rank, sizeof(int));
    t.size = 1; for (int i = 0; i < rank; ++i) { t.shape[i] = shape[i]; t.size *= (size_t)shape[i]; }
    t.data = xcalloc(t.size, sizeof(float)); return t;
}
static Tensor tensor_clone(const Tensor *src) { Tensor t = tensor_create(src->rank, src->shape); c_memcpy(t.data, src->data, src->size * sizeof(float)); return t; }
static void tensor_free(Tensor *t) { free(t->shape); free(t->data); c_memset(t, 0, sizeof(*t)); }
static inline float get1(const Tensor *t, int i) { return t->data[i]; }
static inline float get2(const Tensor *t, int i, int j) { return t->data[i * t->shape[1] + j]; }
static inline void set2(Tensor *t, int i, int j, float v) { t->data[i * t->shape[1] + j] = v; }
static inline float get4(const Tensor *t, int a, int b, int c, int d) { return t->data[((a * t->shape[1] + b) * t->shape[2] + c) * t->shape[3] + d]; }
static inline void set4(Tensor *t, int a, int b, int c, int d, float v) { t->data[((a * t->shape[1] + b) * t->shape[2] + c) * t->shape[3] + d] = v; }

static void infer_shape(JsonValue *j, int **shape, int *rank) {
    int cap = 4; *shape = xcalloc((size_t)cap, sizeof(int)); *rank = 0;
    while (j && j->type == JV_ARRAY) {
        if (*rank == cap) { cap *= 2; *shape = xrealloc(*shape, sizeof(int) * (size_t)cap); }
        (*shape)[(*rank)++] = (int)j->count;
        if (j->count == 0) break;
        j = j->items[0];
    }
}
static void flatten_json(JsonValue *j, float *out, size_t *cursor) {
    if (j->type == JV_ARRAY) for (size_t i = 0; i < j->count; ++i) flatten_json(j->items[i], out, cursor);
    else if (j->type == JV_NUMBER) out[(*cursor)++] = (float)j->number;
    else DIE("Expected numeric tensor leaf");
}
static Tensor tensor_from_json(JsonValue *j) {
    int *shape = NULL, rank = 0; infer_shape(j, &shape, &rank);
    Tensor t = tensor_create(rank, shape); free(shape);
    size_t cursor = 0; flatten_json(j, t.data, &cursor);
    return t;
}

static char *trim(char *s) {
    while (c_isspace(*s)) s++;
    char *end = s + c_strlen(s);
    while (end > s && c_isspace(end[-1])) *--end = '\0';
    return s;
}
static bool file_exists(const char *path) { FILE *f = fopen(path, "rb"); if (f) { fclose(f); return true; } return false; }
static void dirname_of(const char *path, char *out) { c_strcpy(out, path); char *slash = c_strrchr(out, '/'); if (slash) *slash = '\0'; else c_strcpy(out, "."); }
static void join_path(char *out, const char *a, const char *b) { if (b[0] == '/') snprintf(out, MAX_PATH_LEN, "%s", b); else snprintf(out, MAX_PATH_LEN, "%s/%s", a, b); }
static void parse_toml_string(const char *value, char *out) {
    const char *p = value; while (c_isspace(*p)) p++;
    if (*p++ != '"') DIE("Expected quoted TOML string value: %s", value);
    size_t n = 0;
    while (*p && *p != '"') {
        char ch = *p++;
        if (ch == '\\') { ch = *p++; if (ch == 'n') ch = '\n'; else if (ch == 'r') ch = '\r'; else if (ch == 't') ch = '\t'; }
        out[n++] = ch;
    }
    out[n] = '\0';
}
static AppConfig load_app_config(void) {
    const char *candidates[] = {"config.toml", "../config.toml", "create_from_scratch/config.toml"};
    char config_path[MAX_PATH_LEN] = {0};
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); ++i) if (file_exists(candidates[i])) { c_strcpy(config_path, candidates[i]); break; }
    if (!config_path[0]) DIE("Could not find config.toml. Run from create_from_scratch/, 5_c_implementation/, or the repository root.");
    FILE *f = fopen(config_path, "r"); if (!f) DIE("Could not open %s", config_path);
    char artifacts[MAX_PATH_LEN] = {0}, recreated[MAX_PATH_LEN] = {0};
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        char *hash = c_strchr(line, '#'); if (hash) *hash = '\0';
        char *eq = c_strchr(line, '='); if (!eq) continue; *eq = '\0';
        char *key = trim(line); char *value = trim(eq + 1); char parsed[MAX_PATH_LEN];
        if (!*key) continue;
        parse_toml_string(value, parsed);
        if (c_strcmp(key, "model_split_into_files") == 0) c_strcpy(artifacts, parsed);
        if (c_strcmp(key, "path_to_c_implementation_intermediary") == 0 ||
            (!recreated[0] && c_strcmp(key, "path_to_cpp_implementation_intermediary") == 0)) {
            c_strcpy(recreated, parsed);
        }
    }
    fclose(f);
    if (!artifacts[0]) DIE("Missing model_split_into_files in config.toml");
    if (!recreated[0]) DIE("Missing path_to_c_implementation_intermediary in config.toml");
    char base[MAX_PATH_LEN]; dirname_of(config_path, base);
    AppConfig cfg; join_path(cfg.artifacts_dir, base, artifacts); join_path(cfg.recreated_dir, base, recreated); return cfg;
}

static int expected_weight_count(const char *class_name, JsonValue *cfg) {
    if (!c_strcmp(class_name, "Dense") || !c_strcmp(class_name, "Conv2D") || !c_strcmp(class_name, "Conv2DTranspose")) return jbool(cfg, "use_bias", true) ? 2 : 1;
    if (!c_strcmp(class_name, "BatchNormalization")) return (jbool(cfg, "scale", true) ? 1 : 0) + (jbool(cfg, "center", true) ? 1 : 0) + 2;
    return 0;
}
static LayerWeights *find_weights(ModelArtifacts *a, const char *name) { for (int i = 0; i < a->weight_count; ++i) if (!c_strcmp(a->weights[i].name, name)) return &a->weights[i]; DIE("Missing weights for layer %s", name); }
static ModelArtifacts load_model(const char *dir) {
    char p1[MAX_PATH_LEN * 2], p2[MAX_PATH_LEN * 2]; snprintf(p1, sizeof(p1), "%s/config.json", dir); snprintf(p2, sizeof(p2), "%s/model.weights.json", dir);
    JsonValue *config = json_parse_file(p1); JsonValue *weights_payload = json_parse_file(p2);
    JsonValue *layers = jreq(jreq(config, "config"), "layers"); JsonValue *flat = jreq(weights_payload, "weights");
    if (layers->type != JV_ARRAY || flat->type != JV_ARRAY) DIE("Invalid model artifacts");
    ModelArtifacts a = {0}; a.layers = layers; a.weight_count = (int)layers->count; a.weights = xcalloc(layers->count, sizeof(LayerWeights));
    size_t cursor = 0;
    for (size_t i = 0; i < layers->count; ++i) {
        JsonValue *layer = layers->items[i]; const char *name = jstr(layer, "name", NULL); const char *class_name = jstr(layer, "class_name", NULL);
        int expected = expected_weight_count(class_name, jreq(layer, "config"));
        a.weights[i].name = xstrdup(name); a.weights[i].count = expected; a.weights[i].items = xcalloc((size_t)expected, sizeof(Tensor));
        for (int w = 0; w < expected; ++w) { if (cursor >= flat->count) DIE("Not enough weights"); a.weights[i].items[w] = tensor_from_json(flat->items[cursor++]); }
    }
    return a;
}

static float apply_activation(float v, const char *activation) {
    if (!c_strcmp(activation, "linear")) return v;
    if (!c_strcmp(activation, "relu")) return v > 0.0f ? v : 0.0f;
    if (!c_strcmp(activation, "tanh")) return c_tanhf(v);
    if (!c_strcmp(activation, "sigmoid")) return 1.0f / (1.0f + c_expf(-v));
    DIE("Unsupported activation: %s", activation);
}
static void apply_activation_in_place(Tensor *t, const char *activation) { for (size_t i = 0; i < t->size; ++i) t->data[i] = apply_activation(t->data[i], activation); }
static int starts_with(const char *s, const char *prefix) { return c_strncmp(s, prefix, c_strlen(prefix)) == 0; }
static int ceil_div_float(int a, int b) { return (int)c_ceilf((float)a / (float)b); }

static Tensor handle_dense(const char *name, const Tensor *v, JsonValue *cfg, ModelArtifacts *a) {
    LayerWeights *lw = find_weights(a, jstr(cfg, "name", name)); Tensor *kernel = &lw->items[0]; Tensor *bias = jbool(cfg, "use_bias", true) ? &lw->items[1] : NULL;
    int shape[] = {v->shape[0], jint(cfg, "units", 0)}; Tensor out = tensor_create(2, shape);
    for (int b = 0; b < shape[0]; ++b) for (int u = 0; u < shape[1]; ++u) {
        float acc = 0.0f; for (int i = 0; i < v->shape[1]; ++i) acc += get2(v, b, i) * get2(kernel, i, u);
        if (bias) acc += get1(bias, u);
        set2(&out, b, u, acc);
    }
    return out;
}
static Tensor handle_batch_norm(const char *name, const Tensor *v, JsonValue *cfg, ModelArtifacts *a) {
    LayerWeights *lw = find_weights(a, name); Tensor *gamma = &lw->items[0], *beta = &lw->items[1], *mean = &lw->items[2], *var = &lw->items[3];
    Tensor out = tensor_clone(v); int channels = v->shape[v->rank - 1]; int spatial = 1; for (int i = 0; i + 1 < v->rank; ++i) spatial *= v->shape[i]; float eps = jfloat(cfg, "epsilon", 1e-3f);
    for (int s = 0; s < spatial; ++s) for (int c = 0; c < channels; ++c) { int idx = s * channels + c; float norm = (v->data[idx] - get1(mean, c)) / c_sqrtf(get1(var, c) + eps); out.data[idx] = get1(gamma, c) * norm + get1(beta, c); }
    return out;
}
static Tensor handle_leaky_relu(const Tensor *v, JsonValue *cfg) { Tensor out = tensor_clone(v); float slope = jfloat(cfg, "negative_slope", 0.3f); for (size_t i = 0; i < out.size; ++i) if (out.data[i] < 0.0f) out.data[i] *= slope; return out; }
static Tensor handle_reshape(const Tensor *v, JsonValue *cfg) { JsonValue *target = jreq(cfg, "target_shape"); int shape[8]; shape[0] = v->shape[0]; for (size_t i = 0; i < target->count; ++i) shape[i + 1] = jint_at(target, i); Tensor out = tensor_create((int)target->count + 1, shape); c_memcpy(out.data, v->data, v->size * sizeof(float)); return out; }
static void add_channel_bias_4d(Tensor *t, const Tensor *bias) { for (int b=0;b<t->shape[0];++b) for (int y=0;y<t->shape[1];++y) for (int x=0;x<t->shape[2];++x) for (int c=0;c<t->shape[3];++c) set4(t,b,y,x,c,get4(t,b,y,x,c)+get1(bias,c)); }

static Tensor handle_conv2d(const char *name, const Tensor *v, JsonValue *cfg, ModelArtifacts *a) {
    LayerWeights *lw = find_weights(a, jstr(cfg, "name", name)); Tensor *kernel = &lw->items[0]; JsonValue *ksize = jreq(cfg,"kernel_size"), *strides = jget(cfg,"strides"), *dilation = jget(cfg,"dilation_rate");
    int kh=jint_at(ksize,0), kw=jint_at(ksize,1), sh=strides?jint_at(strides,0):1, sw=strides?jint_at(strides,1):1, dh=dilation?jint_at(dilation,0):1, dw=dilation?jint_at(dilation,1):1;
    int bsz=v->shape[0], in_h=v->shape[1], in_w=v->shape[2], in_c=v->shape[3], filters=jint(cfg,"filters",0), groups=jint(cfg,"groups",1);
    int channels_per_group=in_c/groups, filters_per_group=filters/groups, eff_h=(kh-1)*dh+1, eff_w=(kw-1)*dw+1, out_h=0,out_w=0,pad_top=0,pad_left=0; const char *padding=jstr(cfg,"padding","valid");
    if (!c_strcmp(padding,"valid")) { out_h=(in_h-eff_h)/sh+1; out_w=(in_w-eff_w)/sw+1; }
    else if (!c_strcmp(padding,"same")) { out_h=ceil_div_float(in_h,sh); out_w=ceil_div_float(in_w,sw); int th=(out_h-1)*sh+eff_h-in_h; int tw=(out_w-1)*sw+eff_w-in_w; pad_top=(th>0?th:0)/2; pad_left=(tw>0?tw:0)/2; }
    else DIE("Unsupported Conv2D padding: %s", padding);
    int shape[]={bsz,out_h,out_w,filters}; Tensor out=tensor_create(4,shape);
    for(int b=0;b<bsz;++b) for(int oy=0;oy<out_h;++oy){ int iyb=oy*sh-pad_top; for(int ox=0;ox<out_w;++ox){ int ixb=ox*sw-pad_left; for(int ky=0;ky<kh;++ky){ int iy=iyb+ky*dh; if(iy<0||iy>=in_h) continue; for(int kx=0;kx<kw;++kx){ int ix=ixb+kx*dw; if(ix<0||ix>=in_w) continue; for(int g=0;g<groups;++g){ int is=g*channels_per_group, ie=is+channels_per_group, fs=g*filters_per_group, fe=fs+filters_per_group; for(int f=fs;f<fe;++f){ float acc=get4(&out,b,oy,ox,f); for(int c=is;c<ie;++c) acc += get4(v,b,iy,ix,c)*get4(kernel,ky,kx,c-is,f); set4(&out,b,oy,ox,f,acc); }}}}}}
    if (jbool(cfg,"use_bias",true)) add_channel_bias_4d(&out, &lw->items[1]);
    apply_activation_in_place(&out, jstr(cfg,"activation","linear"));
    return out;
}
static Tensor handle_conv2d_transpose(const char *name, const Tensor *v, JsonValue *cfg, ModelArtifacts *a) {
    LayerWeights *lw=find_weights(a,jstr(cfg,"name",name)); Tensor *kernel=&lw->items[0]; JsonValue *ksize=jreq(cfg,"kernel_size"), *strides=jreq(cfg,"strides");
    int kh=jint_at(ksize,0), kw=jint_at(ksize,1), sh=jint_at(strides,0), sw=jint_at(strides,1), filters=jint(cfg,"filters",0); int bsz=v->shape[0], in_h=v->shape[1], in_w=v->shape[2], in_c=v->shape[3];
    int out_h,out_w,pad_top=0,pad_left=0; const char *padding=jstr(cfg,"padding","valid"); if(!c_strcmp(padding,"same")){out_h=in_h*sh;out_w=in_w*sw;pad_top=((kh-sh)>0?(kh-sh):0)/2;pad_left=((kw-sw)>0?(kw-sw):0)/2;} else if(!c_strcmp(padding,"valid")){out_h=(in_h-1)*sh+kh;out_w=(in_w-1)*sw+kw;} else DIE("Unsupported Conv2DTranspose padding: %s", padding);
    int shape[]={bsz,out_h,out_w,filters}; Tensor out=tensor_create(4,shape);
    for(int b=0;b<bsz;++b) for(int iy=0;iy<in_h;++iy){ int by=iy*sh; for(int ix=0;ix<in_w;++ix){ int bx=ix*sw; for(int ky=0;ky<kh;++ky){ int oy=by+ky-pad_top; if(oy<0||oy>=out_h) continue; for(int kx=0;kx<kw;++kx){ int ox=bx+kx-pad_left; if(ox<0||ox>=out_w) continue; for(int f=0;f<filters;++f){ float acc=get4(&out,b,oy,ox,f); for(int c=0;c<in_c;++c) acc += get4(v,b,iy,ix,c)*get4(kernel,ky,kx,f,c); set4(&out,b,oy,ox,f,acc); }}}}}
    if (jbool(cfg,"use_bias",true)) add_channel_bias_4d(&out, &lw->items[1]);
    apply_activation_in_place(&out, jstr(cfg,"activation","linear"));
    return out;
}
static Tensor handle_resize(const Tensor *v, JsonValue *cfg) {
    int th=jint(cfg,"height",0), tw=jint(cfg,"width",0); int bsz=v->shape[0], ih=v->shape[1], iw=v->shape[2], ch=v->shape[3]; int shape[]={bsz,th,tw,ch}; Tensor out=tensor_create(4,shape);
    for(int b=0;b<bsz;++b) {
        for(int oy=0;oy<th;++oy){ float y=oy*((float)ih/(float)th); int y0=(int)c_floorf(y); if(y0>ih-1)y0=ih-1; int y1=y0+1; if(y1>ih-1)y1=ih-1; float wy=y-y0; for(int ox=0;ox<tw;++ox){ float x=ox*((float)iw/(float)tw); int x0=(int)c_floorf(x); if(x0>iw-1)x0=iw-1; int x1=x0+1; if(x1>iw-1)x1=iw-1; float wx=x-x0; for(int c=0;c<ch;++c){ float top=(1-wx)*get4(v,b,y0,x0,c)+wx*get4(v,b,y0,x1,c); float bot=(1-wx)*get4(v,b,y1,x0,c)+wx*get4(v,b,y1,x1,c); set4(&out,b,oy,ox,c,(1-wy)*top+wy*bot); }}}
    }
    return out;
}
static Tensor recreate_layer(const char *name, JsonValue *cfg, const Tensor *in, ModelArtifacts *a) {
    if (starts_with(name,"input_layer")) return tensor_clone(in);
    if (starts_with(name,"dense")) return handle_dense(name,in,cfg,a);
    if (starts_with(name,"batch_normalization")) return handle_batch_norm(name,in,cfg,a);
    if (starts_with(name,"leaky_re_lu")) return handle_leaky_relu(in,cfg);
    if (starts_with(name,"reshape")) return handle_reshape(in,cfg);
    if (starts_with(name,"conv2d_transpose")) return handle_conv2d_transpose(name,in,cfg,a);
    if (starts_with(name,"conv2d")) return handle_conv2d(name,in,cfg,a);
    if (starts_with(name,"resizing")) return handle_resize(in,cfg);
    DIE("Unsupported layer: %s", name);
}

static void mkdir_p(const char *path) {
    char tmp[MAX_PATH_LEN]; snprintf(tmp, sizeof(tmp), "%s", path); for (char *p = tmp + 1; *p; ++p) if (*p == '/') { *p = '\0'; mkdir(tmp, 0777); *p = '/'; } mkdir(tmp, 0777);
}
static void write_tensor_json_rec(FILE *f, const Tensor *t, int dim, size_t *cursor) {
    if (dim == t->rank) { fprintf(f, "%.9g", t->data[(*cursor)++]); return; }
    fputc('[', f); for (int i=0;i<t->shape[dim];++i) { if (i) fputc(',', f); write_tensor_json_rec(f,t,dim+1,cursor); } fputc(']', f);
}
static void save_tensor(const char *dir, int index, const char *label, const Tensor *t) {
    mkdir_p(dir); char safe[256]; snprintf(safe, sizeof(safe), "%s", label); for(char *p=safe;*p;++p) if(*p=='/'||*p==' ') *p='_'; char path[MAX_PATH_LEN]; snprintf(path,sizeof(path),"%s/values_%03d_%s.txt",dir,index,safe); FILE *f=fopen(path,"w"); if(!f) DIE("Could not write %s",path); size_t c=0; write_tensor_json_rec(f,t,0,&c); fclose(f);
}
static int to_u8(float v) { int scaled=(int)c_lroundf((v+1.0f)*127.5f); if(scaled<0) return 0; if(scaled>255) return 255; return scaled; }
static void save_ppm(const Tensor *img, const char *path) { if(img->rank!=4 || img->shape[1]!=OUTPUT_IMAGE_SIZE || img->shape[2]!=OUTPUT_IMAGE_SIZE || img->shape[3]!=3) DIE("Expected output tensor shape [1,100,100,3]"); FILE *f=fopen(path,"wb"); if(!f) DIE("Could not write %s",path); fprintf(f,"P6\n%d %d\n255\n",img->shape[2],img->shape[1]); for(int y=0;y<img->shape[1];++y) for(int x=0;x<img->shape[2];++x){ unsigned char rgb[3]={(unsigned char)to_u8(get4(img,0,y,x,0)),(unsigned char)to_u8(get4(img,0,y,x,1)),(unsigned char)to_u8(get4(img,0,y,x,2))}; fwrite(rgb,1,3,f);} fclose(f); }

int main(void) {
    AppConfig cfg = load_app_config();
    ModelArtifacts artifacts = load_model(cfg.artifacts_dir);
    int input_shape[] = {1, INPUT_SIZE}; Tensor current = tensor_create(2, input_shape); c_memcpy(current.data, CUSTOM_INPUT_VECTOR, sizeof(CUSTOM_INPUT_VECTOR));
    save_tensor(cfg.recreated_dir, 0, "original", &current);
    for (size_t i = 0; i < artifacts.layers->count; ++i) {
        JsonValue *layer = artifacts.layers->items[i]; const char *name = jstr(layer,"name",NULL); Tensor next = recreate_layer(name, jreq(layer,"config"), &current, &artifacts); tensor_free(&current); current = next; save_tensor(cfg.recreated_dir, (int)i + 1, name, &current);
    }
    if (current.rank == 4 && current.shape[0] > 0 && current.shape[3] == 3) { char out[MAX_PATH_LEN * 2]; snprintf(out,sizeof(out),"%s/out.ppm",cfg.recreated_dir); save_ppm(&current,out); printf("Saved recreated image to: %s\n", out); }
    printf("Saved recreated values to: %s\n", cfg.recreated_dir);
    tensor_free(&current);
    return 0;
}