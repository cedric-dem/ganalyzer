#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace {

constexpr int kInputSize = 49;
constexpr int kOutputImageSize = 100;

struct AppConfig {
  std::filesystem::path artifacts_dir;
  std::filesystem::path recreated_dir;
};

std::string trim(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::size_t start = value.find_first_not_of(whitespace);
  if (start == std::string::npos) return "";
  const std::size_t end = value.find_last_not_of(whitespace);
  return value.substr(start, end - start + 1);
}

std::string parse_toml_string_value(const std::string& value) {
  std::string trimmed = trim(value);
  if (trimmed.size() < 2 || trimmed.front() != '"') {
    throw std::runtime_error("Expected quoted TOML string value: " + value);
  }

  std::string parsed;
  bool escaping = false;
  for (std::size_t i = 1; i < trimmed.size(); ++i) {
    char ch = trimmed[i];
    if (escaping) {
      switch (ch) {
        case '"':
        case '\\':
          parsed.push_back(ch);
          break;
        case 'n':
          parsed.push_back('\n');
          break;
        case 'r':
          parsed.push_back('\r');
          break;
        case 't':
          parsed.push_back('\t');
          break;
        default:
          parsed.push_back(ch);
          break;
      }
      escaping = false;
      continue;
    }

    if (ch == '\\') {
      escaping = true;
      continue;
    }
    if (ch == '"') return parsed;
    parsed.push_back(ch);
  }

  throw std::runtime_error("Unterminated TOML string value: " + value);
}

std::filesystem::path resolve_config_path() {
  const std::array<std::filesystem::path, 2> candidates = {
      std::filesystem::path("config.toml"),
      std::filesystem::path("../config.toml"),
  };

  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) return candidate;
  }

  throw std::runtime_error("Could not find config.toml. Run from create_from_scratch/ or 3_implementation_cpp/.");
}

std::filesystem::path resolve_config_relative_path(const std::filesystem::path& config_path,
                                                   const std::string& configured_path) {
  std::filesystem::path path(configured_path);
  if (path.is_absolute()) return path;
  return config_path.parent_path() / path;
}

AppConfig load_app_config() {
  const std::filesystem::path config_path = resolve_config_path();
  std::ifstream config_file(config_path);
  if (!config_file) throw std::runtime_error("Could not open config file: " + config_path.string());

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(config_file, line)) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) line = line.substr(0, comment);
    line = trim(line);
    if (line.empty()) continue;

    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) continue;

    const std::string key = trim(line.substr(0, equals));
    const std::string value = parse_toml_string_value(line.substr(equals + 1));
    values[key] = value;
  }

  const auto artifacts_it = values.find("model_split_into_files");
  if (artifacts_it == values.end()) throw std::runtime_error("Missing model_split_into_files in config.toml");

  const auto recreated_it = values.find("path_to_cpp_implementation_intermediary");
  if (recreated_it == values.end()) {
    throw std::runtime_error("Missing path_to_cpp_implementation_intermediary in config.toml");
  }

  return {
      resolve_config_relative_path(config_path, artifacts_it->second),
      resolve_config_relative_path(config_path, recreated_it->second),
  };
}

const std::array<float, kInputSize> kCustomInputVector = {
    -0.16f, -0.15f, +0.19f, -0.14f, +0.05f, +0.66f, -0.07f,
    -0.06f, -0.04f, -0.0f, +0.51f, -0.05f, +0.23f, -0.17f,
    -0.06f, +0.07f, +0.39f, +0.23f, -0.08f, +0.02f, -0.14f,
    -0.16f, +0.0f, +0.05f, +0.09f, +0.15f, +0.09f, -0.12f,
    -0.54f, +0.18f, -0.44f, -0.13f, -0.2f, +0.03f, -0.23f,
    -0.14f, -0.46f, +0.2f, +0.03f, -0.36f, +0.42f, -0.12f,
    -0.19f, -0.25f, -0.23f, -0.04f, -0.18f, +0.18f, -0.19f,
};

struct Tensor {
  std::vector<int> shape;
  std::vector<float> data;

  Tensor() = default;
  explicit Tensor(std::vector<int> shape_) : shape(std::move(shape_)) {
    int total = 1;
    for (int d : shape) total *= d;
    data.assign(total, 0.0f);
  }

  int offset(const std::vector<int>& idx) const {
    if (idx.size() != shape.size()) throw std::runtime_error("Index rank mismatch.");
    int off = 0;
    int stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      off += idx[i] * stride;
      stride *= shape[i];
    }
    return off;
  }

  float get(const std::vector<int>& idx) const { return data[offset(idx)]; }
  void set(const std::vector<int>& idx, float v) { data[offset(idx)] = v; }

  float get1(int i0) const { return data[i0]; }
  void set1(int i0, float v) { data[i0] = v; }

  float get2(int i0, int i1) const { return data[i0 * shape[1] + i1]; }
  void set2(int i0, int i1, float v) { data[i0 * shape[1] + i1] = v; }

  float get4(int i0, int i1, int i2, int i3) const {
    return data[((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3];
  }
  void set4(int i0, int i1, int i2, int i3, float v) {
    data[((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3] = v;
  }
};

int ceil_int(float v) { return static_cast<int>(std::ceil(v)); }
int floor_int(float v) { return static_cast<int>(std::floor(v)); }

float fast_exp(float x) { return std::exp(x); }
float fast_sqrt(float x) {
  if (x < 0.0f) throw std::runtime_error("Cannot sqrt negative value");
  return std::sqrt(x);
}
float fast_tanh(float x) { return std::tanh(x); }

float apply_activation(float value, const std::string& activation) {
  if (activation == "linear") return value;
  if (activation == "relu") return value > 0.0f ? value : 0.0f;
  if (activation == "tanh") return fast_tanh(value);
  if (activation == "sigmoid") return 1.0f / (1.0f + fast_exp(-value));
  throw std::runtime_error("Unsupported activation: " + activation);
}

void apply_activation_in_place(Tensor& t, const std::string& activation) {
  for (float& v : t.data) v = apply_activation(v, activation);
}

std::vector<int> infer_shape(const json& j) {
  if (!j.is_array()) return {};
  std::vector<int> shape;
  const json* cur = &j;
  while (cur->is_array()) {
    shape.push_back(static_cast<int>(cur->size()));
    if (cur->empty()) break;
    cur = &(*cur)[0];
  }
  return shape;
}

void flatten_json(const json& j, std::vector<float>& out) {
  if (j.is_array()) {
    for (const auto& item : j) flatten_json(item, out);
  } else {
    out.push_back(j.get<float>());
  }
}

Tensor tensor_from_json(const json& j) {
  Tensor t;
  t.shape = infer_shape(j);
  flatten_json(j, t.data);
  return t;
}

json tensor_to_json_rec(const Tensor& t, int dim, int& cursor) {
  if (dim == static_cast<int>(t.shape.size())) {
    return t.data.at(cursor++);
  }
  json arr = json::array();
  for (int i = 0; i < t.shape[dim]; ++i) {
    arr.push_back(tensor_to_json_rec(t, dim + 1, cursor));
  }
  return arr;
}

json tensor_to_json(const Tensor& t) {
  int cursor = 0;
  return tensor_to_json_rec(t, 0, cursor);
}

void save_json_value(const std::filesystem::path& base_dir, int index, const std::string& label, const json& value) {
  std::filesystem::create_directories(base_dir);
  std::string safe_label = label;
  std::replace(safe_label.begin(), safe_label.end(), '/', '_');
  std::replace(safe_label.begin(), safe_label.end(), ' ', '_');

  std::ostringstream name;
  name << "values_" << std::setw(3) << std::setfill('0') << index << "_" << safe_label << ".txt";
  std::ofstream out(base_dir / name.str());
  out << value.dump();
}

int expected_weight_count_for_layer(const std::string& class_name, const json& cfg) {
  if (class_name == "Dense" || class_name == "Conv2D" || class_name == "Conv2DTranspose") {
    return cfg.value("use_bias", true) ? 2 : 1;
  }
  if (class_name == "BatchNormalization") {
    int count = 0;
    if (cfg.value("scale", true)) ++count;
    if (cfg.value("center", true)) ++count;
    return count + 2;
  }
  return 0;
}

struct ModelArtifacts {
  json layer_defs;
  std::unordered_map<std::string, std::vector<Tensor>> layer_weights;
};

ModelArtifacts load_model_from_artifacts(const std::filesystem::path& artifacts_dir) {
  std::ifstream config_f(artifacts_dir / "config.json");
  std::ifstream weights_f(artifacts_dir / "model.weights.json");
  if (!config_f || !weights_f) throw std::runtime_error("Missing artifact files.");

  json config = json::parse(config_f);
  json weights_payload = json::parse(weights_f);

  const json& layer_defs = config["config"]["layers"];
  const json& flat_weights = weights_payload["weights"];

  std::unordered_map<std::string, std::vector<Tensor>> layer_weights;
  int cursor = 0;

  for (const auto& layer : layer_defs) {
    std::string name = layer["name"].get<std::string>();
    std::string class_name = layer["class_name"].get<std::string>();
    int expected = expected_weight_count_for_layer(class_name, layer["config"]);

    std::vector<Tensor> arr;
    for (int i = 0; i < expected; ++i) arr.push_back(tensor_from_json(flat_weights.at(cursor + i)));
    cursor += expected;
    layer_weights[name] = std::move(arr);
  }

  return {layer_defs, layer_weights};
}

Tensor handle_dense(const std::string& layer_name, const Tensor& values, const json& cfg,
                    const std::unordered_map<std::string, std::vector<Tensor>>& weights) {
  (void)layer_name;
  int batch = values.shape.at(0);
  int input_width = values.shape.at(1);
  int units = cfg["units"].get<int>();
  bool use_bias = cfg.value("use_bias", true);

  Tensor out({batch, units});

  const auto& layer_w = weights.at(cfg.value("name", layer_name));
  const Tensor& kernel = layer_w.at(0);
  const Tensor* bias = use_bias ? &layer_w.at(1) : nullptr;

  for (int b = 0; b < batch; ++b) {
    for (int u = 0; u < units; ++u) {
      float acc = 0.0f;
      for (int i = 0; i < input_width; ++i) {
        acc += values.get2(b, i) * kernel.get2(i, u);
      }
      if (bias) acc += bias->get1(u);
      out.set2(b, u, acc);
    }
  }
  return out;
}

Tensor handle_batch_norm(const std::string& layer_name, const Tensor& values, const json& cfg,
                         const std::unordered_map<std::string, std::vector<Tensor>>& weights) {
  const auto& w = weights.at(layer_name);
  const Tensor& gamma = w.at(0);
  const Tensor& beta = w.at(1);
  const Tensor& moving_mean = w.at(2);
  const Tensor& moving_var = w.at(3);
  float eps = cfg.value("epsilon", 1e-3f);

  Tensor out = values;
  int channels = values.shape.back();
  int spatial = 1;
  for (size_t i = 0; i + 1 < values.shape.size(); ++i) spatial *= values.shape[i];
  for (int s = 0; s < spatial; ++s) {
    for (int c = 0; c < channels; ++c) {
      int idx = s * channels + c;
      float norm = (values.data[idx] - moving_mean.get({c})) / fast_sqrt(moving_var.get({c}) + eps);
      out.data[idx] = gamma.get({c}) * norm + beta.get({c});
    }
  }
  return out;
}

Tensor handle_leaky_relu(const Tensor& values, const json& cfg) {
  float slope = cfg.value("negative_slope", 0.3f);
  Tensor out = values;
  for (float& v : out.data) {
    if (v < 0.0f) v *= slope;
  }
  return out;
}

Tensor handle_reshape(const Tensor& values, const json& cfg) {
  std::vector<int> target = cfg["target_shape"].get<std::vector<int>>();
  std::vector<int> out_shape = {values.shape.at(0)};
  out_shape.insert(out_shape.end(), target.begin(), target.end());
  Tensor out;
  out.shape = out_shape;
  out.data = values.data;
  return out;
}

void add_channel_bias_4d(Tensor& t, const Tensor& bias) {
  int bsz = t.shape[0], h = t.shape[1], w = t.shape[2], c = t.shape[3];
  for (int b = 0; b < bsz; ++b) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        for (int ch = 0; ch < c; ++ch) {
          t.set4(b, y, x, ch, t.get4(b, y, x, ch) + bias.get1(ch));
        }
      }
    }
  }
}

Tensor handle_conv2d(const std::string& layer_name, const Tensor& values, const json& cfg,
                     const std::unordered_map<std::string, std::vector<Tensor>>& weights) {
  int bsz = values.shape[0], in_h = values.shape[1], in_w = values.shape[2], in_c = values.shape[3];

  const auto& layer_w = weights.at(cfg.value("name", layer_name));
  const Tensor& kernel = layer_w.at(0);

  auto ksize = cfg["kernel_size"].get<std::vector<int>>();
  auto strides = cfg.value("strides", std::vector<int>{1, 1});
  auto dilation = cfg.value("dilation_rate", std::vector<int>{1, 1});
  int kh = ksize[0], kw = ksize[1];
  int sh = strides[0], sw = strides[1];
  int dh = dilation[0], dw = dilation[1];
  int filters = cfg["filters"].get<int>();
  int groups = cfg.value("groups", 1);

  int channels_per_group = in_c / groups;
  int filters_per_group = filters / groups;
  int eff_h = (kh - 1) * dh + 1;
  int eff_w = (kw - 1) * dw + 1;

  std::string padding = cfg.value("padding", std::string("valid"));
  int out_h = 0, out_w = 0, pad_top = 0, pad_left = 0;
  if (padding == "valid") {
    out_h = (in_h - eff_h) / sh + 1;
    out_w = (in_w - eff_w) / sw + 1;
  } else if (padding == "same") {
    out_h = ceil_int(static_cast<float>(in_h) / sh);
    out_w = ceil_int(static_cast<float>(in_w) / sw);
    int total_h = std::max((out_h - 1) * sh + eff_h - in_h, 0);
    int total_w = std::max((out_w - 1) * sw + eff_w - in_w, 0);
    pad_top = total_h / 2;
    pad_left = total_w / 2;
  } else {
    throw std::runtime_error("Unsupported Conv2D padding: " + padding);
  }

  Tensor out({bsz, out_h, out_w, filters});

  for (int b = 0; b < bsz; ++b) {
    for (int oy = 0; oy < out_h; ++oy) {
      int in_y_base = oy * sh - pad_top;
      for (int ox = 0; ox < out_w; ++ox) {
        int in_x_base = ox * sw - pad_left;
        for (int ky = 0; ky < kh; ++ky) {
          int in_y = in_y_base + ky * dh;
          if (in_y < 0 || in_y >= in_h) continue;
          for (int kx = 0; kx < kw; ++kx) {
            int in_x = in_x_base + kx * dw;
            if (in_x < 0 || in_x >= in_w) continue;
            for (int g = 0; g < groups; ++g) {
              int in_start = g * channels_per_group;
              int in_end = in_start + channels_per_group;
              int f_start = g * filters_per_group;
              int f_end = f_start + filters_per_group;
              for (int f = f_start; f < f_end; ++f) {
                float acc = out.get4(b, oy, ox, f);
                for (int c = in_start; c < in_end; ++c) {
                  acc += values.get4(b, in_y, in_x, c) * kernel.get4(ky, kx, c - in_start, f);
                }
                out.set4(b, oy, ox, f, acc);
              }
            }
          }
        }
      }
    }
  }

  if (cfg.value("use_bias", true)) add_channel_bias_4d(out, layer_w.at(1));
  apply_activation_in_place(out, cfg.value("activation", std::string("linear")));
  return out;
}

Tensor handle_conv2d_transpose(const std::string& layer_name, const Tensor& values, const json& cfg,
                               const std::unordered_map<std::string, std::vector<Tensor>>& weights) {
  int bsz = values.shape[0], in_h = values.shape[1], in_w = values.shape[2], in_c = values.shape[3];
  const auto& layer_w = weights.at(cfg.value("name", layer_name));
  const Tensor& kernel = layer_w.at(0);

  auto ksize = cfg["kernel_size"].get<std::vector<int>>();
  auto strides = cfg["strides"].get<std::vector<int>>();
  int kh = ksize[0], kw = ksize[1], sh = strides[0], sw = strides[1];
  int filters = cfg["filters"].get<int>();

  std::string padding = cfg.value("padding", std::string("valid"));
  int out_h, out_w, pad_top = 0, pad_left = 0;
  if (padding == "same") {
    out_h = in_h * sh;
    out_w = in_w * sw;
    pad_top = std::max(kh - sh, 0) / 2;
    pad_left = std::max(kw - sw, 0) / 2;
  } else if (padding == "valid") {
    out_h = (in_h - 1) * sh + kh;
    out_w = (in_w - 1) * sw + kw;
  } else {
    throw std::runtime_error("Unsupported Conv2DTranspose padding: " + padding);
  }

  Tensor out({bsz, out_h, out_w, filters});
  for (int b = 0; b < bsz; ++b) {
    for (int iy = 0; iy < in_h; ++iy) {
      int base_y = iy * sh;
      for (int ix = 0; ix < in_w; ++ix) {
        int base_x = ix * sw;
        for (int ky = 0; ky < kh; ++ky) {
          int oy = base_y + ky - pad_top;
          if (oy < 0 || oy >= out_h) continue;
          for (int kx = 0; kx < kw; ++kx) {
            int ox = base_x + kx - pad_left;
            if (ox < 0 || ox >= out_w) continue;
            for (int f = 0; f < filters; ++f) {
              float acc = out.get4(b, oy, ox, f);
              for (int c = 0; c < in_c; ++c) {
                acc += values.get4(b, iy, ix, c) * kernel.get4(ky, kx, f, c);
              }
              out.set4(b, oy, ox, f, acc);
            }
          }
        }
      }
    }
  }

  if (cfg.value("use_bias", true)) add_channel_bias_4d(out, layer_w.at(1));
  apply_activation_in_place(out, cfg.value("activation", std::string("linear")));
  return out;
}

Tensor handle_resize(const Tensor& values, const json& cfg) {
  int target_h = cfg["height"].get<int>();
  int target_w = cfg["width"].get<int>();
  int bsz = values.shape[0], in_h = values.shape[1], in_w = values.shape[2], ch = values.shape[3];

  Tensor out({bsz, target_h, target_w, ch});
  for (int b = 0; b < bsz; ++b) {
    for (int oy = 0; oy < target_h; ++oy) {
      float y = oy * (static_cast<float>(in_h) / target_h);
      int y0 = std::min(floor_int(y), in_h - 1);
      int y1 = std::min(y0 + 1, in_h - 1);
      float wy = y - y0;
      for (int ox = 0; ox < target_w; ++ox) {
        float x = ox * (static_cast<float>(in_w) / target_w);
        int x0 = std::min(floor_int(x), in_w - 1);
        int x1 = std::min(x0 + 1, in_w - 1);
        float wx = x - x0;
        for (int c = 0; c < ch; ++c) {
          float top = (1.0f - wx) * values.get4(b, y0, x0, c) + wx * values.get4(b, y0, x1, c);
          float bottom = (1.0f - wx) * values.get4(b, y1, x0, c) + wx * values.get4(b, y1, x1, c);
          out.set4(b, oy, ox, c, (1.0f - wy) * top + wy * bottom);
        }
      }
    }
  }
  return out;
}

int to_u8(float v) {
  int scaled = static_cast<int>(std::lround((v + 1.0f) * 127.5f));
  return std::clamp(scaled, 0, 255);
}

void save_as_ppm(const Tensor& image_4d, const std::filesystem::path& path) {
  int h = image_4d.shape[1];
  int w = image_4d.shape[2];
  if (h != kOutputImageSize || w != kOutputImageSize) {
    throw std::runtime_error("Expected fixed output image size 100x100.");
  }
  std::ofstream out(path, std::ios::binary);
  out << "P6\n" << w << " " << h << "\n255\n";
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      unsigned char rgb[3] = {
          static_cast<unsigned char>(to_u8(image_4d.get4(0, y, x, 0))),
          static_cast<unsigned char>(to_u8(image_4d.get4(0, y, x, 1))),
          static_cast<unsigned char>(to_u8(image_4d.get4(0, y, x, 2))),
      };
      out.write(reinterpret_cast<char*>(rgb), 3);
    }
  }
}

Tensor recreate_layer(const std::string& name, const json& cfg, const Tensor& in,
                     const std::unordered_map<std::string, std::vector<Tensor>>& weights) {
  if (name.rfind("input_layer", 0) == 0) return in;
  if (name.rfind("dense", 0) == 0) return handle_dense(name, in, cfg, weights);
  if (name.rfind("batch_normalization", 0) == 0) return handle_batch_norm(name, in, cfg, weights);
  if (name.rfind("leaky_re_lu", 0) == 0) return handle_leaky_relu(in, cfg);
  if (name.rfind("reshape", 0) == 0) return handle_reshape(in, cfg);
  if (name.rfind("conv2d_transpose", 0) == 0) return handle_conv2d_transpose(name, in, cfg, weights);
  if (name.rfind("conv2d", 0) == 0) return handle_conv2d(name, in, cfg, weights);
  if (name.rfind("resizing", 0) == 0) return handle_resize(in, cfg);
  throw std::runtime_error("Unsupported layer: " + name);
}

}  // namespace

int main() {
  try {
    const AppConfig config = load_app_config();
    ModelArtifacts artifacts = load_model_from_artifacts(config.artifacts_dir);

    Tensor current({1, kInputSize});
    current.data.assign(kCustomInputVector.begin(), kCustomInputVector.end());

    save_json_value(config.recreated_dir, 0, "original", tensor_to_json(current));

    int layer_index = 1;
    for (const auto& layer : artifacts.layer_defs) {
      std::string name = layer["name"].get<std::string>();
      current = recreate_layer(name, layer["config"], current, artifacts.layer_weights);
      save_json_value(config.recreated_dir, layer_index++, name, tensor_to_json(current));
    }

    if (current.shape.size() == 4 && current.shape[0] > 0 && current.shape[3] == 3) {
      std::filesystem::path out_path = config.recreated_dir / "out.ppm";
      save_as_ppm(current, out_path);
      std::cout << "Saved recreated image to: " << out_path << "\n";
    }

    std::cout << "Saved recreated values to: " << config.recreated_dir << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}