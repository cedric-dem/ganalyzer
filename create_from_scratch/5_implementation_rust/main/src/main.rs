use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

struct AppConfig {
    artifacts_dir: PathBuf,
    recreated_dir: PathBuf,
}

fn trim(value: &str) -> &str {
    value.trim_matches([' ', '\t', '\r', '\n'])
}

fn parse_toml_string_value(value: &str) -> Result<String, String> {
    let trimmed = trim(value);
    if !trimmed.starts_with('"') {
        return Err(format!("Expected quoted TOML string value: {value}"));
    }

    let mut parsed = String::new();
    let mut escaping = false;
    for ch in trimmed[1..].chars() {
        if escaping {
            match ch {
                '"' | '\\' => parsed.push(ch),
                'n' => parsed.push('\n'),
                'r' => parsed.push('\r'),
                't' => parsed.push('\t'),
                other => parsed.push(other),
            }
            escaping = false;
            continue;
        }

        if ch == '\\' {
            escaping = true;
            continue;
        }
        if ch == '"' {
            return Ok(parsed);
        }
        parsed.push(ch);
    }

    Err(format!("Unterminated TOML string value: {value}"))
}

fn resolve_config_path() -> Result<PathBuf, String> {
    let candidates = [
        PathBuf::from("config.toml"),
        PathBuf::from("../config.toml"),
        PathBuf::from("../../config.toml"),
        PathBuf::from("create_from_scratch/config.toml"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(String::from(
        "Could not find config.toml. Run from create_from_scratch/, 4_rust_implementation/, 4_rust_implementation/main/, or the repository root.",
    ))
}

fn resolve_config_relative_path(config_path: &Path, configured_path: &str) -> PathBuf {
    let path = PathBuf::from(configured_path);
    if path.is_absolute() {
        return path;
    }
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

fn load_app_config() -> Result<AppConfig, String> {
    let config_path = resolve_config_path()?;
    let contents = fs::read_to_string(&config_path)
        .map_err(|e| format!("Could not open config file {}: {e}", config_path.display()))?;

    let mut values = HashMap::new();
    for raw_line in contents.lines() {
        let line_without_comment = raw_line.split_once('#').map_or(raw_line, |(line, _)| line);
        let line = trim(line_without_comment);
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        values.insert(trim(key).to_string(), parse_toml_string_value(value)?);
    }

    let artifacts_dir = values
        .get("model_split_into_files")
        .ok_or_else(|| String::from("Missing model_split_into_files in config.toml"))?;
    let recreated_dir = values
        .get("path_to_rust_implementation_intermediary")
        .ok_or_else(|| {
            String::from("Missing path_to_rust_implementation_intermediary in config.toml")
        })?;

    Ok(AppConfig {
        artifacts_dir: resolve_config_relative_path(&config_path, artifacts_dir),
        recreated_dir: resolve_config_relative_path(&config_path, recreated_dir),
    })
}

const CUSTOM_INPUT_VECTOR: [f32; 49] = [
    -0.16, -0.15, 0.19, -0.14, 0.05, 0.66, -0.07, -0.06, -0.04, -0.0, 0.51, -0.05, 0.23, -0.17,
    -0.06, 0.07, 0.39, 0.23, -0.08, 0.02, -0.14, -0.16, 0.0, 0.05, 0.09, 0.15, 0.09, -0.12, -0.54,
    0.18, -0.44, -0.13, -0.2, 0.03, -0.23, -0.14, -0.46, 0.2, 0.03, -0.36, 0.42, -0.12, -0.19,
    -0.25, -0.23, -0.04, -0.18, 0.18, -0.19,
];

#[derive(Clone, Default)]
struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    fn zeros(shape: Vec<usize>) -> Self {
        let total = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; total],
        }
    }

    fn offset(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len(), "Index rank mismatch");
        let mut off = 0usize;
        let mut stride = 1usize;
        for i in (0..self.shape.len()).rev() {
            off += idx[i] * stride;
            stride *= self.shape[i];
        }
        off
    }

    fn get(&self, idx: &[usize]) -> f32 {
        self.data[self.offset(idx)]
    }

    fn set(&mut self, idx: &[usize], v: f32) {
        let o = self.offset(idx);
        self.data[o] = v;
    }
}

fn apply_activation(v: f32, activation: &str) -> f32 {
    match activation {
        "linear" => v,
        "relu" => v.max(0.0),
        "tanh" => v.tanh(),
        "sigmoid" => 1.0 / (1.0 + (-v).exp()),
        _ => panic!("Unsupported activation: {activation}"),
    }
}

fn apply_activation_in_place(t: &mut Tensor, activation: &str) {
    for v in &mut t.data {
        *v = apply_activation(*v, activation);
    }
}

fn infer_shape(j: &Value) -> Vec<usize> {
    if !j.is_array() {
        return vec![];
    }
    let mut shape = vec![];
    let mut cur = j;
    while let Some(arr) = cur.as_array() {
        shape.push(arr.len());
        if arr.is_empty() {
            break;
        }
        cur = &arr[0];
    }
    shape
}

fn flatten_json(j: &Value, out: &mut Vec<f32>) {
    if let Some(arr) = j.as_array() {
        for v in arr {
            flatten_json(v, out);
        }
    } else {
        out.push(j.as_f64().unwrap() as f32);
    }
}

fn tensor_from_json(j: &Value) -> Tensor {
    let mut t = Tensor {
        shape: infer_shape(j),
        data: vec![],
    };
    flatten_json(j, &mut t.data);
    t
}

fn tensor_to_json_rec(t: &Tensor, dim: usize, cursor: &mut usize) -> Value {
    if dim == t.shape.len() {
        let v = t.data[*cursor];
        *cursor += 1;
        return json!(v);
    }
    let mut arr = Vec::with_capacity(t.shape[dim]);
    for _ in 0..t.shape[dim] {
        arr.push(tensor_to_json_rec(t, dim + 1, cursor));
    }
    Value::Array(arr)
}

fn tensor_to_json(t: &Tensor) -> Value {
    let mut cursor = 0;
    tensor_to_json_rec(t, 0, &mut cursor)
}

fn save_json_value(base_dir: &Path, index: usize, label: &str, value: &Value) {
    fs::create_dir_all(base_dir).unwrap();
    let safe = label.replace('/', "_").replace(' ', "_");
    let path = base_dir.join(format!("values_{index:03}_{safe}.txt"));
    fs::write(path, value.to_string()).unwrap();
}

fn expected_weight_count_for_layer(class_name: &str, cfg: &Value) -> usize {
    match class_name {
        "Dense" | "Conv2D" | "Conv2DTranspose" => {
            if cfg
                .get("use_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(true)
            {
                2
            } else {
                1
            }
        }
        "BatchNormalization" => {
            let mut count = 0;
            if cfg.get("scale").and_then(|v| v.as_bool()).unwrap_or(true) {
                count += 1;
            }
            if cfg.get("center").and_then(|v| v.as_bool()).unwrap_or(true) {
                count += 1;
            }
            count + 2
        }
        _ => 0,
    }
}

struct ModelArtifacts {
    layer_defs: Vec<Value>,
    layer_weights: HashMap<String, Vec<Tensor>>,
}

fn load_model_from_artifacts(artifacts_dir: &Path) -> ModelArtifacts {
    let config: Value =
        serde_json::from_str(&fs::read_to_string(artifacts_dir.join("config.json")).unwrap())
            .unwrap();
    let weights_payload: Value = serde_json::from_str(
        &fs::read_to_string(artifacts_dir.join("model.weights.json")).unwrap(),
    )
    .unwrap();

    let layer_defs = config["config"]["layers"].as_array().unwrap().clone();
    let flat_weights = weights_payload["weights"].as_array().unwrap();

    let mut layer_weights = HashMap::new();
    let mut cursor = 0usize;
    for layer in &layer_defs {
        let name = layer["name"].as_str().unwrap().to_string();
        let class_name = layer["class_name"].as_str().unwrap();
        let expected = expected_weight_count_for_layer(class_name, &layer["config"]);
        let mut arr = Vec::with_capacity(expected);
        for i in 0..expected {
            arr.push(tensor_from_json(&flat_weights[cursor + i]));
        }
        cursor += expected;
        layer_weights.insert(name, arr);
    }

    ModelArtifacts {
        layer_defs,
        layer_weights,
    }
}

fn handle_dense(
    layer_name: &str,
    values: &Tensor,
    cfg: &Value,
    weights: &HashMap<String, Vec<Tensor>>,
) -> Tensor {
    let batch = values.shape[0];
    let input_width = values.shape[1];
    let units = cfg["units"].as_u64().unwrap() as usize;
    let use_bias = cfg
        .get("use_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let mut out = Tensor::zeros(vec![batch, units]);

    let w = weights
        .get(
            cfg.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(layer_name),
        )
        .unwrap();
    let kernel = &w[0];
    let bias = if use_bias { Some(&w[1]) } else { None };

    for b in 0..batch {
        for u in 0..units {
            let mut acc = 0.0f32;
            for i in 0..input_width {
                acc += values.get(&[b, i]) * kernel.get(&[i, u]);
            }
            if let Some(bias_t) = bias {
                acc += bias_t.get(&[u]);
            }
            out.set(&[b, u], acc);
        }
    }
    out
}

fn handle_batch_norm(
    layer_name: &str,
    values: &Tensor,
    cfg: &Value,
    weights: &HashMap<String, Vec<Tensor>>,
) -> Tensor {
    let w = weights.get(layer_name).unwrap();
    let gamma = &w[0];
    let beta = &w[1];
    let moving_mean = &w[2];
    let moving_var = &w[3];
    let eps = cfg.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-3) as f32;

    let mut out = values.clone();
    let channels = *values.shape.last().unwrap();
    let spatial: usize = values.shape[..values.shape.len() - 1].iter().product();
    for s in 0..spatial {
        for c in 0..channels {
            let idx = s * channels + c;
            let norm =
                (values.data[idx] - moving_mean.get(&[c])) / (moving_var.get(&[c]) + eps).sqrt();
            out.data[idx] = gamma.get(&[c]) * norm + beta.get(&[c]);
        }
    }
    out
}

fn handle_leaky_relu(values: &Tensor, cfg: &Value) -> Tensor {
    let slope = cfg
        .get("negative_slope")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.3) as f32;
    let mut out = values.clone();
    for v in &mut out.data {
        if *v < 0.0 {
            *v *= slope;
        }
    }
    out
}

fn handle_reshape(values: &Tensor, cfg: &Value) -> Tensor {
    let target = cfg["target_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let mut shape = vec![values.shape[0]];
    shape.extend(target);
    Tensor {
        shape,
        data: values.data.clone(),
    }
}

fn add_channel_bias_4d(t: &mut Tensor, bias: &Tensor) {
    let (bsz, h, w, c) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    for b in 0..bsz {
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    let v = t.get(&[b, y, x, ch]) + bias.get(&[ch]);
                    t.set(&[b, y, x, ch], v);
                }
            }
        }
    }
}

fn handle_conv2d(
    layer_name: &str,
    values: &Tensor,
    cfg: &Value,
    weights: &HashMap<String, Vec<Tensor>>,
) -> Tensor {
    let (bsz, in_h, in_w, in_c) = (
        values.shape[0],
        values.shape[1],
        values.shape[2],
        values.shape[3],
    );
    let w = weights
        .get(
            cfg.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(layer_name),
        )
        .unwrap();
    let kernel = &w[0];

    let ksize = cfg["kernel_size"].as_array().unwrap();
    let kh = ksize[0].as_u64().unwrap() as usize;
    let kw = ksize[1].as_u64().unwrap() as usize;

    let strides = cfg
        .get("strides")
        .and_then(|v| v.as_array())
        .map(|a| {
            vec![
                a[0].as_u64().unwrap() as usize,
                a[1].as_u64().unwrap() as usize,
            ]
        })
        .unwrap_or_else(|| vec![1, 1]);
    let dilation = cfg
        .get("dilation_rate")
        .and_then(|v| v.as_array())
        .map(|a| {
            vec![
                a[0].as_u64().unwrap() as usize,
                a[1].as_u64().unwrap() as usize,
            ]
        })
        .unwrap_or_else(|| vec![1, 1]);

    let (sh, sw) = (strides[0], strides[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let filters = cfg["filters"].as_u64().unwrap() as usize;
    let groups = cfg.get("groups").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

    let channels_per_group = in_c / groups;
    let filters_per_group = filters / groups;
    let eff_h = (kh - 1) * dh + 1;
    let eff_w = (kw - 1) * dw + 1;

    let padding = cfg
        .get("padding")
        .and_then(|v| v.as_str())
        .unwrap_or("valid");
    let (out_h, out_w, pad_top, pad_left) = if padding == "valid" {
        (
            (in_h - eff_h) / sh + 1,
            (in_w - eff_w) / sw + 1,
            0usize,
            0usize,
        )
    } else if padding == "same" {
        let out_h = (in_h as f32 / sh as f32).ceil() as usize;
        let out_w = (in_w as f32 / sw as f32).ceil() as usize;
        let total_h = ((out_h - 1) * sh + eff_h).saturating_sub(in_h);
        let total_w = ((out_w - 1) * sw + eff_w).saturating_sub(in_w);
        (out_h, out_w, total_h / 2, total_w / 2)
    } else {
        panic!("Unsupported Conv2D padding: {padding}");
    };

    let mut out = Tensor::zeros(vec![bsz, out_h, out_w, filters]);

    for b in 0..bsz {
        for oy in 0..out_h {
            let in_y_base = oy as isize * sh as isize - pad_top as isize;
            for ox in 0..out_w {
                let in_x_base = ox as isize * sw as isize - pad_left as isize;
                for ky in 0..kh {
                    let in_y = in_y_base + ky as isize * dh as isize;
                    if in_y < 0 || in_y >= in_h as isize {
                        continue;
                    }
                    for kx in 0..kw {
                        let in_x = in_x_base + kx as isize * dw as isize;
                        if in_x < 0 || in_x >= in_w as isize {
                            continue;
                        }
                        for g in 0..groups {
                            let in_start = g * channels_per_group;
                            let in_end = in_start + channels_per_group;
                            let f_start = g * filters_per_group;
                            let f_end = f_start + filters_per_group;
                            for f in f_start..f_end {
                                let mut acc = out.get(&[b, oy, ox, f]);
                                for c in in_start..in_end {
                                    acc += values.get(&[b, in_y as usize, in_x as usize, c])
                                        * kernel.get(&[ky, kx, c - in_start, f]);
                                }
                                out.set(&[b, oy, ox, f], acc);
                            }
                        }
                    }
                }
            }
        }
    }

    if cfg
        .get("use_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
    {
        add_channel_bias_4d(&mut out, &w[1]);
    }
    apply_activation_in_place(
        &mut out,
        cfg.get("activation")
            .and_then(|v| v.as_str())
            .unwrap_or("linear"),
    );
    out
}

fn handle_conv2d_transpose(
    layer_name: &str,
    values: &Tensor,
    cfg: &Value,
    weights: &HashMap<String, Vec<Tensor>>,
) -> Tensor {
    let (bsz, in_h, in_w, in_c) = (
        values.shape[0],
        values.shape[1],
        values.shape[2],
        values.shape[3],
    );
    let w = weights
        .get(
            cfg.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(layer_name),
        )
        .unwrap();
    let kernel = &w[0];

    let ksize = cfg["kernel_size"].as_array().unwrap();
    let strides = cfg["strides"].as_array().unwrap();
    let (kh, kw) = (
        ksize[0].as_u64().unwrap() as usize,
        ksize[1].as_u64().unwrap() as usize,
    );
    let (sh, sw) = (
        strides[0].as_u64().unwrap() as usize,
        strides[1].as_u64().unwrap() as usize,
    );
    let filters = cfg["filters"].as_u64().unwrap() as usize;

    let padding = cfg
        .get("padding")
        .and_then(|v| v.as_str())
        .unwrap_or("valid");
    let (out_h, out_w, pad_top, pad_left) = if padding == "same" {
        (
            in_h * sh,
            in_w * sw,
            (kh.saturating_sub(sh)) / 2,
            (kw.saturating_sub(sw)) / 2,
        )
    } else if padding == "valid" {
        ((in_h - 1) * sh + kh, (in_w - 1) * sw + kw, 0usize, 0usize)
    } else {
        panic!("Unsupported Conv2DTranspose padding: {padding}");
    };

    let mut out = Tensor::zeros(vec![bsz, out_h, out_w, filters]);

    for b in 0..bsz {
        for iy in 0..in_h {
            let base_y = iy * sh;
            for ix in 0..in_w {
                let base_x = ix * sw;
                for ky in 0..kh {
                    let oy = base_y as isize + ky as isize - pad_top as isize;
                    if oy < 0 || oy >= out_h as isize {
                        continue;
                    }
                    for kx in 0..kw {
                        let ox = base_x as isize + kx as isize - pad_left as isize;
                        if ox < 0 || ox >= out_w as isize {
                            continue;
                        }
                        for f in 0..filters {
                            let mut acc = out.get(&[b, oy as usize, ox as usize, f]);
                            for c in 0..in_c {
                                acc += values.get(&[b, iy, ix, c]) * kernel.get(&[ky, kx, f, c]);
                            }
                            out.set(&[b, oy as usize, ox as usize, f], acc);
                        }
                    }
                }
            }
        }
    }

    if cfg
        .get("use_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
    {
        add_channel_bias_4d(&mut out, &w[1]);
    }
    apply_activation_in_place(
        &mut out,
        cfg.get("activation")
            .and_then(|v| v.as_str())
            .unwrap_or("linear"),
    );
    out
}

fn handle_resize(values: &Tensor, cfg: &Value) -> Tensor {
    let target_h = cfg["height"].as_u64().unwrap() as usize;
    let target_w = cfg["width"].as_u64().unwrap() as usize;
    let (bsz, in_h, in_w, ch) = (
        values.shape[0],
        values.shape[1],
        values.shape[2],
        values.shape[3],
    );

    let mut out = Tensor::zeros(vec![bsz, target_h, target_w, ch]);
    for b in 0..bsz {
        for oy in 0..target_h {
            let y = oy as f32 * (in_h as f32 / target_h as f32);
            let y0 = y.floor() as usize;
            let y0 = y0.min(in_h - 1);
            let y1 = (y0 + 1).min(in_h - 1);
            let wy = y - y0 as f32;
            for ox in 0..target_w {
                let x = ox as f32 * (in_w as f32 / target_w as f32);
                let x0 = (x.floor() as usize).min(in_w - 1);
                let x1 = (x0 + 1).min(in_w - 1);
                let wx = x - x0 as f32;
                for c in 0..ch {
                    let top =
                        (1.0 - wx) * values.get(&[b, y0, x0, c]) + wx * values.get(&[b, y0, x1, c]);
                    let bottom =
                        (1.0 - wx) * values.get(&[b, y1, x0, c]) + wx * values.get(&[b, y1, x1, c]);
                    out.set(&[b, oy, ox, c], (1.0 - wy) * top + wy * bottom);
                }
            }
        }
    }
    out
}

fn to_u8(v: f32) -> u8 {
    let scaled = ((v + 1.0) * 127.5).round() as i32;
    scaled.clamp(0, 255) as u8
}

fn save_as_ppm(image_4d: &Tensor, path: &Path) {
    let h = image_4d.shape[1];
    let w = image_4d.shape[2];
    let mut out = File::create(path).unwrap();
    write!(out, "P6\n{} {}\n255\n", w, h).unwrap();
    for y in 0..h {
        for x in 0..w {
            let rgb = [
                to_u8(image_4d.get(&[0, y, x, 0])),
                to_u8(image_4d.get(&[0, y, x, 1])),
                to_u8(image_4d.get(&[0, y, x, 2])),
            ];
            out.write_all(&rgb).unwrap();
        }
    }
}

fn recreate_layer(
    name: &str,
    cfg: &Value,
    input: &Tensor,
    weights: &HashMap<String, Vec<Tensor>>,
) -> Tensor {
    if name.starts_with("input_layer") {
        return input.clone();
    }
    if name.starts_with("dense") {
        return handle_dense(name, input, cfg, weights);
    }
    if name.starts_with("batch_normalization") {
        return handle_batch_norm(name, input, cfg, weights);
    }
    if name.starts_with("leaky_re_lu") {
        return handle_leaky_relu(input, cfg);
    }
    if name.starts_with("reshape") {
        return handle_reshape(input, cfg);
    }
    if name.starts_with("conv2d_transpose") {
        return handle_conv2d_transpose(name, input, cfg, weights);
    }
    if name.starts_with("conv2d") {
        return handle_conv2d(name, input, cfg, weights);
    }
    if name.starts_with("resizing") {
        return handle_resize(input, cfg);
    }
    panic!("Unsupported layer: {name}");
}

fn main() {
    let config = load_app_config().unwrap_or_else(|e| panic!("{e}"));
    let artifacts = load_model_from_artifacts(&config.artifacts_dir);

    let mut current = Tensor::zeros(vec![1, CUSTOM_INPUT_VECTOR.len()]);
    current.data = CUSTOM_INPUT_VECTOR.to_vec();

    save_json_value(
        &config.recreated_dir,
        0,
        "original",
        &tensor_to_json(&current),
    );

    let mut layer_index = 1usize;
    for layer in &artifacts.layer_defs {
        let name = layer["name"].as_str().unwrap();
        current = recreate_layer(name, &layer["config"], &current, &artifacts.layer_weights);
        save_json_value(
            &config.recreated_dir,
            layer_index,
            name,
            &tensor_to_json(&current),
        );
        layer_index += 1;
    }

    if current.shape.len() == 4 && current.shape[0] > 0 && current.shape[3] == 3 {
        let out_path = config.recreated_dir.join("out.ppm");
        save_as_ppm(&current, &out_path);
        println!("Saved recreated image to: {}", out_path.display());
    }

    println!(
        "Saved recreated values to: {}",
        config.recreated_dir.display()
    );
}