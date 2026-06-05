import json

from PIL import Image
import tomllib


with open("config.toml", "rb") as f:
	config = tomllib.load(f)
model_split_into_files = config["model_split_into_files"]

path_to_non_keras_implementation_intermediary = config["path_to_non_keras_implementation_intermediary"]


_RECREATION_CACHE = None
_LAYER_RECREATION_WEIGHTS_CACHE = None

CUSTOM_INPUT_VECTOR = [[
    -0.16, -0.15, +0.19, -0.14, +0.05, +0.66, -0.07,
    -0.06, -0.04, -0.0, +0.51, -0.05, +0.23, -0.17,
    -0.06, +0.07, +0.39, +0.23, -0.08, +0.02, -0.14,
    -0.16, +0.0, +0.05, +0.09, +0.15, +0.09, -0.12,
    -0.54, +0.18, -0.44, -0.13, -0.2, +0.03, -0.23,
    -0.14, -0.46, +0.2, +0.03, -0.36, +0.42, -0.12,
    -0.19, -0.25, -0.23, -0.04, -0.18, +0.18, -0.19,
]]

ARTIFACTS_DIR = model_split_into_files
RECREATED_DIR = path_to_non_keras_implementation_intermediary


def _deep_copy(value):
    return json.loads(json.dumps(value))


def _floor(value: float) -> int:
    integer_part = int(value)
    if value >= 0.0 or value == integer_part:
        return integer_part
    return integer_part - 1


def _ceil(value: float) -> int:
    integer_part = int(value)
    if value <= 0.0 or value == integer_part:
        return integer_part
    return integer_part + 1


def _exp(value: float) -> float:
    x = float(value)
    if x == 0.0:
        return 1.0
    if x < 0.0:
        return 1.0 / _exp(-x)
    if x > 1.0:
        half = _exp(x / 2.0)
        return half * half

    term = 1.0
    total = 1.0
    for n in range(1, 30):
        term *= x / n
        total += term
    return total


def _sqrt(value: float) -> float:
    x = float(value)
    if x < 0.0:
        raise ValueError(f"Cannot take square root of negative value: {value}")
    if x == 0.0:
        return 0.0
    guess = x if x >= 1.0 else 1.0
    for _ in range(20):
        guess = 0.5 * (guess + x / guess)
    return guess


def _tanh(value: float) -> float:
    x = float(value)
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    exp_pos = _exp(x)
    exp_neg = _exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def _shape_of(value):
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def _zeros(shape):
    if not shape:
        return 0.0
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def _flatten(value):
    if not isinstance(value, list):
        return [float(value)]
    out = []
    for item in value:
        out.extend(_flatten(item))
    return out


def _unflatten(flat, shape):
    cursor = 0

    def build(level):
        nonlocal cursor
        if level == len(shape):
            v = flat[cursor]
            cursor += 1
            return v
        return [build(level + 1) for _ in range(shape[level])]

    rebuilt = build(0)
    if cursor != len(flat):
        raise ValueError("Reshape error: unused flattened values remain.")
    return rebuilt


def _format_value(value):
    return json.dumps(value, ensure_ascii=False)


def save_value(base_dir, index, label, value):
    safe_label = str(label).replace("/", "_").replace(" ", "_")
    output_path = f"{base_dir}/values_{index:03d}_{safe_label}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_format_value(value))


def save_recreated_value(index, label, value):
    save_value(RECREATED_DIR, index, label, value)


def _expected_weight_count_for_layer(layer_class_name, layer_config):
    if layer_class_name in {"Dense", "Conv2D", "Conv2DTranspose"}:
        return 2 if bool(layer_config.get("use_bias", True)) else 1
    if layer_class_name == "BatchNormalization":
        count = 0
        if bool(layer_config.get("scale", True)):
            count += 1
        if bool(layer_config.get("center", True)):
            count += 1
        return count + 2
    return 0


def load_model_from_artifacts():
    config_path = f"{ARTIFACTS_DIR}/config.json"
    metadata_path = f"{ARTIFACTS_DIR}/metadata.json"
    weights_path = f"{ARTIFACTS_DIR}/model.weights.json"

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(weights_path, encoding="utf-8") as f:
        weights_payload = json.load(f)

    flat_weights = [_deep_copy(w) for w in weights_payload["weights"]]
    layer_definitions = config["config"]["layers"]

    layer_weights = {}
    weight_cursor = 0
    for layer_definition in layer_definitions:
        layer_name = layer_definition["name"]
        layer_class_name = layer_definition["class_name"]
        layer_config = layer_definition["config"]
        expected_count = _expected_weight_count_for_layer(layer_class_name, layer_config)
        if expected_count == 0:
            layer_weights[layer_name] = []
            continue
        next_cursor = weight_cursor + expected_count
        layer_weights[layer_name] = flat_weights[weight_cursor:next_cursor]
        weight_cursor = next_cursor

    return layer_definitions, layer_weights, metadata


def _apply_activation(value: float, activation: str) -> float:
    if activation == "linear":
        return value
    if activation == "relu":
        return value if value > 0.0 else 0.0
    if activation == "tanh":
        return _tanh(value)
    if activation == "sigmoid":
        return 1.0 / (1.0 + _exp(-value))
    raise NotImplementedError(f"Unsupported activation: {activation}")


def _apply_activation_tensor(tensor, activation: str):
    if not isinstance(tensor, list):
        return _apply_activation(float(tensor), activation)
    return [_apply_activation_tensor(item, activation) for item in tensor]


def handleDenseLayer(layer_name, values, settings):
    global _RECREATION_CACHE
    current_values = values
    if _RECREATION_CACHE is None:
        with open(f"{ARTIFACTS_DIR}/model.weights.json", encoding="utf-8") as f:
            payload = json.load(f)
        _RECREATION_CACHE = payload["weights"]

    units = int(settings["units"])
    use_bias = bool(settings.get("use_bias", True))
    input_width = len(current_values[0])

    kernel = next((w for w in _RECREATION_CACHE if _shape_of(w) == (input_width, units)), None)
    if kernel is None:
        raise ValueError(f"Could not find dense kernel with shape {(input_width, units)}")

    output = []
    for row in current_values:
        out_row = []
        for unit_idx in range(units):
            acc = 0.0
            for in_idx, in_val in enumerate(row):
                acc += float(in_val) * float(kernel[in_idx][unit_idx])
            out_row.append(acc)
        output.append(out_row)

    if use_bias:
        bias = next((w for w in _RECREATION_CACHE if _shape_of(w) == (units,)), None)
        if bias is None:
            raise ValueError(f"Dense layer expects bias of shape {(units,)}, but none was found.")
        for b in range(len(output)):
            for u in range(units):
                output[b][u] += float(bias[u])
    return output


def handleBatchNormalization(layer_name, values, settings):
    global _LAYER_RECREATION_WEIGHTS_CACHE
    if _LAYER_RECREATION_WEIGHTS_CACHE is None:
        _, _LAYER_RECREATION_WEIGHTS_CACHE, _ = load_model_from_artifacts()

    gamma, beta, moving_mean, moving_variance = _LAYER_RECREATION_WEIGHTS_CACHE.get(layer_name, [])
    epsilon = float(settings.get("epsilon", 1e-3))

    def recurse(v):
        if not isinstance(v, list):
            raise ValueError("BatchNorm expects at least 1D data")
        if v and not isinstance(v[0], list):
            out = []
            for i, x in enumerate(v):
                normalized = (float(x) - float(moving_mean[i])) / _sqrt(float(moving_variance[i]) + epsilon)
                out.append(float(gamma[i]) * normalized + float(beta[i]))
            return out
        return [recurse(item) for item in v]

    return recurse(values)


def handleLeakyReLu(layer_name, values, settings):
    negative_slope = float(settings.get("negative_slope", 0.3))

    def recurse(v):
        if isinstance(v, list):
            return [recurse(item) for item in v]
        val = float(v)
        return val if val >= 0.0 else val * negative_slope

    return recurse(values)


def handleReshape(layer_name, values, settings):
    target_shape = tuple(settings.get("target_shape", ()))
    batch_size = len(values)
    flat = _flatten(values)
    return _unflatten(flat, (batch_size, *target_shape))


def handleConv2D(layer_name, values, settings):
    global _LAYER_RECREATION_WEIGHTS_CACHE
    current_values = values
    bsz, input_h, input_w, input_channels = _shape_of(current_values)

    if _LAYER_RECREATION_WEIGHTS_CACHE is None:
        _, _LAYER_RECREATION_WEIGHTS_CACHE, _ = load_model_from_artifacts()

    layer_name_from_config = settings.get("name", layer_name)
    layer_weights = _LAYER_RECREATION_WEIGHTS_CACHE.get(layer_name_from_config, [])
    kernel = layer_weights[0]

    kernel_h, kernel_w = tuple(settings["kernel_size"])
    stride_h, stride_w = tuple(settings.get("strides", (1, 1)))
    dilation_h, dilation_w = tuple(settings.get("dilation_rate", (1, 1)))
    filters = int(settings["filters"])
    groups = int(settings.get("groups", 1))

    channels_per_group = input_channels // groups
    filters_per_group = filters // groups
    effective_kernel_h = (kernel_h - 1) * dilation_h + 1
    effective_kernel_w = (kernel_w - 1) * dilation_w + 1

    padding = settings.get("padding", "valid").lower()
    if padding == "valid":
        out_h = (input_h - effective_kernel_h) // stride_h + 1
        out_w = (input_w - effective_kernel_w) // stride_w + 1
        pad_top = pad_left = 0
    elif padding == "same":
        out_h = _ceil(input_h / stride_h)
        out_w = _ceil(input_w / stride_w)
        pad_h_total = max((out_h - 1) * stride_h + effective_kernel_h - input_h, 0)
        pad_w_total = max((out_w - 1) * stride_w + effective_kernel_w - input_w, 0)
        pad_top = pad_h_total // 2
        pad_left = pad_w_total // 2
    else:
        raise NotImplementedError(f"Unsupported Conv2D padding mode: {padding}")

    output = _zeros((bsz, out_h, out_w, filters))

    for b in range(bsz):
        for oy in range(out_h):
            in_y_base = oy * stride_h - pad_top
            for ox in range(out_w):
                in_x_base = ox * stride_w - pad_left
                for ky in range(kernel_h):
                    in_y = in_y_base + ky * dilation_h
                    if in_y < 0 or in_y >= input_h:
                        continue
                    for kx in range(kernel_w):
                        in_x = in_x_base + kx * dilation_w
                        if in_x < 0 or in_x >= input_w:
                            continue
                        for group_index in range(groups):
                            in_start = group_index * channels_per_group
                            in_end = in_start + channels_per_group
                            f_start = group_index * filters_per_group
                            f_end = f_start + filters_per_group
                            for f in range(f_start, f_end):
                                acc = output[b][oy][ox][f]
                                for c in range(in_start, in_end):
                                    acc += float(current_values[b][in_y][in_x][c]) * float(kernel[ky][kx][c - in_start][f])
                                output[b][oy][ox][f] = acc

    if settings.get("use_bias", True):
        _add_channel_bias_4d(output, layer_weights[1])

    return _apply_activation_tensor(output, settings.get("activation", "linear"))


def _add_channel_bias_4d(tensor, bias):
    bsz, height, width, channels = _shape_of(tensor)
    for b in range(bsz):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    tensor[b][y][x][c] += float(bias[c])


def handleConv2DTranspose(layer_name, values, settings):
    global _LAYER_RECREATION_WEIGHTS_CACHE
    current_values = values
    bsz, input_h, input_w, input_channels = _shape_of(current_values)

    if _LAYER_RECREATION_WEIGHTS_CACHE is None:
        _, _LAYER_RECREATION_WEIGHTS_CACHE, _ = load_model_from_artifacts()

    layer_weights = _LAYER_RECREATION_WEIGHTS_CACHE.get(settings.get("name", layer_name), [])
    kernel = layer_weights[0]

    kernel_h, kernel_w = tuple(settings["kernel_size"])
    stride_h, stride_w = tuple(settings["strides"])
    filters = int(settings["filters"])

    padding = settings.get("padding", "valid").lower()
    if padding == "same":
        output_h = input_h * stride_h
        output_w = input_w * stride_w
        pad_h_total = max(kernel_h - stride_h, 0)
        pad_w_total = max(kernel_w - stride_w, 0)
        pad_top = pad_h_total // 2
        pad_left = pad_w_total // 2
    elif padding == "valid":
        output_h = (input_h - 1) * stride_h + kernel_h
        output_w = (input_w - 1) * stride_w + kernel_w
        pad_top = 0
        pad_left = 0
    else:
        raise NotImplementedError(f"Unsupported Conv2DTranspose padding mode: {padding}")

    output = _zeros((bsz, output_h, output_w, filters))
    for b in range(bsz):
        for iy in range(input_h):
            base_y = iy * stride_h
            for ix in range(input_w):
                base_x = ix * stride_w
                for ky in range(kernel_h):
                    oy = base_y + ky - pad_top
                    if oy < 0 or oy >= output_h:
                        continue
                    for kx in range(kernel_w):
                        ox = base_x + kx - pad_left
                        if ox < 0 or ox >= output_w:
                            continue
                        for f in range(filters):
                            acc = output[b][oy][ox][f]
                            for c in range(input_channels):
                                # Keras stores Conv2DTranspose kernels as:
                                # (kernel_h, kernel_w, output_channels, input_channels)
                                # so indexing must be [f][c], not [c][f].
                                acc += float(current_values[b][iy][ix][c]) * float(kernel[ky][kx][f][c])
                            output[b][oy][ox][f] = acc

    if settings.get("use_bias", True):
        _add_channel_bias_4d(output, layer_weights[1])

    return _apply_activation_tensor(output, settings.get("activation", "linear"))


def handleResize(layer_name, values, settings):
    current_values = values
    target_h = int(settings.get("height"))
    target_w = int(settings.get("width"))
    bsz, input_h, input_w, channels = _shape_of(current_values)

    y_coords = [y * (input_h / target_h) for y in range(target_h)]
    x_coords = [x * (input_w / target_w) for x in range(target_w)]

    output = _zeros((bsz, target_h, target_w, channels))
    for b in range(bsz):
        for oy in range(target_h):
            y0 = min(_floor(y_coords[oy]), input_h - 1)
            y1 = min(y0 + 1, input_h - 1)
            wy = y_coords[oy] - y0
            for ox in range(target_w):
                x0 = min(_floor(x_coords[ox]), input_w - 1)
                x1 = min(x0 + 1, input_w - 1)
                wx = x_coords[ox] - x0
                for c in range(channels):
                    top = (1.0 - wx) * float(current_values[b][y0][x0][c]) + wx * float(current_values[b][y0][x1][c])
                    bottom = (1.0 - wx) * float(current_values[b][y1][x0][c]) + wx * float(current_values[b][y1][x1][c])
                    output[b][oy][ox][c] = (1.0 - wy) * top + wy * bottom
    return output


def recreated_functions(layer_name, settings, values):
    if layer_name.startswith("input_layer"):
        return values

    elif layer_name.startswith("dense"):
        return handleDenseLayer(layer_name, values, settings)

    elif layer_name.startswith("batch_normalization"):
        return handleBatchNormalization(layer_name, values, settings)

    elif layer_name.startswith("leaky_re_lu"):
        return handleLeakyReLu(layer_name, values, settings)

    elif layer_name.startswith("reshape"):
        return handleReshape(layer_name, values, settings)

    elif layer_name.startswith("conv2d") and not layer_name.startswith("conv2d_transpose"):
        return handleConv2D(layer_name, values, settings)

    elif layer_name.startswith("conv2d_transpose"):
        return handleConv2DTranspose(layer_name, values, settings)

    elif layer_name.startswith("resizing"):
        return handleResize(layer_name, values, settings)

    else:
        raise NotImplementedError(f"Unsupported layer recreation for: {layer_name}")


def _to_uint8_channel(v: float) -> int:
    scaled = int(round((float(v) + 1.0) * 127.5))
    if scaled < 0:
        return 0
    if scaled > 255:
        return 255
    return scaled


def run_without_keras():
    layer_definitions, _, _ = load_model_from_artifacts()

    current_values = _deep_copy(CUSTOM_INPUT_VECTOR)
    save_recreated_value(0, "original", current_values)

    for layer_index, layer in enumerate(layer_definitions, start=1):
        current_values = recreated_functions(layer["name"], layer["config"], current_values)
        save_recreated_value(layer_index, layer["name"], current_values)

    shape = _shape_of(current_values)
    if len(shape) == 4 and shape[0] > 0 and shape[-1] == 3:
        image = current_values[0]
        height, width, _ = _shape_of(image)
        pil_image = Image.new("RGB", (width, height))
        pixels = []
        for y in range(height):
            for x in range(width):
                r, g, b = image[y][x]
                pixels.append((_to_uint8_channel(r), _to_uint8_channel(g), _to_uint8_channel(b)))
        pil_image.putdata(pixels)

        recreated_output_path = f"{RECREATED_DIR}/out.jpg"
        pil_image.save(recreated_output_path, format="JPEG", quality=95)
        print(f"Saved recreated image to: {recreated_output_path}")

    print(f"Saved recreated values to: {RECREATED_DIR}")


run_without_keras()
