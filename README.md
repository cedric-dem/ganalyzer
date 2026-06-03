The focus is live inspection: train or load a model, open the UI, adjust inputs and epochs, and immediately see what the network produces internally and externally.

## Main features

- **Live generator visualization**: randomize or control the latent input and see generated image outputs update.
- **Live discriminator visualization**: feed generated images through the discriminator and inspect its response.
- **Layer-by-layer inspection**: choose network layers and visualize intermediate activations rather than only final outputs.
- **Epoch navigation**: move between saved generator and discriminator checkpoints to watch the model evolve during training.
- **Training statistics**: save losses and discriminator scores during training for later plotting and comparison.
- **Offline sample generators**: produce checkpoint-by-checkpoint evolution samples and continuous latent-space movement sequences as PNG files.
- **Multiple interfaces**: a Vue web UI is the default interface, with a legacy Tkinter UI still available through configuration.
- **Android model path**: trained models can be converted toward TensorFlow Lite usage for the included Android app experiment.

## Repository layout

```text
.
├── model_creator/          # GAN model definitions, training scripts, Flask visualization API, plotting utilities
├── web-ui/                 # Vue/Vite frontend for live model visualization
├── android_app/            # Android experiment for applying exported models
└── README.md
```

Important entry points:

- `model_creator/train_model.py` trains the GAN, saves generator/discriminator checkpoints, and writes statistics.
- `model_creator/produce_sample_outputs.py` renders a standalone batch of generated images from the latest complete checkpoint.
- `model_creator/produce_evolution_sample.py` renders one generated image per saved generator checkpoint from the same latent vector so you can compare how training changes a fixed sample over time.
- `model_creator/produce_continuous_movement.py` renders a sequence from the latest generator checkpoint while gradually changing a few latent-vector values, creating a smooth latent-space movement sample.
- `model_creator/run_UI_server.py` starts the Python visualization backend.
- `model_creator/ganalyzer/GUIWebPage.py` exposes the Flask API used by the web UI.
- `web-ui/src/App.vue` mounts the generator input panel plus generator and discriminator visualization panels.
- `web-ui/src/js/webUI.ts` coordinates frontend state, API calls, selected epochs, and layer visualization updates.

## How the live visualization works

1. The training script saves generator and discriminator models at regular epoch intervals.
2. The Flask backend loads the available checkpoints for the configured model name and latent-space size.
3. The Vue frontend synchronizes with the backend and receives the available layer lists and checkpoint count.
4. When you move sliders or choose a layer, the frontend sends the current input, target model, layer name, and epoch selection to the backend.
5. The backend runs the selected model up to that layer and returns the activation values.
6. The frontend renders those values as images or grids so you can watch the network change live.

## Quick start

### 1. Configure the model and dataset

Edit `model_creator/config.py` to choose the dataset, latent-space size, output size, checkpoint interval, and UI mode.

The current defaults expect image data under a path similar to:

```text
model_creator/datasets/<dataset_name>/<image_size>/
```

The project currently targets RGB image generation, with model configuration centralized in `model_creator/ganalyzer/model_config.py` and `model_creator/config.py`.

### 2. Train or resume a model

From `model_creator/`, run:

```bash
python train_model.py
```

Training resumes from the latest saved epoch when checkpoints already exist. During training, GANalyzer saves:

- generator checkpoints,
- discriminator checkpoints,
- CSV statistics for loss and discriminator output values.


### 3. Generate offline sample image sets

After training has produced generator checkpoints, you can create optional sample sets from `model_creator/`:

```bash
python produce_sample_outputs.py
python produce_evolution_sample.py
python produce_continuous_movement.py
```

`produce_sample_outputs.py` loads every available generator checkpoint for the configured model and writes generated images named like `sample_00.png` under:

```text
model_creator/results/<dataset_name>/<model_name>-ls_<latent_size>/sample_outputs/sample_output_epoch_<epoch>/
```

Use this output to inspect fixed batches of standalone samples from every saved generator checkpoint without coupling sample generation to the training loop.

`produce_evolution_sample.py` loads every saved generator checkpoint for the configured model, applies the same random latent vector to each checkpoint, and writes images named like `evolution_sample_<epoch>.png` under:

```text
model_creator/results/<dataset_name>/<model_name>-ls_<latent_size>/evolution_sample/
```

Use this output to compare how one fixed latent input evolves as training progresses.

`produce_continuous_movement.py` loads the latest saved generator checkpoint, starts from a random latent vector, changes a small number of latent dimensions on each step, and writes a numbered image sequence named like `continuous_movement_0001.png` under:

```text
model_creator/results/<dataset_name>/<model_name>-ls_<latent_size>/continuous_movement/
```

Use this output to inspect local movement through latent space with the final/current generator. The sequence length and number of changed latent dimensions are configured by `CONTINUOUS_MOVEMENT_LENGTH` and `CONTINUOUS_MOVEMENT_NUMBER_CHANGES` in `model_creator/config.py`.

### 4. Start the visualization backend

From `model_creator/`, run:

```bash
python run_ui_server.py
```

By default, `GUI_tkinter = False` in `model_creator/config.py`, so this starts the Flask backend used by the web UI.