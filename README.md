The focus is live inspection: train or load a model, open the UI, adjust inputs and epochs, and immediately see what the network produces internally and externally.

## Main features

- **Live generator visualization**: randomize or control the latent input and see generated image outputs update.
- **Live discriminator visualization**: feed generated images through the discriminator and inspect its response.
- **Layer-by-layer inspection**: choose network layers and visualize intermediate activations rather than only final outputs.
- **Epoch navigation**: move between saved generator and discriminator checkpoints to watch the model evolve during training.
- **Training statistics**: save losses and discriminator scores during training for later plotting and comparison.
- **Post-training asset pipeline**: export TensorFlow Lite models, produce sample and reproduction images, and build training-comparison plots.
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
- `model_creator/produce_assets.py` runs the post-training asset pipeline: TensorFlow Lite exports, generated sample sets, reproduction-search images, and comparison plots.
- `model_creator/run_ui_server.py` starts the Python visualization backend.
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
