# ControlNet Multi-Weather Image Restoration

A minimal, **runnable end-to-end training framework** built on **Stable
Diffusion 2 + ControlNet** for multi-weather image restoration
(derain / desnow / dehaze).

> The goal of this repository is *process first*: get the pipeline running
> and observe initial restoration results. It is **not** optimised for
> state-of-the-art PSNR / SSIM.

---

## ✨ Features

* **SD2 base + ControlNet** restoration pipeline via HuggingFace `diffusers`.
* **Weather-class conditioning** controlled by a single YAML switch
  (`weather_prompt.use_weather_prompt`). When enabled, the weather type
  (rain / snow / haze) is prepended to the SD2 text prompt so the model
  knows *which* degradation it is removing.
* **Single unified config file** (`config/config.yaml`) — every path,
  weight, weather class and hyper-parameter lives there. Startup only
  needs `--config config.yaml`.
* Modular layout — `models/`, `datasets/`, `utils/`, plus a lightweight
  custom `Trainer` (no `accelerate` / HuggingFace `Trainer` magic).

---

## 📁 Project Layout

```
ControlNet/
├── config/
│   └── config.yaml           # the only config file you need to edit
├── models/
│   ├── controlnet_wrapper.py # SD2 + ControlNet wrapper
│   └── weather_conditioning.py # weather-class prompt builder
├── datasets/
│   ├── weather_restoration.py # multi-weather (LQ, GT) dataset
│   └── transforms.py          # paired augmentations
├── utils/
│   ├── config.py              # YAML loader + dotted-key overrides
│   └── logger.py              # logging + seeding
├── train.py                   # Trainer class + training loop
├── main.py                    # CLI entry (train / infer)
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

Everything (paths, weights, weather class, hyper-parameters) lives in
`config/config.yaml`. Startup needs only `--config`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit config/config.yaml — set the three locations below:

#    (a) Where is your SD2 base model?
#        model.base_model_path: "stabilityai/stable-diffusion-2-base"

#    (b) Where is your dataset on disk?
#        dataset.data_root: "/path/to/data"

#    (c) (Inference only) which checkpoint / image to restore?
#        infer.ckpt_path:   "./outputs/controlnet_multi_weather/checkpoint-1000"
#        infer.input_path:  "./demo/rainy.png"
#        infer.output_path: "./demo/rainy_restored.png"

# 3. Train
python main.py --config config/config.yaml

# 4. Switch to inference
#    Edit config.yaml:   project.mode: infer
python main.py --config config/config.yaml
```

> No command-line arguments for training/inference are required.  The
> `--mode` and `--override` flags exist only as escape hatches for ad-hoc
> experiments; under normal use you edit the YAML and re-run
> `python main.py --config config/config.yaml`.

---

## 📂 Dataset Directory

The dataset location is configured by **one line** in `config/config.yaml`:

```yaml
dataset:
  data_root: "./data"          # <-- change this to your dataset path
  train_subdir: "train"
  test_subdir:  "test"
  lq_subdir:    "LQ"
  gt_subdir:    "GT"
```

### Windows paths

If you are on Windows, **do NOT use double-quoted backslashes** —
YAML interprets `\P`, `\W` etc. as (unknown) escape sequences and the
parser will fail.  Pick one of these instead:

```yaml
# 1. Single-quoted  (recommended)
dataset:
  data_root: 'D:\Projects\pycharm\WeaFU-main\dataprocess'

# 2. Forward slashes  (also fine, Python handles both)
dataset:
  data_root: D:/Projects/pycharm/WeaFU-main/dataprocess

# 3. Unquoted
dataset:
  data_root: D:\Projects\pycharm\WeaFU-main\dataprocess
```

Avoid:

```yaml
# ❌ This will raise a YAML ScannerError ("found unknown escape character 'p'")
dataset:
  data_root: "D:\Projects\pycharm\WeaFU-main\dataprocess"
```

Expected layout (weather-first, no separate validation set):

```
<data_root>/
├── rain/
│   ├── train/
│   │   ├── LQ/   # low-quality (degraded) images
│   │   └── GT/   # ground-truth (clean) images — filenames must match LQ/
│   └── test/
│       ├── LQ/
│       └── GT/
├── snow/
│   ├── train/{LQ,GT}
│   └── test/{LQ,GT}
└── haze/
    ├── train/{LQ,GT}
    └── test/{LQ,GT}
```

Folder names (`rain`, `snow`, `haze` …) are the **raw weather class labels**
and must match the keys in `weather_prompt.weather_tokens` in the YAML.
The `test` split is reused for periodic validation during training because
the dataset does not ship a dedicated validation set.

---

## 🔑 How `weather_prompt` Is Used

The YAML block you asked about:

```yaml
weather_prompt:
  use_weather_prompt: true
  prompt_template: "a clean photo after removing {weather}, high quality, sharp"
  negative_prompt: "blurry, low quality, artifacts, noise, distorted"
  weather_tokens:
    rain: "rain"
    snow: "snow"
    haze: "haze"
  cfg_dropout_prob: 0.1
  empty_prompt: ""
```

is consumed by `WeatherPromptEncoder` in `models/weather_conditioning.py:33`
and threaded through the training pipeline as follows:

| Step | Where | What happens |
|------|-------|--------------|
| 1 | `datasets/weather_restoration.py:160` | The dataset returns a `weather` string per item (the parent folder name, e.g. `"rain"`). |
| 2 | `train.py:347-352` | The trainer passes `batch["weather"]` (a list of labels for the batch) into `model.prepare_batch(...)`. |
| 3 | `models/controlnet_wrapper.py:213` | `prepare_batch` calls `self.weather_prompt_encoder.build_prompts(weather_labels, generator=...)`. |
| 4 | `models/weather_conditioning.py:73-101` | For each label the encoder looks up `weather_tokens[label]` (e.g. `haze → "haze"`), substitutes it into `prompt_template` (e.g. `"a clean photo after removing haze, ..."`), and optionally drops the prompt to `empty_prompt` with probability `cfg_dropout_prob` for classifier-free guidance. |
| 5 | `models/controlnet_wrapper.py:214` | The resulting prompts are tokenised by the frozen SD2 text encoder to produce `prompt_embeds`. |
| 6 | `models/controlnet_wrapper.py:241-256` | `prompt_embeds` are fed to **both** the ControlNet (`encoder_hidden_states`) and the SD2 UNet. |

So `weather_prompt` is **not** combined with the dataset files themselves
— it is the bridge that turns the dataset's weather-class label into the
text conditioning that drives SD2.

### The master switch

* `use_weather_prompt: true`  → each prompt becomes e.g.
  `"a clean photo after removing rain, high quality, sharp"`, so the model
  distinguishes rain / snow / haze.
* `use_weather_prompt: false` → every prompt becomes the empty string in
  `empty_prompt`; the weather-class signal is removed entirely (useful
  for ablations).

### Inference

The same encoder is reused at inference (`models/controlnet_wrapper.py:309`).
For a single image the label comes from `infer.weather` in the YAML; for
batch inference it comes from `infer.weather_from_subdir` (parent folder
name) or from `infer.weather` as a fallback.

---

## 🧠 Pipeline Overview

1. **Dataset** returns `(lq_image, gt_image, weather_label)` per item.
2. **VAE** encodes the GT image into SD2 latent space (frozen).
3. **Text encoder** embeds the weather-conditioned prompt (frozen).
4. **ControlNet** takes the *LQ* image as its hint and predicts residual
   signals for the SD2 UNet.
5. **SD2 UNet** (frozen) is steered by both the text embeddings and the
   ControlNet residuals to predict the noise added to the GT latent.
6. Standard ε-prediction MSE loss; gradients flow **only into ControlNet**.

---

## 📝 Notes

* Single-GPU training is implemented directly in `train.py`. For
  multi-GPU you can launch with `torchrun --nproc_per_node=N main.py --config config/config.yaml`.
* `outputs/<experiment_name>/` contains checkpoints, `train.log` and a
  `validation/` folder with stacked image grids
  (`LQ / GT / restored`).
* Adjust `dataset.image_size` in `config/config.yaml` if your GPU is memory
  constrained (256 / 384 / 512 are all supported).
* `weather_prompt.cfg_dropout_prob` provides the unconditional branch for
  classifier-free guidance — set to `0.0` to disable.

### What does `steps=N` mean?

The training loop logs `steps=N` at startup — this is the **total number
of optimization steps** (one gradient update each), **NOT** the diffusion
timesteps.  Concretely:

```
total_steps = ceil(N_samples / batch_size / gradient_accumulation_steps)
            × num_train_epochs
```

Each step does **one** forward+backward pass through ControlNet + SD2 UNet
and samples **one** random diffusion timestep (out of the 1000-step DDPM
schedule) for noise prediction.  So if your dataset has 18 252 images and
you train for 20 epochs at `batch_size=4`, expect `20 × 4563 = 91 260`
total steps.

To shrink training time, adjust in the YAML:

| Knob | Effect |
|------|--------|
| `train.num_train_epochs` | Total epoch count (default 20). |
| `train.max_train_steps` | Hard cap on total steps; overrides `num_train_epochs`. |
| `train.batch_size` | Bigger ⇒ fewer steps per epoch. |
| `train.gradient_accumulation_steps` | Effectively larger batch without OOM. |
| `dataset.image_size` | 256 instead of 512 ⇒ ~4× faster per step. |

### Per-step timing and ETA

A single **live `tqdm` progress bar** spans the entire training run and
updates twice a second with the latest metrics:

```
train |████████████▌        | 2000/91260 [08:32<30:21, 3.91it/s, loss=0.0842, lr=1.0e-05]
```

* **Elapsed** / **remaining** time come from tqdm's internal timer.
* **Rate** (`it/s`) is steps-per-second.
* **Postfix** (`loss`, `loss_l1`, `lr`, `epoch`) updates every step.

Every `log_every_n_steps` (default 20) a structured log line is also
written *above* the bar via `tqdm.write()`, so the training record in
`outputs/<name>/train.log` keeps the same format as before:

```
step=200/91260 | epoch=0 (200/4563) | loss=0.0842 | loss_l1=0.0000
            | lr=1.00e-05 | step=0.92s | epoch_eta=1h10m | total_eta=23h22m
```

To disable the bar (e.g. when piping output to a file in CI), set:

```yaml
train:
  show_progress_bar: false
```

### Per-epoch metrics, checkpointing & best model

At the end of every epoch the loop prints an aggregate metric line **and
** automatically triggers three bookkeeping actions depending on the
YAML:

```yaml
train:
  # Save a full checkpoint every N epochs (in addition to checkpointing_steps).
  checkpointing_epochs: 1

  # Continuously overwrite outputs/<name>/best/ with the best-so-far model.
  # best_metric supports loss-style and image-quality metrics; the
  # comparison direction (min/max) is auto-inferred from the metric name.
  save_best: true
  best_metric: "val_psnr"          # epoch_loss | epoch_loss_l1 | val_psnr | val_ssim
  best_metric_mode: "max"          # explicit; auto-inferred if null

  # Run the model on N test images at every save point and write
  # LQ / GT / restored grids to outputs/<name>/preview/.
  num_preview_images: 4

  # PSNR / SSIM computation (see "Image-quality metrics" below).
  image_quality:
    data_range: 1.0
    backend: "torchmetrics"        # or "pure-torch" (built-in fallback)
```

End-of-epoch output (printed above the bar):

```
epoch 0 | avg_loss=0.084231 | avg_loss_l1=0.000123 | val_psnr=22.51dB
       | val_ssim=0.8214 | steps=4563 | time=1h09m53s (0.92s/step)
  >> NEW BEST val_psnr = 22.510 (prev=None) @ epoch=0 step=4563
```

### Output directory layout

Each saved checkpoint is a **directory** (HuggingFace's standard format,
not a single file):

```
outputs/<name>/
├── checkpoint-<step>/                  # step-based (every checkpointing_steps)
│   ├── diffusion_pytorch_model.safetensors   ← the actual ControlNet weights (~1.4 GB)
│   ├── config.json                          <!-- config.json -->
│   ├── optimizer.pt                         ← Adam momentum/variance state
│   └── scheduler.pt                         ← LR scheduler state
├── checkpoint-epoch-<N>/               # epoch-based (every checkpointing_epochs)
│   └── (same 4 files as above)
├── best/                               # continuously overwritten with the best model
│   ├── diffusion_pytorch_model.safetensors
│   ├── config.json
│   ├── optimizer.pt
│   └── scheduler.pt
├── preview/
│   ├── step-004563.png                 # LQ / GT / restored triplet
│   ├── epoch-000.png
│   ├── best.png                        # current-best preview
│   └── final.png
├── validation/
│   └── step-000500.png                 # periodic validation grid
└── train.log                           # all INFO logs
```

For inference, point `infer.ckpt_path` at the **directory**, not at a
single `.safetensors` file:

```yaml
infer:
  ckpt_path: "./outputs/<name>/best"            # canonical "best so far"
  # or:
  # ckpt_path: "./outputs/<name>/checkpoint-700"
  # ckpt_path: "./outputs/<name>/checkpoint-epoch-0"
```

After every save the trainer logs the exact file list it wrote:

```
INFO | train | Saved checkpoint to ./outputs/controlnet_multi_weather/checkpoint-700
       (files: config.json, diffusion_pytorch_model.safetensors, optimizer.pt, scheduler.pt)
```

### Full-image inference (no crop) & resolution restoration

During training, images are center-cropped to `dataset.image_size` (e.g.,
512x512) to match square training patches.

However, during **inference** (one-shot restoration via `main.py --mode infer`),
cropping is highly undesirable because it would throw away the borders of
your test images.  We implement **full-image warp inference**:

1. **Pre-processing**: The entire `orig_w × orig_h` image is warped directly
   to a square tensor of `image_size × image_size` (`crop=False`, no border
   loss).
2. **Inference**: ControlNet + SD UNet restore the warped square image.
3. **Post-processing**: The restored output is scaled back to the exact
   original resolution (`orig_w × orig_h`) using bicubic interpolation.

This ensures **the entire test image is restored**, and the output matches
your input resolution and aspect ratio pixel-for-pixel without distortion.

To toggle this behavior (default is true):

```yaml
infer:
  output_to_original_size: true         # set to false to keep outputs as 512x512
```

### Image-quality metrics (PSNR / SSIM)

Every epoch end (and every `validation_steps` interval) the loop runs the
full test set through the model and computes **PSNR** and **SSIM** between
the restored outputs and the ground-truth.  Both metrics are computed
on the entire test set (not just the first N preview images) and
written to:

* the terminal log,
* the per-epoch metric line above,
* the `outputs/<name>/train.log` file,
* TensorBoard / wandb under the `val/psnr` and `val/ssim` tags.

Implementation lives in `utils/metrics.py`:

* **Primary backend**: [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/)
  (the canonical PyTorch-native metric library, recommended).
* **Fallback**: a pure-torch implementation that needs no extra deps —
  used automatically when `torchmetrics` is not installed.

To pick which backend to use (or fall back to pure-torch manually):

```yaml
train:
  image_quality:
    backend: "torchmetrics"   # or "pure-torch"
```

Use PSNR / SSIM as the best-model metric:

```yaml
train:
  best_metric: "val_psnr"     # track peak signal-to-noise ratio
  best_metric_mode: "max"     # higher is better (auto-inferred if omitted)
```

`val_psnr` / `val_ssim` keep overwriting `outputs/<name>/best/` whenever
the metric improves, so the `best/` snapshot is always the checkpoint
that gave the cleanest restoration.

Output layout under `outputs/<name>/`:

```
checkpoint-<step>/      # every checkpointing_steps
checkpoint-epoch-<N>/   # every checkpointing_epochs
best/                   # best model so far (overwritten on improvement)
preview/step-004563.png # LQ / GT / restored triplet for that step
preview/epoch-000.png   # LQ / GT / restored triplet for that epoch
preview/best.png        # LQ / GT / restored triplet for the current best
validation/step-*.png   # periodic validation grids (legacy, kept)
train.log               # all INFO logs
```

The preview grids are the easiest way to eyeball restoration quality:
each PNG stacks the same `num_preview_images` test samples vertically as
`LQ / GT / restored`. Compare `preview/best.png` against the most recent
`preview/step-XXXXXX.png` to see whether the latest checkpoint actually
beats the running best.

Happy restoring! 🌤️🌧️❄️🌫️