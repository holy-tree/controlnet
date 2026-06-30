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
  (rain / snow / fog) is prepended to the SD2 text prompt so the model
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
    haze: "fog"                       # "haze" folder is described as "fog"
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
| 4 | `models/weather_conditioning.py:73-101` | For each label the encoder looks up `weather_tokens[label]` (e.g. `haze → "fog"`), substitutes it into `prompt_template` (e.g. `"a clean photo after removing fog, ..."`), and optionally drops the prompt to `empty_prompt` with probability `cfg_dropout_prob` for classifier-free guidance. |
| 5 | `models/controlnet_wrapper.py:214` | The resulting prompts are tokenised by the frozen SD2 text encoder to produce `prompt_embeds`. |
| 6 | `models/controlnet_wrapper.py:241-256` | `prompt_embeds` are fed to **both** the ControlNet (`encoder_hidden_states`) and the SD2 UNet. |

So `weather_prompt` is **not** combined with the dataset files themselves
— it is the bridge that turns the dataset's weather-class label into the
text conditioning that drives SD2.

### The master switch

* `use_weather_prompt: true`  → each prompt becomes e.g.
  `"a clean photo after removing rain, high quality, sharp"`, so the model
  distinguishes rain / snow / fog.
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

Happy restoring! 🌤️🌧️❄️🌫️