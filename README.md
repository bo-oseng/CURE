# CURE

Official implementation of **CURE: Controllable Unified Image Restoration for
Complex Degradations** (ICPR 2026).

[[Project page](https://bo-oseng.github.io/CURE/)]
[[Paper](https://arxiv.org/abs/2607.03044)]
[[CCDD-11 dataset](https://huggingface.co/datasets/ses7720/CCDD-11)]
[[Pretrained models](https://huggingface.co/ses7720/CURE)]

![CURE overview](project-page/static/images/method/overview-05.jpg)

CURE extends unified image restoration with explicit control over what to
restore, how strongly to restore it, and the order in which multiple
degradations are removed.

## Installation

Python 3.11 and a CUDA-capable PyTorch installation are recommended.

```bash
conda create -n cure python=3.11 -y
conda activate cure
pip install -r requirements.txt
```

## Dataset

Download [CCDD-11 from Hugging Face](https://huggingface.co/datasets/ses7720/CCDD-11):

```bash
pip install -U huggingface_hub
hf download ses7720/CCDD-11 --repo-type dataset --local-dir data
```

The training utilities expect the extracted data under
`data/half_train/main_data`, `data/half_train/sub_data`, and
`data/half_test/main_data`. Raw data and generated HDF5 files are ignored by
Git.

Build the aligned patch database used for CURE training:

```bash
python tools/prepare_h5.py cure
```

The following additional command is only needed when retraining the
OneRestore baseline from scratch. It creates `datasets_h5/half_og_train.h5`,
which is consumed by `bash/02_train_OneRestore_baseline.sh`:

```bash
python tools/prepare_h5.py baseline
```

## Checkpoints

Download the released checkpoints from the
[CURE model repository on Hugging Face](https://huggingface.co/ses7720/CURE):

```bash
pip install -U huggingface_hub
hf download ses7720/CURE \
  CURE_restorer.tar \
  OneRestore_embedder.tar \
  OneRestore_restorer.tar \
  --local-dir checkpoints
```

| File | Usage |
| --- | --- |
| `CURE_restorer.tar` | CURE inference and evaluation |
| `OneRestore_embedder.tar` | Prompt encoder used for inference and training |
| `OneRestore_restorer.tar` | Pretrained OneRestore baseline and CURE training initialization |

The same files are also available from this
[Google Drive mirror](https://drive.google.com/drive/folders/1xUXAgRPXZjntPR4m4BCh8Tyw2BXGhqSE?usp=sharing).
Hugging Face is the recommended source because it provides versioned model
hosting and more reliable command-line downloads.

## Inference

Place the CURE restorer and OneRestore embedder weights in `checkpoints/` as
`CURE_restorer.tar` and `OneRestore_embedder.tar`, then run either a composite
prompt or an ordered sequence:

```bash
python inference.py \
  --input examples/input.png \
  --output outputs/restored.png \
  --prompt low_haze \
  --strength 1.0

python inference.py \
  --input examples/input.png \
  --output outputs/sequential.png \
  --sequence low haze
```

Use `--checkpoint` and `--embedder-checkpoint` to override the default weight
paths.

## Training

The scripts use `torchrun`; set `GPUS` to the number of local GPU processes.

```bash
GPUS=4 bash bash/01_train_OneRestore_embedder.sh
GPUS=4 bash bash/02_train_OneRestore_baseline.sh
GPUS=4 bash bash/03_train_CURE.sh
```

Run `python <script> --help` for all dataset, checkpoint, optimization, and
resume options.

## Citation

```bibtex
@inproceedings{kim2026cure,
  title     = {CURE: Controllable Unified Image Restoration for Complex Degradations},
  author    = {Kim, Boseong and Cho, Donghyeon},
  booktitle = {International Conference on Pattern Recognition},
  year      = {2026}
}
```
