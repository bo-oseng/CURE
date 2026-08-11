# CURE

Official implementation of **CURE: Controllable Unified Image Restoration for
Complex Degradations** (ICPR 2026).

[[Project page](https://bo-oseng.github.io/CURE/)]
[[Paper](https://arxiv.org/abs/2607.03044)]
[[CCDD-11 dataset](https://huggingface.co/datasets/ses7720/CCDD-11)]

![CURE overview](project-page/static/images/method/overview-05.jpg)

CURE extends unified image restoration with explicit control over what to
restore, how strongly to restore it, and the order in which multiple
degradations are removed.

## Installation

Python 3.10+ and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
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

Build the aligned patch databases used for training:

```bash
python tools/prepare_h5.py baseline
python tools/prepare_h5.py cure
```

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

## Project page

The static site lives in `project-page/` and is deployed to GitHub Pages by
`.github/workflows/pages.yml` whenever `main` changes. To preview it locally:

```bash
python -m http.server 8000 --directory project-page
```

Then open <http://localhost:8000>.

## Citation

```bibtex
@inproceedings{kim2026cure,
  title     = {CURE: Controllable Unified Image Restoration for Complex Degradations},
  author    = {Kim, Boseong and Cho, Donghyeon},
  booktitle = {International Conference on Pattern Recognition},
  year      = {2026}
}
```
