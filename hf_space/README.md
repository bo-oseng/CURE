---
title: CURE
emoji: 🎛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
python_version: "3.11"
suggested_hardware: t4-small
models:
  - ses7720/CURE
tags:
  - image-restoration
  - controllable-image-restoration
  - low-light-enhancement
  - dehazing
  - deraining
  - desnowing
preload_from_hub:
  - ses7720/CURE CURE_restorer.tar,OneRestore_embedder.tar
---

# CURE: Controllable Unified Image Restoration

This Gradio Space exposes the five CURE inference modes:

- full one-step restoration;
- ratio-controlled restoration;
- selective degradation removal;
- identity/no-restoration inference; and
- ordered two-stage restoration.

The model weights are loaded from [ses7720/CURE](https://huggingface.co/ses7720/CURE).
The source code and command-line inference tools are available on
[GitHub](https://github.com/bo-oseng/CURE).
