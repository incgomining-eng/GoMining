---
title: MOSS-Image NSFW LoRA Demo
emoji: 🌒
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
python_version: "3.10"
pinned: false
license: apache-2.0
short_description: 18+ demo of the Moss-image NSFW LoRA on MOSS-Image-SFT
models:
  - axiomofmind/Moss-image-nsfw-uncensored-lora
  - OpenMOSS-Team/MOSS-Image-SFT
tags:
  - not-for-all-audiences
  - nsfw
  - text-to-image
  - lora
---

# MOSS-Image NSFW LoRA demo (18+)

Gradio demo for [axiomofmind/Moss-image-nsfw-uncensored-lora](https://huggingface.co/axiomofmind/Moss-image-nsfw-uncensored-lora)
on top of [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT).

- Runs on **ZeroGPU** (set the Space hardware to ZeroGPU).
- Defaults follow the model card: trigger `moss_xvstyle`, LoRA scale 0.8, 30 steps, CFG 3.0, 1024×1024, `uncond_skip_layers=[22]`.
- Switch between the default 1,750-step checkpoint and the original 2,000-step checkpoint.

**Adults only.** Prompts referring to minors, age-ambiguous subjects, non-consent, or real/identifiable people are refused.
