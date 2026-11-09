"""Gradio demo for axiomofmind/Moss-image-nsfw-uncensored-lora (ZeroGPU)."""
import spaces  # must be imported before torch on ZeroGPU

import json
import random
import re

import gradio as gr
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from peft.tuners.lora.layer import LoraLayer
from safetensors.torch import load_file

LORA_REPO = "axiomofmind/Moss-image-nsfw-uncensored-lora"
BASE_MODEL = "OpenMOSS-Team/MOSS-Image-SFT"
TRIGGER = "moss_xvstyle"
MAX_SEED = 2**31 - 1

# Both checkpoints from the LoRA repo, with their recommended scales.
CHECKPOINTS = {
    "Focused 1750-step (default)": ("", 0.8),
    "Original 2000-step": ("original-checkpoint-2000/", 0.7),
}

# Refuse prompts that point at minors, non-consent, or real people.
BLOCKED = re.compile(
    r"\b(child|children|kid|kids|minor|minors|underage|under[- ]?age|teen|teens|teenage|teenager|"
    r"young(er)?[- ]?(girl|boy|looking)|little (girl|boy)|loli|lolita|shota|schoolgirl|schoolboy|"
    r"school uniform|preteen|pre-teen|toddler|infant|baby|juvenile|adolescent|"
    r"rape|raped|non[- ]?consensual|forced|unconscious|asleep|drugged|"
    r"celebrity|deepfake|look[- ]?alike)\b",
    re.IGNORECASE,
)


def check_prompt(text: str) -> None:
    m = BLOCKED.search(text or "")
    if m:
        raise gr.Error(
            f"Prompt refused (matched “{m.group(0)}”). This demo only generates "
            "clearly adult, consenting, fictional subjects."
        )
    # Ages under 18 written as numbers, e.g. "age 16".
    ages = re.findall(r"\bage[d]?\s*(\d{1,2})\b", text or "", re.IGNORECASE)
    ages += re.findall(r"\b(\d{1,2})[- ]?(?:yo|y/o|years?[- ]old)\b", text or "", re.IGNORECASE)
    for age in ages:
        if int(age) < 18:
            raise gr.Error("Prompt refused: subjects must be adults (18+).")


print("Loading MOSS-Image-SFT…")
pipe = DiffusionPipeline.from_pretrained(
    BASE_MODEL,
    custom_pipeline="pipeline_moss_image",
    trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)

# Inject an empty LoRA adapter once, then swap weights per checkpoint.
cfg = json.load(open(hf_hub_download(LORA_REPO, "adapter_config.json"), encoding="utf-8"))
inject_adapter_in_model(
    LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], target_modules=cfg["target_modules"]),
    pipe.transformer,
)

LORA_STATES = {}
for name, (prefix, _) in CHECKPOINTS.items():
    path = hf_hub_download(LORA_REPO, f"{prefix}pytorch_lora_weights.safetensors")
    LORA_STATES[name] = {
        k.removeprefix("transformer."): v.to(torch.bfloat16)
        for k, v in load_file(path, device="cpu").items()
    }

current = {"ckpt": None}


def load_checkpoint(name: str) -> None:
    if current["ckpt"] == name:
        return
    _, unexpected = set_peft_model_state_dict(pipe.transformer, LORA_STATES[name])
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA tensors: {unexpected[:5]}")
    current["ckpt"] = name


load_checkpoint(next(iter(CHECKPOINTS)))
pipe.transformer.to(torch.bfloat16)
pipe.to("cuda")
print("Ready.")


def set_lora_scale(scale: float) -> None:
    for layer in pipe.transformer.modules():
        if isinstance(layer, LoraLayer):
            layer.set_scale("default", float(scale))


@spaces.GPU(duration=120)
def generate(
    prompt: str,
    checkpoint: str = "Focused 1750-step (default)",
    lora_scale: float = 0.8,
    steps: int = 30,
    guidance: float = 3.0,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    randomize_seed: bool = False,
    negative_prompt: str = "",
    progress=gr.Progress(track_tqdm=True),
):
    """Generate an adult photographic image with the Moss NSFW LoRA.

    Args:
        prompt: Description of the image. The trigger `moss_xvstyle` is added if missing.
        checkpoint: Which LoRA checkpoint to use.
        lora_scale: LoRA strength (0 disables it). 0.7-0.9 recommended.
        steps: Number of inference steps.
        guidance: Classifier-free guidance scale.
        width: Image width in pixels (multiple of 16).
        height: Image height in pixels (multiple of 16).
        seed: Random seed.
        randomize_seed: Pick a random seed instead.
        negative_prompt: Optional negative prompt.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    check_prompt(prompt)
    if not prompt.lower().startswith(TRIGGER):
        prompt = f"{TRIGGER}, {prompt}"

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    load_checkpoint(checkpoint)
    set_lora_scale(lora_scale)

    image = pipe(
        prompt,
        negative_prompt=negative_prompt or "",
        width=int(width) // 16 * 16,
        height=int(height) // 16 * 16,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        uncond_skip_layers=[22],
        generator=torch.Generator("cpu").manual_seed(int(seed)),
    ).images[0]
    return image, seed


def on_checkpoint_change(name: str):
    return gr.update(value=CHECKPOINTS[name][1])


EXAMPLES = [
    "moss_xvstyle, a realistic photograph of an adult woman in her thirties with long dark wavy hair, "
    "reclining nude on a deep green velvet sofa, soft morning window light, detailed skin texture, quiet living room.",
    "moss_xvstyle, a tasteful boudoir photograph of an adult woman with short curly black hair, wearing an "
    "open cream satin robe, seated on a white marble bathroom vanity, warm vanity bulbs, calm confident expression.",
    "moss_xvstyle, a detailed adult photograph of a freckled woman with long copper-red hair kneeling on rumpled "
    "white bedsheets, sheer pale-blue lingerie, diffused midday sunlight through gauzy curtains, airy bedroom.",
]

CSS = "#col-container { max-width: 1100px; margin: 0 auto; }"

with gr.Blocks(title="MOSS-Image NSFW LoRA") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            f"""
# MOSS-Image NSFW LoRA (18+)
Demo of [{LORA_REPO}](https://huggingface.co/{LORA_REPO}) on [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}).

**Adults only. Explicit content.** By using this Space you confirm you are 18+ and that it is legal for you to view
this content. Prompts involving minors, age-ambiguous subjects, non-consent, or real people are refused.
The trigger `{TRIGGER}` is added automatically.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=5, placeholder=f"{TRIGGER}, a photograph of an adult woman…")
                run = gr.Button("Generate", variant="primary")
                with gr.Accordion("Settings", open=False):
                    checkpoint = gr.Radio(list(CHECKPOINTS), value=next(iter(CHECKPOINTS)), label="Checkpoint")
                    lora_scale = gr.Slider(0.0, 1.2, value=0.8, step=0.05, label="LoRA scale")
                    steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 7.0, value=3.0, step=0.1, label="Guidance")
                    with gr.Row():
                        width = gr.Slider(512, 1536, value=1024, step=16, label="Width")
                        height = gr.Slider(512, 1536, value=1024, step=16, label="Height")
                    with gr.Row():
                        seed = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed")
                        randomize_seed = gr.Checkbox(value=True, label="Randomize seed")
                    negative_prompt = gr.Textbox(label="Negative prompt", value="")
            with gr.Column(scale=1):
                result = gr.Image(label="Result", format="png")

        gr.Examples(EXAMPLES, inputs=[prompt], label="Example prompts")

    checkpoint.change(on_checkpoint_change, checkpoint, lora_scale)
    gr.on(
        [run.click, prompt.submit],
        generate,
        inputs=[prompt, checkpoint, lora_scale, steps, guidance, width, height, seed, randomize_seed, negative_prompt],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch(css=CSS, mcp_server=True)
