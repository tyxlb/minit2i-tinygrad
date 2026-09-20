import json

from huggingface_hub import hf_hub_download
from tinygrad.nn.state import load_state_dict, safe_load

from pipeline import MiniT2IMMJiTModel, MiniT2ITextToImagePipeline, MMJiTConfig

model_type = "minit2i-b-16"

cfg = hf_hub_download("MiniT2I/MiniT2I", f"{model_type}/transformer/config.json")
with open(cfg) as f:
    cfg = json.load(f)
cfg.pop("_class_name")
cfg.pop("_diffusers_version")
cfg = MMJiTConfig(**cfg)

transformer = MiniT2IMMJiTModel(cfg)
w = hf_hub_download(
    "MiniT2I/MiniT2I", f"{model_type}/transformer/diffusion_pytorch_model.safetensors"
)
state_dict = safe_load(w)
load_state_dict(transformer, state_dict, False)
del state_dict

pipe = MiniT2ITextToImagePipeline(transformer)
image = pipe(
    "A lonely astronaut standing on a quiet beach under two moons.",
    guidance_scale=2.5,
    num_inference_steps=100,
)[0]
image.save("tiny.png")
