import os
import torch
from ncnn_sd_pipeline import StableDiffusionPipeline

# config
device = "cuda"
from_model = "../diffusers-model"

# load model
pipe = StableDiffusionPipeline.from_pretrained(from_model, torch_dtype=torch.float32)
pipe = pipe.to(device)

pipe.load_ncnn(["model/unet.ncnn.param","model/unet.ncnn.bin"])

image = pipe(
    prompt="best quality, ultra high res, (photorealistic:1.4), 1girl, thighhighs, (big chest), (upper body), (Kpop idol), (aegyo sal:1), (platinum blonde hair:1), ((puffy eyes)), looking at viewer, facing front, smiling, ((naked))",
    negative_prompt="paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glan, ((cloth))",
    num_inference_steps=50, guidance_scale=7.5,
    height=512, width=512,
)[0]

image.save('test.png')

pipe.clear_ncnn()