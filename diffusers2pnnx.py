import os
import torch
from diffusers import StableDiffusionPipeline

# config
device = "cuda"
from_model = "../diffusers-model"
to_model = "model"
height, width = 512, 512



# check
assert height % 8 == 0 and width % 8 == 0
height, width = height // 8, width // 8
os.makedirs(to_model, exist_ok=True)

# load model
pipe = StableDiffusionPipeline.from_pretrained(from_model, torch_dtype=torch.float32)
pipe = pipe.to(device)

# jit unet
unet = torch.jit.trace(pipe.unet, (torch.rand(1,4,height,width).to(device),torch.rand(1).to(device),torch.rand(1,77,768).to(device)))
unet.save(os.path.join(to_model,"unet.pt"))

# # test
# image = pipe(
#     prompt="best quality, ultra high res, (photorealistic:1.4), 1girl, thighhighs, (big chest), (upper body), (Kpop idol), (aegyo sal:1), (platinum blonde hair:1), ((puffy eyes)), looking at viewer, facing front, smiling, ((naked))",
#     negative_prompt="paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glan, ((cloth))",
#     num_inference_steps=50, guidance_scale=7.5, generator=torch.Generator(device=device).manual_seed(1234),
#     height=512, width=512,
# ).images[0]
# image.save('test.png')