
## Prepare diffusers model

### Setup Environment
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"

pip install --upgrade transformers accelerate xformer
pip install safetensors
```

### (Optional) Convert ckpt/safetensor to diffusers style
```
python diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path chilloutmix/Chilloutmix-ema-fp16.safetensors --from_safetensors --to_safetensors --scheduler_type euler-ancestral --image_size 512 --prediction_type epsilon --device cpu --from_safetensors --to_safetensors --dump_path ./diffusers-model
```

### (Optional) Test diffusers model
```
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("../diffusers-model", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "best quality, ultra high res, (photorealistic:1.4), 1girl, thighhighs, (big chest), (upper body), (Kpop idol), (aegyo sal:1), (platinum blonde hair:1), ((puffy eyes)), looking at viewer, facing front, smiling, ((naked))"
negative_prompt = "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glan, ((cloth))"
image = pipe(prompt).images[0]
image.save('test.png')
```