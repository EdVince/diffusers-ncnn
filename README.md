# diffusers-ncnn
Port [diffusers](https://github.com/huggingface/diffusers) model to [ncnn](https://github.com/Tencent/ncnn)

diffusers is the well-known repo for stable diffusion pipeline, there are countless SD model in huggingface with diffusers format.

With diffusers-ncnn, you can port whatever SD model to ncnn. It doesn't rely on python or pytorch and is a lightweight inference engine. Further more, you can also put the SD model on your phone.

Note: Besides diffusers, there is a famous repo: [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), but its code is awful and I used to export novelai models based on it. If you are interested in it, please refer to: [Stable-Diffusion-NCNN](https://github.com/EdVince/Stable-Diffusion-NCNN), the c++ code of this repo is mostly copied from that.

## Prepare diffusers model

### Setup Environment
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"

pip install --upgrade transformers accelerate xformer
pip install safetensors
```

### (Optional) Convert ckpt/safetensor to diffusers style
```bash
python diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path chilloutmix/Chilloutmix-ema-fp16.safetensors --scheduler_type euler-ancestral --image_size 512 --prediction_type epsilon --device cpu --from_safetensors --to_safetensors --dump_path ./diffusers-model
```

### (Optional) Test diffusers model
```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("../diffusers-model", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    prompt="best quality, ultra high res, (photorealistic:1.4), 1girl, thighhighs, (big chest), (upper body), (Kpop idol), (aegyo sal:1), (platinum blonde hair:1), ((puffy eyes)), looking at viewer, facing front, smiling, ((naked))",
    negative_prompt="paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glan, ((cloth))",
    num_inference_steps=50, guidance_scale=7.5, generator=torch.Generator(device="cuda").manual_seed(1234),
    height=512, width=512,
).images[0]

image.save('test.png')
```

## Export to ncnn

### Modify some code in python

1. UNet
```python
file: ./diffusers/src/diffusers/models/unet_2d_condition.py
position: the output of the "forward" func in class "UNet2DConditionModel"

# if not return_dict:
#     return (sample,)

# return UNet2DConditionOutput(sample=sample)

return sample
```

### Export to pnnx
```
python diffusers2pnnx.py
```

### Export to ncnn
```bash
pnnx unet.pt inputshape=[1,4,64,64],[1],[1,77,768]
```

### Modify the ncnn.param
1. unet.ncnn.param
```bash
change: "Expand                   Tensor.expand_51         xxxxxx"
    to: "Noop                     Tensor.expand_51         xxxxxx"
```

### Test in python
```bash
python test_ncnn.py
```

## Run with x86
1. prepare model
```
assets
    vocab.txt
    log_sigmas.bin
    AutoencoderKL-base-fp16.param
    FrozenCLIPEmbedder-fp16.param
    AutoencoderKL-fp16.bin (please download from https://github.com/EdVince/Stable-Diffusion-NCNN)
    FrozenCLIPEmbedder-fp16.bin (please download from https://github.com/EdVince/Stable-Diffusion-NCNN)
    unet.ncnn.bin (it should be generated by yourself)
    unet.ncnn.param (it should be generated by yourself)
```
2. compile it with visual studio