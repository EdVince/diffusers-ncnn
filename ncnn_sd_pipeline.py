import ncnn
import torch
import numpy as np
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import DiffusionPipeline

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: EulerAncestralDiscreteScheduler,
        safety_checker: None,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _encode_prompt(self,prompt,negative_prompt,device):
        # positive prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device))
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # negative prompt
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))
        negative_prompt_embeds = negative_prompt_embeds[0]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        return negative_prompt_embeds, prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_latents(self, height, width, dtype, device):
        shape = (1, 4, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = torch.randn(shape, generator=None, device=device, dtype=dtype, layout=torch.strided).to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def load_ncnn(self, unet_path):
        self.unet_ncnn = ncnn.Net()
        self.unet_ncnn.load_param(unet_path[0])
        self.unet_ncnn.load_model(unet_path[1])

    def clear_ncnn(self):
        self.unet_ncnn.clear()

    @torch.no_grad()
    def __call__(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        device = self.device

        # 1. Encode input prompt
        negative_prompt_embeds, prompt_embeds = self._encode_prompt(prompt,negative_prompt,device)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        latents = self.prepare_latents(height,width,prompt_embeds.dtype,device)

        # 4. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                # noise_pred_uncond = self.unet(latent_model_input,t,negative_prompt_embeds)
                with self.unet_ncnn.create_extractor() as ex:
                    latent_model_input_numpy = latent_model_input.squeeze(0).cpu().numpy().astype(np.float32)
                    t_numpy = t.view(1).cpu().numpy().astype(np.float32)
                    negative_prompt_embeds_numpy = negative_prompt_embeds.squeeze(0).cpu().numpy().astype(np.float32)
                    ex.input("in0", ncnn.Mat(latent_model_input_numpy).clone())
                    ex.input("in1", ncnn.Mat(t_numpy).clone())
                    ex.input("in2", ncnn.Mat(negative_prompt_embeds_numpy).clone())
                    _, out0 = ex.extract("out0")
                    noise_pred_uncond = torch.from_numpy(np.array(out0)).unsqueeze(0).cuda()
                # noise_pred_text = self.unet(latent_model_input,t,prompt_embeds)
                with self.unet_ncnn.create_extractor() as ex:
                    latent_model_input_numpy = latent_model_input.squeeze(0).cpu().numpy().astype(np.float32)
                    t_numpy = t.view(1).cpu().numpy().astype(np.float32)
                    prompt_embeds_numpy = prompt_embeds.squeeze(0).cpu().numpy().astype(np.float32)
                    ex.input("in0", ncnn.Mat(latent_model_input_numpy).clone())
                    ex.input("in1", ncnn.Mat(t_numpy).clone())
                    ex.input("in2", ncnn.Mat(prompt_embeds_numpy).clone())
                    _, out0 = ex.extract("out0")
                    noise_pred_text = torch.from_numpy(np.array(out0)).unsqueeze(0).cuda()

                # perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 5. Decode the latent
        image = self.decode_latents(latents)

        # 6. Convert to Image
        image = self.numpy_to_pil(image)

        return image
