from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
    StableVideoDiffusionPipeline,
    retrieve_timesteps,
)
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _resize_with_antialiasing_safe(input, size, interpolation="bicubic", align_corners=True):
    """Wrapper for resize that uses the standard function."""
    # Since we're not using MPS anymore, we can use the original function
    return _resize_with_antialiasing(input, size)


class DepthCrafterPipeline(StableVideoDiffusionPipeline):
    
    @property
    def _execution_device(self):
        """
        Returns the device on which the pipeline should be executed.
        Note: MPS is not used due to lack of Conv3D support.
        """
        # If device attribute exists and is set
        if hasattr(self, 'device') and self.device is not None:
            if self.device != torch.device("meta"):
                return self.device
        
        # Check if model has hooks (for CPU offloading)
        if hasattr(self.unet, "_hf_hook"):
            for module in self.unet.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        
        # Try to get device from model parameters
        try:
            return next(self.unet.parameters()).device
        except:
            pass
            
        # Default fallback based on availability
        # MPS doesn't support Conv3D, so we use CPU for Apple Silicon
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ) -> torch.Tensor:
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: image_embeddings in shape of [b, 1024]
        """

        video_224 = _resize_with_antialiasing_safe(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            tmp = self.feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings.append(self.image_encoder(tmp).image_embeds)  # [b, 1024]

        embeddings = torch.cat(embeddings, dim=0)  # [t, 1024]
        return embeddings

    @torch.inference_mode()
    def encode_vae_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ):
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: vae latents in shape of [b, c, h, w]
        """
        video_latents = []
        for i in range(0, video.shape[0], chunk_size):
            video_latents.append(
                self.vae.encode(video[i : i + chunk_size]).latent_dist.mode()
            )
        video_latents = torch.cat(video_latents, dim=0)
        return video_latents

    @staticmethod
    def check_inputs(video, height, width):
        """
        :param video:
        :param height:
        :param width:
        :return:
        """
        if not isinstance(video, torch.Tensor) and not isinstance(video, np.ndarray):
            raise ValueError(
                f"Expected `video` to be a `torch.Tensor` or `VideoReader`, but got a {type(video)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

    @torch.no_grad()
    def __call__(
        self,
        video: Union[np.ndarray, torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        window_size: Optional[int] = 110,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap: int = 25,
        track_time: bool = False,
    ):
        """
        :param video: in shape [t, h, w, c] if np.ndarray or [t, c, h, w] if torch.Tensor, in range [0, 1]
        :param height:
        :param width:
        :param num_inference_steps:
        :param guidance_scale:
        :param window_size: sliding window processing size
        :param fps:
        :param motion_bucket_id:
        :param noise_aug_strength:
        :param decode_chunk_size:
        :param generator:
        :param latents:
        :param output_type:
        :param callback_on_step_end:
        :param callback_on_step_end_tensor_inputs:
        :param return_dict:
        :return:
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(video, height, width)

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale

        # 3. Encode input video
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(video, torch.Tensor)
        video = video.to(device=device, dtype=self.dtype)
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]

        if track_time:
            import time
            start_time = time.time()
            encode_time = None
            denoise_time = None

        video_embeddings = self.encode_video(
            video, chunk_size=decode_chunk_size
        ).unsqueeze(
            0
        )  # [1, t, 1024]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 4. Encode input image using VAE
        noise = randn_tensor(
            video.shape, generator=generator, device=device, dtype=video.dtype
        )
        video = video + noise_aug_strength * noise  # in [t, c, h, w]

        # pdb.set_trace()
        needs_upcasting = (
            self.vae.dtype == torch.float32 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(
            0
        )  # [1, t, c, h, w]

        if track_time:
            encode_time = time.time()
            elapsed_time_ms = (encode_time - start_time) * 1000
            print(f"Elapsed time for encoding video: {elapsed_time_ms:.2f} ms")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # cast back to fp32 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            7,
            127,
            noise_aug_strength,
            video_embeddings.dtype,
            batch_size,
            1,
            False,
        )  # [1 or 2, 3]
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            batch_size,
            window_size,
            num_channels_latents,
            height,
            width,
            video_embeddings.dtype,
            device,
            generator,
            latents,
        )  # [1, t, c, h, w]
        latents_all = None

        idx_start = 0
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)
        else:
            weights = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # inference strategy for long videos
        # two main strategies: 1. noise init from previous frame, 2. segments stitching
        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            # 9. Denoising loop
            latents = latents_init[:, : idx_end - idx_start].clone()
            latents_init = torch.cat(
                [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
            )

            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if latents_all is not None and i == 0:
                        latents[:, :overlap] = (
                            latents_all[:, -overlap:]
                            + latents[:, :overlap]
                            / self.scheduler.init_noise_sigma
                            * self.scheduler.sigmas[i]
                        )

                    latent_model_input = latents  # [1, t, c, h, w]
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )  # [1, t, c, h, w]
                    latent_model_input = torch.cat(
                        [latent_model_input, video_latents_current], dim=2
                    )
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=video_embeddings_current,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        latent_model_input = latents
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )
                        latent_model_input = torch.cat(
                            [latent_model_input, torch.zeros_like(latent_model_input)],
                            dim=2,
                        )
                        noise_pred_uncond = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=torch.zeros_like(
                                video_embeddings_current
                            ),
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]

                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred - noise_pred_uncond
                        )
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

            if latents_all is None:
                latents_all = latents.clone()
            else:
                assert weights is not None
                # latents_all[:, -overlap:] = (
                #     latents[:, :overlap] + latents_all[:, -overlap:]
                # ) / 2.0
                latents_all[:, -overlap:] = latents[
                    :, :overlap
                ] * weights + latents_all[:, -overlap:] * (1 - weights)
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride

        if track_time:
            denoise_time = time.time()
            elapsed_time_ms = (denoise_time - encode_time) * 1000
            print(f"Elapsed time for denoising video: {elapsed_time_ms:.2f} ms")

        if not output_type == "latent":
            # cast back to fp32 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float32)
            frames = self.decode_latents(latents_all, num_frames, decode_chunk_size)

            if track_time:
                decode_time = time.time()
                elapsed_time_ms = (decode_time - denoise_time) * 1000
                print(f"Elapsed time for decoding video: {elapsed_time_ms:.2f} ms")

            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        else:
            frames = latents_all

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
