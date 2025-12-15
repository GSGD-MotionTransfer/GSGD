import os
import sys

# Add reenact-anything to path for utils imports
sys.path.append(os.path.join(os.path.dirname(__file__), "reenact-anything"))

import einops
import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from einops import rearrange
from scipy.spatial import Delaunay
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as nnf
from utils.embds_inversion_utils import pipeline_decode_latents

# Assume it knows the path of the train_reenact submodule
from utils.train_utils import _get_add_time_ids, rand_log_normal, tensor_to_vae_latent
from utils.video_utils import export_to_gif


# =============================================
# ===== Helper function for SDS gradients =====
# =============================================
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class SDSSVDLoss(nn.Module):

    _global_pipe = None

    def __init__(
        self,
        pipeline,
        first_frame_cond_per_view,
        motion_embedding_per_view,
        cfg,
        device,
        run_folder="./",
        debug=False,
    ):
        super(SDSSVDLoss, self).__init__()

        self.cfg = cfg
        self.device = device
        self.run_folder = run_folder
        self.debug = debug
        self.first_frame_cond_per_view = first_frame_cond_per_view
        self.motion_embedding_per_view = motion_embedding_per_view

        self.pipe = pipeline

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        if cfg.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.del_text_encoders:
            del self.pipe.tokenizer
            del self.pipe.text_encoder

        self.conditional_latents_per_view, self.added_time_ids = (
            self.prepare_static_latents(
                self.first_frame_cond_per_view,
                list(self.motion_embedding_per_view.values())[0].dtype,
            )
        )

    def log_latents(self, step=""):
        """
        Note - This function shouldn't be run often, since it clears the GPU's cache
        """
        save_latents_f = self._save_latents
        suffix = "_" + str(self.last_t_index.item()) + ".gif"

        original_latents = self.clean_noise_from_latents(
            self.last_noised_latent_zt,
            self.last_v_target,
            self.last_t,
        )
        save_latents_f(
            original_latents,
            f"{step}_original_latents{suffix}",
        )

        generated_latents = self.clean_noise_from_latents(
            self.last_noised_latent_zt,
            self.last_v_pred,
            self.last_t,
        )
        save_latents_f(
            generated_latents,
            f"{step}_generated_latents{suffix}",
        )
        save_latents_f(self.last_noised_latent_zt, f"{step}_noisy_latents{suffix}")
        save_latents_f(self.last_v_target, f"{step}_v_target{suffix}")
        save_latents_f(self.last_v_pred, f"{step}_predicted_v{suffix}")

        del original_latents
        del generated_latents
        torch.cuda.empty_cache()

    def _save_latents(self, latents, file_name):
        num_frames = latents.shape[1]
        # decoded_frames = self.pipe.vae.decoder(latent_for_vae_decoder)
        decoded_frames = pipeline_decode_latents(
            self.pipe, latents, num_frames=num_frames, decode_chunk_size=2
        )[0]

        for i in range(num_frames):
            img = decoded_frames[i]
            decoded_frames[i] = np.array(img)

        # Save the video
        save_to_folder = os.path.join(self.run_folder, "debugging_decoded_latents")
        if not os.path.isdir(save_to_folder):
            os.mkdir(save_to_folder)
        save_path = os.path.join(save_to_folder, file_name)
        print(f"Saving decoded latents to {save_path}")
        export_to_gif(decoded_frames, save_path, 8)

        # Clear for later
        del latents
        del decoded_frames
        torch.cuda.empty_cache()

    def sanity_encode_decode(self, x_to_encode, step):
        # NOTE - it get too heavy for the video model here. For some reason the following line add 15GB to the gpu memeory for itself, then it collases on the "saving" process
        cur_latents = self.prepare_latents(x_to_encode)
        self._save_latents(cur_latents, f"{step}_encoded_decoded.gif")

    def prepare_static_latents(self, first_frame_cond_per_view, motion_embedding_dtype):
        """
        Parameters:
        first_frame_cond_per_view (dict): A dictionary where each key is an angle and each value is a torch.Tensor with dimensions [batch_size, 1, channels, height, width].
        motion_embedding_dtype (torch.dtype): ONLY to take its dtype.
        """
        first_frame_cond_example = list(first_frame_cond_per_view.values())[0]
        # First frame conditioning
        bsz = first_frame_cond_example.shape[0]
        cond_noise = torch.randn_like(first_frame_cond_example)
        cond_sigmas = rand_log_normal(
            shape=[
                bsz,
            ],
            loc=-3.0,
            scale=0.5,
        ).to(first_frame_cond_example.device)
        noise_aug_strength = cond_sigmas[0]  # TODO: support batch > 1
        cond_sigmas = cond_sigmas[:, None, None, None, None]

        conditional_latents_per_view = {}
        for angle, first_frame_cond in self.first_frame_cond_per_view.items():
            conditional_pixel_values = cond_noise * cond_sigmas + first_frame_cond
            conditional_latents = tensor_to_vae_latent(
                conditional_pixel_values, self.pipe.vae
            )[:, 0, :, :, :]
            conditional_latents = (
                conditional_latents / self.pipe.vae.config.scaling_factor
            )
            conditional_latents_per_view[angle] = conditional_latents

        # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
        # However, I am unable to fully align with the calculation method of the motion score,
        # so I adopted this approach. The same applies to the 'fps' (frames per second).
        added_time_ids = _get_add_time_ids(
            self.pipe.unet,
            7,  # fixed
            127,  # motion_bucket_id = 127, fixed
            noise_aug_strength,  # noise_aug_strength == cond_sigmas
            motion_embedding_dtype,
            bsz,
        )
        added_time_ids = added_time_ids.to(first_frame_cond_example.device)
        return conditional_latents_per_view, added_time_ids

    def prepare_latents(self, x):
        """
        Parameters:
        x (torch.Tensor): The augmented input tensor with dimensions [batch_size, frames, channels, height, width].
        """
        assert (
            x.min() >= -1.0 and x.max() <= 1.0 and x.min() < -0.2
        ), "x should be between -1.0 and 1.0, but got min: {}, max: {}".format(
            x.min(), x.max()
        )
        # Encode the video
        video_length = x.shape[1]

        x = rearrange(x, "b f c h w -> (b f) c h w")

        # def encode_fn(x):
        #     return self.pipe.vae.encode(x).latent_dist.sample()

        # latents = torch.utils.checkpoint.checkpoint(
        #     encode_fn, x
        # )  # Checkpoint is used to save memory, otherwise collapses OOM on 48GB GPU
        latents = self.pipe.vae.encode(x).latent_dist.sample()

        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        latents = latents * self.pipe.vae.config.scaling_factor
        return latents

    def add_noise_to_latents(self, latent_z, timestep, return_noise=True, eps=None):

        # sample noise if not given some as an input
        if eps is None:
            if self.cfg.same_noise_for_frames:  # This works badly. Do not use.
                eps = torch.randn_like(
                    latent_z[:, :, 0, :, :]
                )  # create noise for single frame
                eps = einops.repeat(eps, "b c h w -> b c f h w", f=latent_z.shape[2])
            else:
                eps = torch.randn_like(latent_z)

        # zt = alpha_t * latent_z + sigma_t * eps
        assert (
            self.pipe.scheduler.begin_index is None
        ), "I intend this line to be run https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py#L704 otherwise I might be wrong"
        noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

        if return_noise:
            return noised_latent_zt, eps

        return noised_latent_zt

    def clean_noise_from_latents(
        self,
        noisy_samples: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: torch.IntTensor,
    ) -> torch.Tensor:
        # In the original code, here the scheduler is used to step a single step formard (xt to xt-1, not to x0) - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L598
        # Here we use it to step from xt to x0
        # "noise_pred" can also be v_prediction, not neccesarily the noise prediction

        # See here the function used here - https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py#L684
        print("prediction_type scheduler", self.pipe.scheduler.config.prediction_type)
        cleaned_samples = self.pipe.scheduler.step(
            noise_pred, timestep, noisy_samples
        ).pred_original_sample
        return cleaned_samples

    # overload this if inheriting for VSD etc.
    def get_sds_eps_to_subract(self, eps_orig, z_in, timestep_in):
        return eps_orig

    def drop_nans(self, grads):
        assert torch.isfinite(grads).all()
        if torch.isnan(grads).any():
            print("Warning. {} nans in the gradient", torch.isnan(grads).sum())
        return torch.nan_to_num(grads.detach().float(), 0.0, 0.0, 0.0)

    def get_grad_weights(self, timestep):
        weighing = (1 + self.pipe.scheduler.sigmas[timestep] ** 2) * (
            self.pipe.scheduler.sigmas[timestep] ** -2.0
        )
        return weighing

    def sds_grads(self, latent_z, angle):
        if angle is None:
            angle = list(self.motion_embedding_per_view.keys())[0]
            Warning("No angle provided. Using the first one {}".format(angle))

        with torch.no_grad():
            # sample timesteps
            timestep = torch.randint(
                low=self.cfg.sds_timestep_low,
                high=min(950, self.cfg.timesteps)
                - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device,
                dtype=torch.long,
            )

            # In the case of the euler discrete scheduler, the timestep is derived from the sigmas, see here - https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py#L246
            scheduler_timestep = (
                len(self.pipe.scheduler.alphas) - timestep.item()
            )  # - timestep since sigmas[0] is the highest noise
            # Note - len(sigmas) = 1001, sigmas are derived from alphas, then added an "0" to its end. len(alphas) = 1000.
            euler_sched_matching_sigma = self.pipe.scheduler.sigmas[scheduler_timestep]
            euler_sched_matching_timestep = torch.Tensor(
                [0.25 * euler_sched_matching_sigma.log()]
            ).to(self.device)

            # add noise
            noised_latent_zt, eps = self.add_noise_to_latents(
                latent_z, euler_sched_matching_timestep, return_noise=True
            )
            scaled_noised_latent_zt = self.pipe.scheduler.scale_model_input(  # they do that also in train_svd.py, just themselves not using this function rather implenet themselves the same
                noised_latent_zt, euler_sched_matching_timestep
            )
            conditional_latents = (
                self.conditional_latents_per_view[angle]
                .unsqueeze(1)
                .repeat(1, noised_latent_zt.shape[1], 1, 1, 1)
            )
            inp_noisy_latents = torch.cat(
                [scaled_noised_latent_zt, conditional_latents], dim=2
            )

            # denoise
            # z_in = torch.cat(
            #     [noised_latent_zt] * 2
            # )  # expand latents for classifier free guidance
            # timestep_in = torch.cat([timestep] * 2)
            # No classifier free guidance
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            v_pred = self.pipe.unet(
                inp_noisy_latents,
                euler_sched_matching_timestep,  # timestep,  # maybe I need the sigma induced "timestep" here? and in the cleaning?
                encoder_hidden_states=self.motion_embedding_per_view[angle],
                added_time_ids=self.added_time_ids,
            ).sample

            v_target = (eps - euler_sched_matching_sigma * latent_z) / torch.sqrt(
                euler_sched_matching_sigma**2 + 1.0
            )

            w = self.get_grad_weights(scheduler_timestep)
            grad_z = (
                v_pred - v_target
            )  # * w # NOTE - not using weighting here. Not sure if I should or not.

            grad_z = self.drop_nans(grad_z)

        if self.debug:
            # Convert epsilon to v_prediction that the scheduler expects
            # v = alpha * eps - sigma * x
            alpha_t = self.pipe.scheduler.alphas[scheduler_timestep]
            sigma_t = self.pipe.scheduler.sigmas[scheduler_timestep]
            ideal_v_prediction_matching_eps = (
                alpha_t * eps - sigma_t * latent_z
            )  # TODO: save here the noisy latents after diffusing before scaling. I think? maybe should be after scaling?

            # For later if required:
            self.last_noised_latent_zt = noised_latent_zt  # TODO: save here the noisy latents after diffusing before scaling.
            self.last_eps = eps
            self.last_v_target = v_target
            self.last_v_pred = v_pred
            self.last_t = euler_sched_matching_timestep
            self.last_t_index = timestep

        return grad_z

    def forward(self, x_aug, angle, grad_scale=1.0):
        with torch.no_grad():
            latent_z = self.prepare_latents(x_aug)

        grad_z = grad_scale * self.sds_grads(latent_z, angle)

        sds_loss = SpecifyGradient.apply(latent_z, grad_z)
        # Process each frame separately to manage memory
        detached_x_aug = x_aug.detach()
        video_length = x_aug.shape[1]
        detached_x_aug.requires_grad_(True)

        max_chunk_size = 4  # 4: 40GB RAM. 8: 70GB. 11+: OOM on 80GB A100
        for chunk_start in range(0, video_length, max_chunk_size):
            # Get chunk of frames
            chunk_end = min(chunk_start + max_chunk_size, video_length)
            chunk = detached_x_aug[:, chunk_start:chunk_end]  # [B, chunk_size, C, H, W]

            # Encode chunk
            chunk_latents = self.prepare_latents(chunk)

            # Get corresponding gradients for this chunk
            chunk_grads = grad_z[:, chunk_start:chunk_end]

            # Apply gradients using SpecifyGradient
            chunk_loss = SpecifyGradient.apply(chunk_latents, chunk_grads)
            chunk_loss.backward()

            # Clear computation graph for this chunk to free memory
            chunk_latents = None
            chunk_grads = None
            chunk_loss = None
            torch.cuda.empty_cache()
        # Get accumulated gradients
        input_grad = detached_x_aug.grad.clone()
        detached_x_aug.grad = None
        detached_x_aug.requires_grad_(False)

        # Apply gradients manually since we've already accumulated them
        sds_loss = SpecifyGradient.apply(x_aug, input_grad)

        return sds_loss
