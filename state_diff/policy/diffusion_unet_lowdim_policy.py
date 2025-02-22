from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from state_diff.model.common.normalizer import LinearNormalizer
from state_diff.policy.base_lowdim_policy import BaseLowdimPolicy
from state_diff.model.diffusion.conditional_unet1d import ConditionalUnet1D
from state_diff.model.diffusion.mask_generator import LowdimMaskGenerator
from state_diff.model.common.losses import normal_kl, discretized_gaussian_log_likelihood
from state_diff.model.common.nn import mean_flat
from state_diff.model.common.resample import create_named_schedule_sampler, LossAwareSampler

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            output_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.sample_temperature = kwargs.get('sample_temperature', 1.0)

        self.kwargs = kwargs
        self.schedule_sampler = None
        if self.kwargs.get('schedule_sampler', None) == 'loss-second-moment':
            self.schedule_sampler = create_named_schedule_sampler(self.kwargs.get('schedule_sampler', None) , self.noise_scheduler.config.num_train_timesteps ) 

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
            device = self.device
            # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
            self.betas = self.noise_scheduler.betas.to(device)
            self.alphas = self.noise_scheduler.alphas.to(device)
            self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = torch.cat((torch.tensor([1]).to(device)  , self.alphas_cumprod[:-1]))
            self.alphas_cumprod_next = torch.cat((self.alphas_cumprod[1:], torch.tensor([0]).to(device) ))

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

            # calculations for posterior q(x_{t-1} | x_t, x_0)
            self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            self.posterior_log_variance_clipped = torch.log(
                torch.cat((torch.tensor([self.posterior_variance[1]]).to(device), self.posterior_variance[1:]))
            )
            self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
            )

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = self.sample_temperature*torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for idx, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            
            if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
                # model_output: (B, T, C) -> (B, C, T)
                model_output = model_output.swapaxes(1,2)
                trajectory = trajectory.swapaxes(1,2)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                ).prev_sample
            
            # trajectory: (B, C, T) -> (B, T, C)
            if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
                trajectory = trajectory.swapaxes(1,2)


        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        # nsample: (B, T, Da+Do)
        # deno_trajectories: (B, number of intermediate steps, T, Da+Do)
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
            
        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
            if deno_trajectories is not None:
                deno_trajectories = deno_trajectories[:, :, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        new_noise = noise + self.kwargs.get('input_pertub', 0) * torch.randn_like(noise)

        bsz = trajectory.shape[0]

        # # Sample a random timestep for each image

        if self.kwargs.get('schedule_sampler', None) == 'loss-second-moment':
            timesteps, weights = self.schedule_sampler.sample(bsz, trajectory.device)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # eqution (4) in DDPM
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, new_noise, timesteps)
        
        terms = {}
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        
        if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
            B, C = noisy_trajectory.shape[0], noisy_trajectory.shape[2]
            assert pred.shape == (B, noisy_trajectory.shape[1], C * 2)
            model_output, model_var_values = torch.split(pred, C, dim=2)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=2)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, 
                r=frozen_out: r,
                x_start=trajectory,
                x_t=noisy_trajectory,
                t=timesteps,
                clip_denoised=False,
                model_kwargs={'local_cond': local_cond, 'global_cond': global_cond}
            )["output"]
            assert model_output.shape == noise.shape == trajectory.shape

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        if "vb" in terms:
            loss = F.mse_loss(model_output, target, reduction='none')
        else:
            loss = F.mse_loss(pred, target, reduction='none')

        loss = loss * loss_mask.type(loss.dtype)
        # loss = mean_flat(loss)
        # loss = loss + terms["vb"]*0.001 if "vb" in terms else loss # TODO

        if self.schedule_sampler != None and isinstance(self.schedule_sampler, LossAwareSampler):
            loss = mean_flat(loss)
            if "vb" in terms:
                loss = loss + terms["vb"]*self.kwargs.get('vb_factor', 0.01)  

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    timesteps, loss.detach()
                )
            loss = (loss * weights).mean()
        else:
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            if "vb" in terms:
                loss = mean_flat(loss)
                loss = loss + terms["vb"]*self.kwargs.get('vb_factor', 0.01) 

            loss = loss.mean()
        return loss



