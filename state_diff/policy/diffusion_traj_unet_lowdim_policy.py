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
from state_diff.policy.base_lowdim_policy import apply_conditioning_init

class DiffusionTrajUnetLowdimPolicy(BaseLowdimPolicy):
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
        self.obs_as_impainting = kwargs.get('obs_as_impainting', True)
        self.mask_generator = LowdimMaskGenerator(
            output_dim=obs_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond or self.obs_as_impainting) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = kwargs.get('num_agents', 1)
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.inv_hidden_dim = kwargs.get('inv_hidden_dim', 256)
        self.sample_temperature = kwargs.get('sample_temperature', 1.0)
        self.inv_factor = kwargs.get('inv_factor', 0.5)
        self.inv_model_include_agent_info = kwargs.get('inv_model_include_agent_info', True)
        self.skip_inv_model = kwargs.get('skip_inv_model', False)
        self.n_his_inv = kwargs.get('n_his_inv', 1)
        self.n_fut_inv = kwargs.get('n_fut_inv', 1)
        self.inv_act = kwargs.get('inv_act', 'relu')


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
        if not self.skip_inv_model:
            self.inv_model_input_dim = self.obs_dim * (self.n_his_inv+self.n_fut_inv) if self.inv_model_include_agent_info else (self.obs_dim-self.action_dim) * (self.n_his_inv+self.n_fut_inv)
            if self.num_agents == 1:
                self.inv_model = self.create_inv_model()
            else:
                self.inv_model1 = self.create_inv_model()
                self.inv_model2 = self.create_inv_model()
                
    def create_inv_model(self):
        activation = nn.ReLU if self.inv_act == 'relu' else nn.Mish
        return nn.Sequential(
            nn.Linear(self.inv_model_input_dim, self.inv_hidden_dim),
            activation(),
            nn.Linear(self.inv_hidden_dim, self.inv_hidden_dim),
            activation(),
            nn.Linear(self.inv_hidden_dim, self.action_dim),
        )

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        B, T, obs_dim = condition_data.shape
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = self.sample_temperature*torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        if self.obs_as_impainting:
            trajectory = apply_conditioning_init(trajectory, condition_data, n_obs_steps=self.n_obs_steps)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for idx, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            if self.obs_as_impainting:
                trajectory = apply_conditioning_init(trajectory, condition_data, n_obs_steps=self.n_obs_steps)
            else: 
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
        if self.obs_as_impainting:
            trajectory = apply_conditioning_init(trajectory, condition_data, n_obs_steps=self.n_obs_steps)
        else:
            trajectory[condition_mask] = condition_data[condition_mask]        

        if not self.skip_inv_model:
            
            if not self.inv_model_include_agent_info:
                inv_input_traj = trajectory[:, :, :-self.action_dim]
            else:
                inv_input_traj = trajectory

            obs_comb = torch.tensor([])
            inv_traj_slice_len = T - self.n_his_inv - self.n_fut_inv + 1
            # Create a range of indices for slicing
            indices = torch.arange(self.n_his_inv + self.n_fut_inv).unsqueeze(1) + torch.arange(inv_traj_slice_len)
            # Use advanced indexing to get the slices in one go
            traj_slices = torch.swapaxes(inv_input_traj[:,indices,:], 1, 2) # swap the 
            # Reshape the tensor to the desired shape and concatenate along the last dimension
            obs_comb = traj_slices.reshape(-1, self.inv_model_input_dim)

            if self.num_agents == 1:
                action = self.inv_model(obs_comb)
            else:
                action1 = self.inv_model1(obs_comb)
                action2 = self.inv_model2(obs_comb)
                action = torch.cat([action1, action2], dim=-1)
                
            action = action.reshape(B, T - self.n_his_inv - self.n_fut_inv + 1, self.action_dim*self.num_agents)
        else:
            action = trajectory[:, self.n_obs_steps:, -self.action_dim:]

        return action

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], action_slice=None) -> Dict[str, torch.Tensor]:
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
        Da = self.obs_dim

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
            cond_data[:,:To] = nobs[:,:To]
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_mask[:,:To] = True
        else:
            # condition through impainting
            shape = (B, T, Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To] = nobs[:,:To]
            cond_mask[:,:To] = True

        # run sampling
        # nsample: (B, T, Da+Do)
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred, action_slice)
        
    
        # get action
        if self.pred_action_steps_only:
            action = action_pred
        elif self.skip_inv_model:
            action = action_pred[:, :self.n_action_steps]
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]

        
        result = {
            'action': action,
            'action_pred': action_pred,
        }

        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, action_slice=None):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action'] 
        # If action_slice is provided, use it to slice the action tensor
        if action_slice is not None:
            action = action[:, :, action_slice]

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
            condition_mask = torch.zeros_like(obs, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(obs.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(obs.shape, device=obs.device)
        new_noise = noise + self.kwargs.get('input_pertub', 0) * torch.randn_like(noise)

        bsz = obs.shape[0]
        T = obs.shape[1]

        # # Sample a random timestep for each image

        if self.kwargs.get('schedule_sampler', None) == 'loss-second-moment':
            timesteps, weights = self.schedule_sampler.sample(bsz, obs.device)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=obs.device
            ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # eqution (4) in DDPM
        noisy_obs = self.noise_scheduler.add_noise(
            obs, new_noise, timesteps)
        
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        if self.obs_as_impainting:
            noisy_obs = apply_conditioning_init(noisy_obs, obs, n_obs_steps=self.n_obs_steps)
        else:
            noisy_obs[condition_mask] = obs[condition_mask]


        # Predict the noise residual
        pred = self.model(noisy_obs, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        
        terms = {}
        
        if self.kwargs.get('variance_type', None) in  ['learned', 'learned_range']:
            B, C = noisy_obs.shape[0], noisy_obs.shape[2]
            assert pred.shape == (B, noisy_obs.shape[1], C * 2)
            model_output, model_var_values = torch.split(pred, C, dim=2)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=2)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, 
                r=frozen_out: r,
                x_start=obs,
                x_t=noisy_obs,
                t=timesteps,
                clip_denoised=False,
                model_kwargs={'local_cond': local_cond, 'global_cond': global_cond}
            )["output"]
            assert model_output.shape == noise.shape == obs.shape

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = obs
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        if "vb" in terms:
            diffuse_loss = F.mse_loss(model_output, target, reduction='none')
        else:
            diffuse_loss = F.mse_loss(pred, target, reduction='none')

        diffuse_loss = diffuse_loss * loss_mask.type(diffuse_loss.dtype)

        if self.schedule_sampler != None and isinstance(self.schedule_sampler, LossAwareSampler):
            diffuse_loss = mean_flat(diffuse_loss)
            if "vb" in terms:
                diffuse_loss = diffuse_loss + terms["vb"]*self.kwargs.get('vb_factor', 0.01)  # TODO

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    timesteps, diffuse_loss.detach()
                )
            diffuse_loss = (diffuse_loss * weights).mean()
        else:
            diffuse_loss = reduce(diffuse_loss, 'b ... -> b (...)', 'mean')
            if "vb" in terms:
                diffuse_loss = mean_flat(diffuse_loss)
                diffuse_loss = diffuse_loss + terms["vb"]*self.kwargs.get('vb_factor', 0.01) # TODO

            diffuse_loss = diffuse_loss.mean()

        if not self.skip_inv_model:
            # Calculating inv loss

            if not self.inv_model_include_agent_info:
                inv_input_traj = obs[:, :, :-self.action_dim]
            else:
                inv_input_traj = obs

            inv_traj_slice_len = T - self.n_his_inv - self.n_fut_inv + 1
            # Create a range of indices for slicing
            indices = torch.arange(self.n_his_inv + self.n_fut_inv).unsqueeze(1) + torch.arange(inv_traj_slice_len)
            # Use advanced indexing to get the slices in one go
            traj_slices = torch.swapaxes(inv_input_traj[:,indices,:], 1, 2) # swap the 
            # Reshape the tensor to the desired shape and concatenate along the last dimension
            x_comb_t = traj_slices.reshape(-1, self.inv_model_input_dim)

            a_t = action[:, self.n_his_inv-1 : T-self.n_fut_inv, :]

            a_t = a_t.reshape(-1, self.action_dim * self.num_agents)
            if self.num_agents == 1:
                pred_a_t = self.inv_model(x_comb_t)
            else:
                pred_a_t1 = self.inv_model1(x_comb_t)
                pred_a_t2 = self.inv_model2(x_comb_t)
                pred_a_t = torch.cat([pred_a_t1, pred_a_t2], dim=-1)   
            inv_loss = F.mse_loss(pred_a_t, a_t)
            loss = (1 - self.inv_factor) * diffuse_loss + self.inv_factor * inv_loss
        else:
            loss = diffuse_loss
        loss_dict = {
            'loss': loss,
            'diffuse_loss': diffuse_loss,
            'inv_loss': inv_loss if not self.skip_inv_model else None,
        }
        return loss_dict