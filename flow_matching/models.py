import random
from flow_matching import schedulers
import pos_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tfd
import numpy as np
import networks
from omegaconf import open_dict
import guidance
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from flow_matching.hard_constraint_legalizer import Legalizer

class FlowMatchingModel(nn.Module):
    backbones = {
        "att_gnn": networks.AttGNN,
    }
    time_encoders = {
        "sinusoid": pos_encoding.SinusoidContEncoding,
    }
    def __init__(
            self, 
            backbone, 
            backbone_params,
            input_shape, 
            t_encoding_type, 
            t_encoding_dim, 
            legality_guidance_weight=0.0,
            hpwl_guidance_weight=0.0,
            grad_descent_steps=0,
            grad_descent_rate=0.0,
            alpha_init=0.0,  # for opt guidance
            alpha_lr=0.0,  # for opt guidance
            alpha_critical_factor=0.0,  # for opt guidance
            legality_potential_target=0.0,  # for opt guidance
            use_adam=False,  # for opt guidance
            max_diffusion_steps=1000,  # default eval timesteps
            guidance_mode="none",  # none | sgd | opt
            guidance_step = 1000,
            noise_schedule="linear", 
            mask_key=None, 
            use_mask_as_input=False, 
            device="cpu", 
            legality_softmax_factor_min=10.0,
            legality_softmax_factor_max=10.0,
            legality_softmax_critical_factor=0,  # fraction of time at which legality factor reaches max
            sampling_method="euler",  # euler | heun
            prior_noise_dist = 'normal',
            use_hard_constraint_in_sampling = False,
            hard_constraint_w_reg = 0.0,
            hard_constraint_w_dist = 0.0,
            **kwargs
    ):
        super().__init__()
        with open_dict(backbone_params):
            backbone_params.update({
                "in_node_features": input_shape[1],
                "out_node_features": input_shape[1],
                "t_encoding_dim": t_encoding_dim,
                "device": device,
                "mask_key": mask_key if use_mask_as_input else None,
            })

        print(f"Backbone name: {backbone}, params: {backbone_params}")
        if t_encoding_dim > 0:
            self.t_encoder = FlowMatchingModel.time_encoders[t_encoding_type](t_encoding_dim)
        self.mask_key = mask_key
        self.t_encoding_dim = t_encoding_dim
        self._reverse_model = FlowMatchingModel.backbones[backbone](**backbone_params)
        self.input_shape = input_shape
        self.max_diffusion_steps = max_diffusion_steps
        if noise_schedule == "linear":
            print("Using linear noise schedule for flow matching")
            self._scheduler = schedulers.LinearTimeScheduler()
        elif noise_schedule == "cosine":
            print("Using cosine noise schedule for flow matching. Note: only linear schedule is supported for flow matching Now.")
            self._scheduler = schedulers.CosineScheduler()
        else:
            raise NotImplementedError("Only linear schedule is supported for flow matching")
        
        self._lossfn = nn.MSELoss(reduction="mean")

        # cache some variables
        if prior_noise_dist == 'normal':
            print("Using normal noise distribution for flow matching")
            self._noise_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        elif prior_noise_dist == 'uniform':
            print("Using uniform noise distribution for flow matching")
            self._noise_dist = tfd.Uniform(torch.tensor([-1.0], device=device), torch.tensor([1.0], device=device))
        elif prior_noise_dist.startswith('normal_sigma'):
            # e.g., "normal_sigma_0.5"
            sigma = float(prior_noise_dist.split('_')[-1])
            print(f"Using normal distribution with σ={sigma} for flow matching")
            self._noise_dist = tfd.Normal(loc=torch.tensor([0.0], device=device),
                                        scale=torch.tensor([sigma], device=device))

        elif prior_noise_dist == 'truncated_normal':
            print("Using truncated normal distribution for flow matching")

            base_dist = tfd.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

            class TruncatedNormalDist:
                def sample(self, shape):
                    x = base_dist.sample(shape)
                    return torch.clamp(x, -1.0, 1.0)  # clip to layout region
            self._noise_dist = TruncatedNormalDist()


        self._t_dist = tfd.Uniform(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self.device = device
        
        # guidance parameters
        self.is_guided_sampling = (
            (legality_guidance_weight > 0.0 or hpwl_guidance_weight > 0.0 or alpha_lr > 0.0 or alpha_init > 0.0)
            and (grad_descent_steps > 0)
            and (grad_descent_rate > 0.0)
            and (guidance_mode != "none")
        )
        self.guidance_mode = guidance_mode
        self.legality_guidance_weight = legality_guidance_weight
        self.hpwl_guidance_weight = hpwl_guidance_weight
        self.grad_descent_steps = grad_descent_steps
        self.grad_descent_rate = grad_descent_rate
        self.alpha_init = alpha_init
        self.alpha_lr = alpha_lr
        self.alpha_critical_factor = alpha_critical_factor
        self.legality_potential_target = legality_potential_target
        self.use_adam = use_adam
        self.legality_softmax_factor_min = legality_softmax_factor_min
        self.legality_softmax_factor_max = legality_softmax_factor_max
        self.legality_softmax_critical_factor = legality_softmax_critical_factor
        self.sampling_method = sampling_method

        self.use_hard_constraint_in_sampling = use_hard_constraint_in_sampling
        self.hard_constraint_scores = { 'w_reg': hard_constraint_w_reg, 'w_dist': hard_constraint_w_dist }

        if self.use_hard_constraint_in_sampling:
            print(f"Using hard constraint in sampling.")


    def __call__(self, x, cond, t):
        # input: x=(B, V, F), cond=PyG Data (V nodes, E edges), t=(B)
        # output: vector filed: v_t=(B, V, F)
        t_embed = self.t_encoder(t)
        v_pred = self._reverse_model(x, cond, t_embed)
        return v_pred
    

            
    def loss(self, x, cond, _, idx=None, split="train"):
        B = x.shape[0]
        t = self._t_dist.sample((B,)).squeeze(dim=-1)

        mask = self.get_mask(x, cond)
        
        x_1 = self._noise_dist.sample(x.shape).squeeze(dim=-1)
        x_t = (1 - t).view(B,1,1) * x_1 + t.view(B,1,1) * x
        v_t = (x - x_1)

        v_pred = self(x_t, cond, t)

        # FM core loss
        loss = self._loss(v_pred, v_t, mask)
        metrics = {
            "loss": loss.item(),
            "velocity_norm": v_pred.norm().item(),
        }
        return loss, metrics


    def forward_samples(self, x, cond, intermediate_every=0):
        intermediates = []
        mask = None
        if self.mask_key and self.mask_key in cond:
            mask = self.get_mask(x, cond)
        
        step_size = intermediate_every / self.max_diffusion_steps if intermediate_every else 1
        x_0 = self._noise_dist.sample(x.shape).squeeze(dim=-1)  # (B, V, F)
        
        for t in torch.arange(0, 1 + 1e-9, step_size, device=self.device):
            t = t.expand(x.shape[0])
            x_t = (1 - t).view(x.shape[0], 1, 1) * x_0 + t.view(x.shape[0], 1, 1) * x  # (B, V, F)
            x_t = torch.where(mask, x, x_t) if mask is not None else x_t
            intermediates.append(x_t)
        
        return intermediates

    def reverse_samples(self, B, x_in, cond, num_timesteps=-1, intermediate_every=0, mask_override=None, method="euler"):
        # B: batch size, 
        # intermediate_every: how often to save intermediate results,
        # only support B = 1 when use_hard_constraint_in_sampling = True
        method = self.sampling_method.lower()
        if method not in ["euler"]:
            raise ValueError(f"Unknown sampling method: {method}")
        
        batch_shape = (B, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)

        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._noise_dist.sample(batch_shape).squeeze(dim=-1)  # (B, V, F)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x


        intermediates = [x]
        self._scheduler.set_timesteps(num_timesteps)
        timesteps = self._scheduler.timesteps
        if self.use_hard_constraint_in_sampling:
            assert B == 1, "Hard constraint sampling only supports batch size 1."
            legalizer = Legalizer(cond = cond, grid_res= 128, scores = self.hard_constraint_scores, device=self.device, dtype=x.dtype, padding=0.001)
            legalizer.run = torch.compile(legalizer.run, mode="reduce-overhead")
            print("Initialized Legalizer for hard constraint sampling.")

        
        for i, t in enumerate(timesteps[:-1]):
            t_next = timesteps[i + 1]
            t_vec = torch.tensor(t, device=self.device).expand(B)
            t_next_vec = torch.tensor(t_next, device=self.device).expand(B)

            # 1) Extrapolation: u1 = u_t + (1 - t) * v_theta(u_t)
            v_pred = self(x, cond, t_vec)
            u1 = x + (1 - t_vec.view(B, 1, 1)) * v_pred

            # 2) Correction: project u1 onto constraint manifold
            if self.use_hard_constraint_in_sampling:
                with torch.no_grad():
                    u1_hat = legalizer.run(u1)  
            else:
                u1_hat = u1

            # 3) Interpolation: u_{t'} = (1 - t') * x + t' * u1_hat
            x = (1 - t_next_vec.view(B, 1, 1)) * x + t_next_vec.view(B, 1, 1) * u1_hat

            x = torch.clamp(x, -2, 2)
            x = torch.where(mask, x_in, x)
            if intermediate_every and (i+1) % intermediate_every == 0:
                intermediates.append(x)
        return x, intermediates
    
    def get_mask(self, x, cond):
        if self.mask_key and self.mask_key in cond:
            mask = cond[self.mask_key]
            B, V, F = x.shape
            mask = mask.view(1, V, 1)
            return mask
        else:
            return None

    def _loss(self, prediction, target, mask=None):
        if mask is not None:
            numel = torch.numel(mask)
            squared_error = torch.square(prediction - target)
            squared_error.masked_fill_(mask, 0)
            mse = torch.mean(squared_error) * (numel / (numel - torch.sum(mask)))
            return mse
        else:
            return self._lossfn(prediction, target)
