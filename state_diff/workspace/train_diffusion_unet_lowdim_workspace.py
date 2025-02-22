if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import copy
import numpy as np
import random
import wandb
import tqdm
import shutil
import pdb
import time

from state_diff.common.pytorch_util import dict_apply, optimizer_to
from state_diff.workspace.base_workspace import BaseWorkspace
from state_diff.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from state_diff.policy.diffusion_traj_unet_lowdim_policy import DiffusionTrajUnetLowdimPolicy
from state_diff.dataset.base_dataset import BaseLowdimDataset
from state_diff.env_runner.base_lowdim_runner import BaseLowdimRunner
from state_diff.common.checkpoint_util import TopKCheckpointManager
from state_diff.common.json_logger import JsonLogger
from state_diff.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(cfg.training.seed)  # For multi-GPU setups
        
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        if 'traj' in cfg.policy._target_:
            self.model: DiffusionTrajUnetLowdimPolicy
            self.ema_model: DiffusionTrajUnetLowdimPolicy = None
        else:
            self.model: DiffusionUnetLowdimPolicy
            self.ema_model: DiffusionUnetLowdimPolicy = None

        self.model = hydra.utils.instantiate(cfg.policy)

        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self, rank=0, world_size=1):
        cfg = copy.deepcopy(self.cfg)

        # Add a variable to keep track of the best validation loss
        best_val_loss = float('inf')  # Initialize to a high value
        epochs_without_improvement = 0  # Count epochs without improvement in validation loss
        max_epochs_without_improvement = cfg.max_epochs_without_improvement  # Stop training if no improvement in 30 epochs

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        if world_size > 1:
            sampler = DistributedSampler(dataset)
            cfg.dataloader.shuffle = False
        else:
            sampler = None
        train_dataloader = DataLoader(
            dataset, 
            sampler=sampler,
            **cfg.dataloader
        )


        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        if  rank == 0:
            # configure env runner
            env_runner: BaseLowdimRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir,
                debug=cfg.debug)
            assert isinstance(env_runner, BaseLowdimRunner)

            # configure logging
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        if world_size > 1:
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(cfg.training.device)
        
        if world_size>1:
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)
            underlying_model = self.model.module


        else:
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            underlying_model = self.model
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        if rank == 0 or world_size == 1:
            log_path = os.path.join(self.output_dir, 'logs.json.txt')
        else:
            log_path = None
            
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses, train_diffuse_losses, train_inv_losses = [], [], []
                train_losses_dict = {}
                inference_times = []

                if world_size > 1:
                    train_dataloader.sampler.set_epoch(self.epoch)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss_dict = underlying_model.compute_loss(batch) 
                        raw_loss = raw_loss_dict if torch.is_tensor(raw_loss_dict) else raw_loss_dict['loss']
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(underlying_model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        if not torch.is_tensor(raw_loss_dict):
                            for key, value in raw_loss_dict.items():
                                if key == 'loss':
                                    continue
                                train_losses_dict[key] = train_losses_dict.get(key, [])
                                train_losses_dict[key].append(value.item() if isinstance(value, torch.Tensor) else value)

                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if rank == 0:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                for key, value in train_losses_dict.items():
                    if len(value) > 0:
                        step_log[key] = np.mean(value)


                # ========= eval for this epoch ==========
                policy = underlying_model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and (rank == 0):
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and (rank == 0):
                    with torch.no_grad():
                        val_losses, val_diffuse_losses, val_inv_losses = [], [], []
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_dict = underlying_model.compute_loss(batch)
                                loss = loss_dict if torch.is_tensor(loss_dict) else loss_dict['loss']
                                val_losses.append(loss)
                                if not torch.is_tensor(loss_dict):
                                    val_diffuse_losses.append(loss_dict['diffuse_loss'])
                                    val_inv_losses.append(loss_dict['inv_loss'])

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_diffuse_losses) > 0:
                            val_diffuse_loss = torch.nanmean(torch.tensor(val_diffuse_losses)).item()
                            step_log['val_diffuse_loss'] = val_diffuse_loss
                        if len(val_inv_losses) > 0:
                            val_inv_loss = torch.nanmean(torch.tensor(val_inv_losses)).item()
                            step_log['val_inv_loss'] = val_inv_loss 

                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            
                            # Update best validation loss and reset improvement counter
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                epochs_without_improvement = 0
                            else:
                                epochs_without_improvement += 1
                            # Check if the validation loss has not improved for a set number of epochs
                            if epochs_without_improvement >= max_epochs_without_improvement:
                                print("Stopping training due to lack of improvement in validation performance.")
                                break


                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and (rank == 0):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        # Measure inference time
                        start_time = time.time()
                        result = policy.predict_action(obs_dict)
                        end_time = time.time()
                        inference_time = end_time - start_time
                        inference_times.append(inference_time)

                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        if 'traj' in cfg.policy._target_ and not cfg.policy.skip_inv_model:
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action[:,cfg.policy.n_his_inv-1:-cfg.policy.n_fut_inv])
                        else:
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action[:,:pred_action.shape[1]])
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and (rank == 0):
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()
                
                if inference_times:
                    avg_inference_time = np.mean(inference_times)
                    step_log['avg_inference_time'] = avg_inference_time

                # end of epoch
                # log of last step is combined with validation and rollout
                if  rank == 0:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()