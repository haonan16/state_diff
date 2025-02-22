import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import copy
import wandb.sdk.data_types.video as wv
from state_diff.env.pusht.pushl_keypoints_env import PushLKeypointsEnv

from state_diff.gym_util.async_vector_env import AsyncVectorEnv
# from state_diff.gym_util.sync_vector_env import SyncVectorEnv
from state_diff.gym_util.multistep_wrapper import MultiStepWrapper
from state_diff.gym_util.video_recording_wrapper import VideoRecordingWrapper
from state_diff.gym_util.video_recorder import VideoRecorder
from state_diff.policy.base_lowdim_policy import BaseLowdimPolicy
from state_diff.common.pytorch_util import dict_apply
from state_diff.env_runner.base_lowdim_runner import BaseLowdimRunner

class PushKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            debug=False,
            **kwargs
        ):
        super().__init__(output_dir)

        env_type = kwargs.get('env_type', 'pushl_traj_lowdim')
        if env_type == 'pushl_traj_lowdim' or env_type == 'pushl_lowdim':
            KeypointsEnv = PushLKeypointsEnv
        else:
            raise ValueError("Invalid env_type.")    
        
        self.env_type = env_type


        self.debug = debug
        if self.debug:
            n_train = 2
            n_test = 1

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        if KeypointsEnv == PushLKeypointsEnv or KeypointsEnv == PushLAsymKeypointsEnv or KeypointsEnv == PushTKeypointsEnv:
            kp_kwargs = KeypointsEnv.genenerate_keypoint_manager_params()
            if 'local_allpoints_map' in kp_kwargs:
                kp_kwargs.pop('local_allpoints_map')
            render_dict = {'render_size': 512, 'default_rendering': False}
            kp_kwargs.update(render_dict)
            
        elif KeypointsEnv == PlanarSweepingEnv:
            kp_kwargs = {'sincos_vs_2points': kwargs.get('sincos_vs_2points', True)}
        else:
            kp_kwargs = dict()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    KeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        num_agents=kwargs.get('num_agents', 1),
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', "train_" + wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', "test_" + wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.visualize_state_denoising = kwargs.get('visualize_state_denoising', False)
        self.num_agents = kwargs.get('num_agents', 1)
        self.action_dim = kwargs.get('action_dim', 2)
        
    def run(self, policy: BaseLowdimPolicy):
        
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_infos = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushLKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                action_dict.pop('deno_trajectories', None)

                # Device transfer and conversion to numpy
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_infos[this_global_slice] = env.call('get_infos')[this_local_slice]

        # import pdb; pdb.set_trace()

        # log
        combined_stats = collections.defaultdict(lambda: {'max_rewards': [], 'success': []})
        log_data = dict()

        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])

            success_value = all_infos[i]['success'][-1]

            # Store max_reward and success data for each prefix
            combined_stats[prefix]['max_rewards'].append(max_reward)
            combined_stats[prefix]['success'].append(success_value)

            # Store individual log data
            log_data[f"{prefix}sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, stats in combined_stats.items():
            mean_max_reward = np.mean(stats['max_rewards'])
            mean_success = np.mean(stats['success'])

            log_data[f"{prefix}mean_score"] = mean_max_reward
            log_data[f"{prefix}success_rate"] = mean_success


        return log_data
