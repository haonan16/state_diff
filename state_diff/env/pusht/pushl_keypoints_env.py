from typing import Dict, Sequence, Union, Optional
from gym import spaces
from state_diff.env.pusht.pushl_env import PushLEnv
from state_diff.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np


class PushLKeypointsEnv(PushLEnv):
    def __init__(self,
                 legacy=False,
                 block_cog=None, 
                 damping=None,
                 render_size=96,
                 keypoint_visible_rate=1.0, 
                 agent_keypoints=False,
                 draw_keypoints=False,
                 reset_to_state=None,
                 render_action=True,
                 local_keypoint_map: Dict[str, np.ndarray]=None, 
                 color_map: Optional[Dict[str, np.ndarray]]=None,
                 obs_key='keypoint',
                 state_key='state',
                 action_key='action',
                 num_agents=1,
                 object_scale=30,
                 default_rendering=True):
        super().__init__(legacy=legacy, block_cog=block_cog, damping=damping, 
                         render_size=render_size, reset_to_state=reset_to_state, 
                         render_action=render_action, object_scale=object_scale,
                         default_rendering=default_rendering)
        ws = self.window_size

        if local_keypoint_map is None:
            kp_kwargs = self.genenerate_keypoint_manager_params(self)
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        if num_agents == 1:
            self.agentmode = 0
        elif num_agents == 2:
            self.agentmode = 1
        
        if self.obs_key == 'keypoint':
            DL0objectkps = np.prod(local_keypoint_map['L0_object'].shape)
            DL1objectkps = np.prod(local_keypoint_map['L1_object'].shape)

        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = DL0objectkps + DL1objectkps
        if agent_keypoints:
            Do += Dagentkps if self.agentmode == 0 else Dagentkps * 2
        else:
            Do += Dagentpos if self.agentmode == 0 else Dagentpos * 2

        Dobs = Do * 2
            
        low = np.zeros((Dobs,), dtype=np.float64)

        high = np.full_like(low, ws)
        high[Do:] = 1.

        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float64)

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(local_keypoint_map=local_keypoint_map, color_map=color_map)
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls, instance=None):
        if instance == None:
            env = PushLEnv()
            kp_manager = PymunkKeypointManager.create_from_pushl_env(env)
        else:
            env = PushLEnv(object_scale=instance.object_scale)
            kp_manager = PymunkKeypointManager.create_from_pushl_env(env, object_scale=env.object_scale)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        if self.obs_key == 'keypoint':
            obs, obs_mask = self._get_keypoint_observation()

        if not self.agent_keypoints:
            agent1_pos = np.array(self.agent1.position)
            obs = np.concatenate([obs, agent1_pos])
            obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])
            
            if self.agentmode != 0:
                agent2_pos = np.array(self.agent2.position)
                obs = np.concatenate([obs, agent2_pos])
                obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])

            
        obs = np.concatenate([obs, obs_mask.astype(obs.dtype)], axis=0)

        return obs

    def _get_keypoint_observation(self):
        obj_map = {'L0_object': self.L0_object, 'L1_object': self.L1_object}
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
        kps = np.concatenate(list(kp_map.values()), axis=0)

        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:, None], 2, axis=1)

        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        self.draw_kp_map = {
            'L0_object': vis_kps[:len(kp_map['L0_object'])],
            'L1_object': vis_kps[len(kp_map['L0_object']):len(kp_map['L0_object'])+len(kp_map['L1_object'])]
        }
        if self.agent_keypoints:
            self.draw_kp_map['agent'] = vis_kps[len(kp_map['L0_object'])+len(kp_map['L1_object']):]

        return kps.flatten(), kps_mask.flatten()


    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img
