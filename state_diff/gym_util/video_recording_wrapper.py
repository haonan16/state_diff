import gym
import numpy as np
from state_diff.gym_util.video_recorder import VideoRecorder
import cv2
import matplotlib.pyplot as plt
class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs
    
    def step(self, action):
        deno_actions = None
        if len(action.shape) == 2:
            deno_actions = action
            action = action[-1]
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)

            if deno_actions is not None:
                frame = self.overlay_points_on_frame(frame, deno_actions)

            assert frame.dtype == np.uint8

            self.video_recoder.write_frame(frame)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
    


    def overlay_points_on_frame(self, frame, points, color=(255, 0, 0), gradient=False):

        """Helper method to overlay deno_actions as points on the frame."""
        # if points.shape[-1] == 4:
        #     points = np.concatenate([points[:, :2], points[:, 2:]], axis=0)
        # elif points.shape[-1] == 6:
        #     points = np.concatenate([points[:, :2], points[:, 2:4], points[:, 4:]], axis=0)
        
        # Mapping of possible dimensions to slicing rules
        slicing_map = {
            4: [slice(0, 2), slice(2, 4)],
            6: [slice(0, 2), slice(2, 4), slice(4, 6)]
        }

        # Check if the last dimension matches a supported format
        slices = slicing_map.get(points.shape[-1])
        if slices:
            points = np.concatenate([points[:, sl] for sl in slices], axis=0)
        frame_size = frame.shape[0]
        if frame_size == 96:
            radians = 1
        elif frame_size == 512:
            radius = 4
            
        if gradient:
            colors = plt.cm.plasma(np.linspace(0, 1, points.shape[0]))[:, :3] * 255
        else:
            colors = [color] * points.shape[0]

        coord = (points / 512 * frame_size).astype(np.int32)
        for idx, point in enumerate(coord):
            x, y = map(int, point)
            cv2.circle(frame, (x, y), radius=radius, color=tuple(colors[idx]), thickness=-1)
        return frame

