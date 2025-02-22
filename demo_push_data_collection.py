import numpy as np
import click
import pygame
from state_diff.common.replay_buffer import ReplayBuffer

# Import env
from state_diff.env.pusht.pushl_keypoints_env import PushLKeypointsEnv


@click.command()
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-d', '--debug', default=True)


def main(render_size, control_hz, agentmode, debug):
    """
    Collect demonstration for the Push task.
    
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    
    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    
    # Setup Push Env
    
    kp_kwargs = PushLKeypointsEnv.genenerate_keypoint_manager_params()
    if 'local_allpoints_map' in kp_kwargs:
        kp_kwargs.pop('local_allpoints_map')
    env = PushLKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    output = 'data/pushl_single_new_env_failure_recov_200_setsizeto200whentraining_restdataisbad'

    
    env.select_draw_keypoint(False)
    agent1 = env.teleop_agent()
    output = output + '_single'
    print("Mode0: Single Mouse / Space Mouse")
    

    # Debug mode
    if debug == True:
        output = "data/temp"

    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    print("Data saved to: ", output)
    clock = pygame.time.Clock()

    # episode-level while loop
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f'starting seed {seed}')

        # set seed for env
        env.seed(seed)
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx}')

        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue

            # teleop start
            action = None
            info = env._get_info()
            
            act1 = agent1.act(obs)
            if not act1 is None:
                action = act1
                
                state = np.concatenate([info['pos_agent1'], info['L0_object_pose'], info['L1_object_pose']])
                n_L0_keypoints, n_L1_keypoints = 9, 10
                keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:n_L0_keypoints+n_L1_keypoints]
                

            # Save the episode if action is not None
            if (not action is None):
                data = {
                        'img': img,
                        'state': np.float32(state),
                        'keypoint': np.float32(keypoint),
                        'action': np.float32(action),
                        'n_contacts': np.float32([info['n_contacts']])
                    }
                episode.append(data)

            # step env and render
            obs, reward, done, info = env.step(action)
            # print("obs: ", obs)
            img = env.render(mode='human')
            
            # regulate control frequency
            clock.tick(control_hz)

        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(len(episode))
            print(f'saved seed {seed}')
            
        else:
            print(f'retry seed {seed}')

if __name__ == "__main__":
    main()
