from __future__ import print_function
import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from state_diff.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class PushLEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            object_scale=30,
            default_rendering=True
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy
        
        self.object_scale = object_scale

        # agent_pos, block_pos, block_angle, U_object_pos, U_object_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2,ws,ws,np.pi*2], dtype=np.float64),
            shape=(8,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action1 = None
        self.latest_action2 = None
        self.reset_to_state = reset_to_state

        self.draw_keypoint = False

        # Mode 0: Single Mouse
        # Mode 1: Space Mouse & Mouse
        # Mode 2: Space Mouse & Space Mouse

        self.done = False
        
        self.default_rendering = default_rendering



    def select_draw_keypoint(self, draw_keypoint):
        self.draw_keypoint = draw_keypoint


    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.L0_object.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        state = self.reset_to_state

        # Failure recovery from edges cases
        if seed <= 50:
            if seed % 4 == 0:
                limits = [45, 60, 45, 470]
                angles = 0
            elif seed % 4 == 1:
                limits = [455, 470, 45, 470]
                angles = np.pi/2
            elif seed % 4 == 2:
                limits = [55, 470, 55, 60]
                angles = np.pi
            elif seed % 4 == 3:
                limits = [45, 470, 455, 470]
                angles = -np.pi/2
            # print("Failure recovery from edges cases")
        # Normal/regular cases
        else:
            limits = [100, 400, 100, 400]
            rs = np.random.RandomState(seed=seed)
            angles = rs.randn() * 2 * np.pi - np.pi
            # print("Normal/regular cases")
        if seed == 19:
            limits = [60, 450, 430, 450]
            angles = -np.pi/2
        if state is None:
            # use legacy RandomState for compatibility
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(limits[0], limits[1]), rs.randint(limits[2], limits[3]),
                angles, # rs.randn() * 2 * np.pi - np.pi,
                rs.randint(limits[0], limits[1]), rs.randint(limits[2], limits[3]),
                angles # rs.randn() * 2 * np.pi - np.pi,
                ])
            # print("State=", state)

        self._set_state(state)
        observation = self._get_obs()
        # print("self.L0_object.position=", self.L0_object.position)
        # print("self.L1_object.position=", self.L1_object.position)
        return observation
    

    def step(self, action):
        action1 = None
        if not action is None:
            action1 = action
            self.simulate_agent(self.agent1, self.sim_hz, self.control_hz, self.k_p, self.k_v, action1)

        reward = self.compute_reward()
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, self.done, info
    
    def compute_reward(self):
        # compute reward
        L0_goal_body = self._get_goal_pose_body(self.L0_goal_pose)
        L0_goal_geom = pymunk_to_shapely(L0_goal_body, self.L0_object.shapes)

        L1_goal_body = self._get_goal_pose_body(self.L1_goal_pose)
        L1_goal_geom = pymunk_to_shapely(L1_goal_body, self.L1_object.shapes)

        L0_object_geom = pymunk_to_shapely(self.L0_object, self.L0_object.shapes)
        L1_object_geom = pymunk_to_shapely(self.L1_object, self.L1_object.shapes)

        L0_object_on_L0_goal_area = L0_goal_geom.intersection(L0_object_geom).area
        L1_object_on_L1_goal_area = L1_goal_geom.intersection(L1_object_geom).area
        L1_object_on_L0_goal_area = L0_goal_geom.intersection(L1_object_geom).area
        L0_object_on_L1_goal_area = L1_goal_geom.intersection(L0_object_geom).area

        self.done = (L0_object_on_L0_goal_area + L1_object_on_L1_goal_area + L1_object_on_L0_goal_area + L0_object_on_L1_goal_area) > self.success_threshold*(L0_goal_geom.area + L1_goal_geom.area)
        
        L0_object_on_L0_goal = L0_goal_geom.intersection(L0_object_geom).area / L0_goal_geom.area
        L1_object_on_L1_goal = L1_goal_geom.intersection(L1_object_geom).area / L1_goal_geom.area
        L1_object_on_L0_goal = L0_goal_geom.intersection(L1_object_geom).area / L0_goal_geom.area
        L0_object_on_L1_goal = L1_goal_geom.intersection(L0_object_geom).area / L1_goal_geom.area

        total_coverage = (L0_object_on_L0_goal + L1_object_on_L1_goal + L1_object_on_L0_goal + L0_object_on_L1_goal)

        reward = np.clip(total_coverage / (2 * self.success_threshold), 0, 1)
        return reward



    def simulate_agent(self, agent, sim_hz, control_hz, k_p, k_v, action):
        dt = 1.0 / sim_hz
        self.n_contact_points = 0
        n_steps = sim_hz // control_hz

        if action is not None:
            for _ in range(n_steps):
                acceleration = k_p * (action - agent.position) + k_v * (Vec2d(0, 0) - agent.velocity)
                agent.velocity += acceleration * dt
                self.space.step(dt)

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent1.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)
    





    def _get_obs(self):
        obs = np.array(
            tuple(self.agent1.position) \
            + tuple(self.L0_object.position) \
            + tuple(self.L1_object.position) \
            + (self.L0_object.angle % (2 * np.pi),)
            + (self.L1_object.angle % (2 * np.pi),))
        return obs
    
    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent1': np.array(self.agent1.position),
            'vel_agent1': np.array(self.agent1.velocity),
            'L0_object_pose': np.array(list(self.L0_object.position) + [self.L0_object.angle]),
            'L1_object_pose': np.array(list(self.L1_object.position) + [self.L1_object.angle]),
            'L0_goal_pose': self.L0_goal_pose,
            'L1_goal_pose': self.L1_goal_pose,
            'n_contacts': n_contact_points_per_step,
            'success': self.done}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)
        # if not self.default_rendering:
        #     draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES  # Disable edge drawing

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.L0_goal_pose)
        for shape in self.L0_object.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        goal_body = self._get_goal_pose_body(self.L1_goal_pose)
        for shape in self.L1_object.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)
            
        # Draw agent and block.
        self.space.debug_draw(draw_options)
        
        if not self.default_rendering:
            # Manually draw the L0 and L1 objects to disable edges
            for shape in self.L0_object.shapes:
                vertices = shape.get_vertices()
                points = [pymunk.pygame_util.to_pygame(self.L0_object.local_to_world(v), canvas) for v in vertices]
                pygame.draw.polygon(canvas, shape.color, points, 0)  # Fill only, no edges

            for shape in self.L1_object.shapes:
                vertices = shape.get_vertices()
                points = [pymunk.pygame_util.to_pygame(self.L1_object.local_to_world(v), canvas) for v in vertices]
                pygame.draw.polygon(canvas, shape.color, points, 0)  # Fill only, no edges
                
            # Manually draw the circles (agents) without edges
            for agent in [self.agent1, getattr(self, 'agent2', None)]:
                if agent:
                    for shape in agent.shapes:
                        if isinstance(shape, pymunk.Circle):
                            pos = pymunk.pygame_util.to_pygame(agent.local_to_world(shape.offset), canvas)
                            pygame.draw.circle(canvas, shape.color, pos, int(shape.radius), 0)  # Fill only, no edges

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action1 is not None):
                action1 = np.array(self.latest_action1)
                coord1 = (action1 / 512 * 96).astype(np.int32)
                marker_size1 = int(8/96*self.render_size)
                thickness1 = int(1/96*self.render_size)
                cv2.drawMarker(img, coord1,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size1, thickness=thickness1)
                
            if self.render_action and (self.latest_action2 is not None):
                action2 = np.array(self.latest_action2)
                coord2 = (action2 / 512 * 96).astype(np.int32)
                marker_size2 = int(8/96*self.render_size)
                thickness2 = int(1/96*self.render_size)
                cv2.drawMarker(img, coord2,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size2, thickness=thickness2)
        
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent1 = state[:2]
        pos_L0 = state[2:4]
        rot_L0 = state[4]
        pos_L1 = state[5:7]
        rot_L1 = state[7]
        self.agent1.position = pos_agent1
        
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.L0_object.position = pos_L0
            self.L0_object.angle = rot_L0
            self.L1_object.position = pos_L1
            self.L1_object.angle = rot_L1
        else:
            self.L0_object.angle = rot_L0
            self.L0_object.position = pos_L0
            self.L1_object.angle = rot_L1
            self.L1_object.position = pos_L1

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        
        if self.agentmode == 0:
            if self.default_rendering:
                self.agent1 = self.add_circle((250, 270), 15, -1)
            else:
                self.agent1 = self.add_circle((250, 250), 15, 0, color=(77, 77, 77))
            self.agent = self.agent1
        else:
            self.agent1 = self.add_circle((270, 250), 15, 0)
            self.agent2 = self.add_circle((250, 270), 15, 1)
        if self.default_rendering:
            self.L0_object = self.add_L0((256, 300), 0, scale=self.object_scale)
        else:
            self.L0_object = self.add_L0((256, 300), 0, scale=self.object_scale, color=(251, 147, 143)) # large L

        self.goal_color = pygame.Color('LightGreen')
        self.L0_goal_pose = np.array([300,200,np.pi/4])  # x, y, theta (in radians)
        
        if self.default_rendering:
            self.L1_object = self.add_L1((200, 200), 0, scale=self.object_scale)
        else:
            self.L1_object = self.add_L1((200, 200), 0, scale=self.object_scale, color=(253, 187, 117)) # small L

        L1_goal_pos = (self.L0_goal_pose[:2].copy())
        goal_distance = 64
        L1_goal_pos[0] -= goal_distance
        L1_goal_pos[1] += goal_distance
        self.L1_goal_pose = np.array([*L1_goal_pos,np.pi*5/4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.9    # 90% coverage.


    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius, select=0, color=''):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        if color == '':
            if select == -1:
                shape.color = pygame.Color('RoyalBlue')
            elif select == 0:
                shape.color = pygame.Color(242, 97, 34) # orange
            else:
                shape.color = pygame.Color(18, 4, 76) # blue
        else:
            shape.color = pygame.Color(color)
            
        self.space.add(body, shape)
        return body

    def add_L0(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        # Define the vertices for the L shape
        vertices1 = [(-length*scale/2, scale),
                     ( length*scale/2, scale),
                     ( length*scale/2, 0),
                     (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        
        vertices2 = [(-scale, scale),
                     (-scale, length*2/4*scale),
                     ( length*scale/2, length*2/4*scale),
                     ( length*scale/2, scale)]
        
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        
        return body


    def add_L1(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        # Define the vertices for the L shape
        vertices1 = [(-length*scale*2/4, scale),
                     ( length*scale*2/4, scale),
                     ( length*scale*2/4, 0),
                     (-length*scale*2/4, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        
        vertices2 = [(scale, scale),
                     (scale, length*2/4*scale),
                     ( length*scale/2, length*2/4*scale),
                     ( length*scale/2, scale)]
        
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        
        return body
