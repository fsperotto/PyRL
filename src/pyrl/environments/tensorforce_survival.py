from tensorforce.environments import Environment
import numpy as np
import gymnasium as gym
import pygame as pg
from typing import Tuple, List, Union


class CustomEnvironment(Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 80}
    
    def __init__(self, render_mode: str=None, size: Union[int, Tuple[int, int]]=20, terminate=False, random=False):
        super().__init__()
        
        self.size = np.array(size)
        if not isinstance(size, tuple):
            self.size = np.array([size, size])
        
        self.window_size = 500

        self.observation_space = gym.spaces.Discrete(self.size[0] * self.size[1])

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.t = 0

        if self.size[0] == self.size[1]:
            self._target_locations = dict(
                major       = np.array([self.size[0] - 2, self.size[1] - 2]), # highest reward
                minor_diag  = np.array([2*(self.size[0] - 1) // 3, 2 * (self.size[1] - 1) // 3]),
                minor_left  = np.array([(self.size[0] - 1) // 2, (self.size[1] - 1) // 4]),
                minor_right = np.array([(self.size[0] - 1) // 4, (self.size[1] - 1) // 2]),
                #minor_nearest = np.array([(self.size[0] - 1) // 8, (self.size[1] - 1) // 8])
                #minor_half = np.array([(self.size[0] - 1) // 3, (self.size[1] - 1) // 3]),
            )
        else:
            self._target_locations = dict(
                major       = np.array([self.size[0] - 2, self.size[1] // 2]), # highest reward
                minor_far  = np.array([2*(self.size[0] - 1) // 3, self.size[1] // 2]),
                minor_near = np.array([(self.size[0] - 1) // 3, self.size[1] // 2]),
            )


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if random:
            esperance = np.random.randint(-2, 0, size=(self.observation_space.n, self.action_space.n, 1))
            variance = 2 * np.random.rand(self.observation_space.n, self.action_space.n, 1)

            self.reward_params = np.concatenate((esperance, variance), axis=2)
        else:
            self.reward_params = None
        
        self.terminate = terminate
        
        self.window = None
        self.clock = None
        
        self.num_cols = self.size[0]
        self.num_rows = self.size[1]

    def states(self):
        return dict(type='int', shape=(), num_values=self.observation_space.n)

    def actions(self):
        return dict(type='int', num_values=self.action_space.n)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()
    
    def _get_obs(self) -> int:
        return self._agent_location[0] * self.size[1] + self._agent_location[1]
        # return self._agent_location

    def _get_info(self) -> dict:
        return dict()

    def reset(self):
        
        self.recharge_mode = False
        self.t = 0

        if self.size[0] == self.size[1]:
            self._agent_location = np.array([0, 0])
        else:
            self._agent_location = np.array([0, self.size[1] // 2])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation
    
    def execute(self, actions):
        done = False
        self.t = self.t + 1
        old_agent_location_array = self._agent_location
        old_agent_location = self._get_obs()
        direction = self._action_to_direction[actions]
        self._agent_location = np.clip(
            self._agent_location + direction, [0, 0], [self.size[0] - 1 , self.size[1] - 1]
        )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        reward = -1 if self.reward_params is None else np.random.normal(
            self.reward_params[old_agent_location, actions, 0],
            self.reward_params[old_agent_location, actions, 1]
        )
        
        for label, target_location in self._target_locations.items():
            if np.array_equal(self._agent_location, target_location):
                if label == "major":
                    reward = 50
                    
                else:
                    reward = 5
                    

        if self.terminate and np.array_equal(self._agent_location, self._target_locations[0]):
            done = True
         
        # return observation, reward, done, False, info
        return observation, done, reward
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pg.init()
            pg.display.init()
            self.window = pg.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        canvas = pg.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size[0]
        )

        for label, target_location in self._target_locations.items():
            pg.draw.rect(
                canvas,
                (255, 0 if label == "major" else 165, 0),
                pg.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        pg.draw.circle(
            canvas,
            (255, 0, 0) if self.recharge_mode else (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for x in range(self.size[1] + 1):
            pg.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.size[0] * pix_square_size, pix_square_size * x),
                width=3,
            )

        for x in range(self.size[0] + 1):
            pg.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.size[1] * pix_square_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()
            pg.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pg.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self) -> None:
        super().close()
        if self.window is not None:
            pg.display.quit()
            pg.quit()
    
    def get_target_states(self) -> List[Tuple[int, int]]:
        return dict((label, self.location_to_state(location)) for label, location in self._target_locations.items())

    def location_to_state(self, location):
        return location[0] * self.size[0] + location[1]
        
 