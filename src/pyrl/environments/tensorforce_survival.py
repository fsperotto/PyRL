from tensorforce.environments import Environment
import numpy as np
import gymnasium as gym
import pygame as pg


class CustomEnvironment(Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: str=None, size: int=20, terminate=False):
        super().__init__()
        self.size = size
        self.window_size = 512

        self.observation_space = gym.spaces.Discrete(self.size * self.size)

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.t = 0

        self._target_locations = np.array([
            [self.size - 2, self.size - 2], # highest reward
            [2*(self.size - 1) // 3, 2 * (self.size - 1) // 3],
            [(self.size - 1) // 4, (self.size - 1) // 2],
            [(self.size - 1) // 2, (self.size - 1) // 4],
        ])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.terminate = terminate
        
        self.window = None
        self.clock = None
        

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
        return self._agent_location[0] * self.size + self._agent_location[1]

    def _get_info(self) -> dict:
        return dict()

    def reset(self):
        
        self.t = 0

        self._agent_location = np.array([0, 0])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    

    def execute(self, actions):
        done = False
        self.t = self.t + 1
        direction = self._action_to_direction[actions]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        reward = -1
        for index, target_location in enumerate(self._target_locations):
            if np.array_equal(self._agent_location, target_location):
                if index == 0:
                    reward = 100
                else:
                    reward = 10

        if self.terminate and np.array_equal(self._agent_location, self._target_locations[0]):
            done = True
         
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
            self.window_size / self.size
        )

        for index, target_location in enumerate(self._target_locations):
            pg.draw.rect(
                canvas,
                (255, 0 if index == 0 else 165, 0),
                pg.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        pg.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for x in range(self.size + 1):
            pg.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pg.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
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
        
        
        
        