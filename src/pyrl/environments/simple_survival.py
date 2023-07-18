from typing import Tuple, List

import numpy as np
import gymnasium as gym
import pygame as pg

class SimpleSurvivalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str=None, size: int=20, random: bool=False) -> None:
        self.size = size
        self.window_width = 1000

        self.observation_space = gym.spaces.Discrete(self.size)

        self.action_space = gym.spaces.Discrete(2)

        self._action_to_direction = {
            0: -1,
            1: 1,
        }

        self.t = 0

        self._target_locations = dict(
            major = np.array([self.size - 1]), # highest reward
            minor = np.array([self.size // 2])
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if random:
            esperance = np.random.randint(-2, 0, size=(self.observation_space.n, self.action_space.n, 1))
            variance = 2 * np.random.rand(self.observation_space.n, self.action_space.n, 1)

            self.reward_params = np.concatenate((esperance, variance), axis=2)
        else:
            self.reward_params = None

        self.window = None
        self.clock = None

    def _get_obs(self) -> int:
        return self._agent_location[0]

    def _get_info(self) -> dict:
        return dict()

    def reset(self, seed=None, options=None) -> Tuple[int, dict]:
        super().reset(seed=seed)

        self.t = 0

        self._agent_location = np.array([0])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.t = self.t + 1
        old_agent_location = self._get_obs()
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        reward = -1 if self.reward_params is None else np.random.normal(
            self.reward_params[old_agent_location, action, 0],
            self.reward_params[old_agent_location, action, 1]
        )
        for label, target_location in self._target_locations.items():
            if np.array_equal(self._agent_location, target_location):
                if label == "major":
                    reward = 100
                else:
                    reward = 10

        return observation, reward, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pix_square_size = (
            self.window_width / self.size
        )

        if self.window is None and self.render_mode == "human":
            pg.init()
            pg.display.init()
            self.window = pg.display.set_mode(
                (self.window_width, self.window_width)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        canvas = pg.Surface((self.window_width, pix_square_size))
        canvas.fill((255, 255, 255))

        for label, target_location in self._target_locations.items():
            pg.draw.rect(
                canvas,
                (255, 0 if label == "major" else 165, 0),
                pg.Rect(
                    (pix_square_size * target_location[0], 0),
                    (pix_square_size, pix_square_size),
                ),
            )

        pg.draw.circle(
            canvas,
            (0, 0, 255),
            ((self._agent_location[0] + 0.5) * pix_square_size, pix_square_size / 2),
            pix_square_size / 3,
        )

        for x in range(self.size + 1):
            pg.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size),
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
        if self.window is not None:
            pg.display.quit()
            pg.quit()

    def get_target_states(self) -> List[Tuple[int, int]]:
        return dict((label, self.location_to_state(location)) for label, location in self._target_locations.items())

    def location_to_state(self, location):
        return location[0]