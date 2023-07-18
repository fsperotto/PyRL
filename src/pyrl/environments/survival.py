from typing import Tuple, List, Union

import numpy as np
import gymnasium as gym
import pygame as pg

from pyrl import Env

class SurvivalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str=None, 
                 size: Union[int, Tuple[int, int]]=20, 
                 minor_r=5, major_r=20, step_r=-1, 
                 random=False) -> None:
        
        self.size = np.array(size)
        if not isinstance(size, tuple):
            self.size = np.array([size, size])
            
        self.num_cols = self.size[0]
        self.num_rows = self.size[1]
        
        self.minor_r = minor_r
        self.major_r = major_r
        self.step_r = step_r
        
        self.window_size = 900
        
        self.observation_space = gym.spaces.Discrete(self.num_rows * self.num_cols)

        self.action_space = gym.spaces.Discrete(4)
        
        self._action_to_direction = {
            0: np.array([1, 0]),  #left
            1: np.array([0, 1]),  #up ?
            2: np.array([-1, 0]), #right
            3: np.array([0, -1]), #down ?
        }

        self.agent = None   #reference to the active agent
        
        self.t = 0
        self.truncated = False

        if self.num_rows == self.num_cols:
            self._target_locations = dict(
                major       = np.array([self.num_cols - 2, self.num_rows - 2]), # highest reward
                minor_diag  = np.array([2*(self.num_cols - 1) // 3, 2 * (self.num_rows - 1) // 3]),
                minor_left  = np.array([(self.num_cols - 1) // 2, (self.num_rows - 1) // 4]),
                minor_right = np.array([(self.num_cols - 1) // 4, (self.num_rows - 1) // 2]),
                #minor_nearest = np.array([(self.size[0] - 1) // 8, (self.size[1] - 1) // 8])
                #minor_half = np.array([(self.size[0] - 1) // 3, (self.size[1] - 1) // 3]),
            )
        else:
            self._target_locations = dict(
                major      = np.array([self.num_cols - 2, self.num_rows // 2]), # highest reward
                minor_far  = np.array([3*(self.num_cols - 1) // 5, self.num_rows // 2]),
                minor_near = np.array([(self.num_cols - 1) // 3, self.num_rows // 2]),
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
        return int(self._agent_location[0] * self.num_rows + self._agent_location[1])

    def _get_info(self) -> dict:
        return dict()

    def reset(self, seed=None, options=None) -> Tuple[int, dict]:

        super().reset(seed=seed)

        self.t = 0
        self.truncated = False

        if self.num_rows == self.num_cols:
            self._agent_location = np.array([0, 0])
        else:
            self._agent_location = np.array([0, self.num_cols // 2])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

   
    def step(self, action):
        done = False
        self.t = self.t + 1
        old_agent_location = self._get_obs()
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, [0, 0], [self.num_cols - 1 , self.num_rows - 1]
        )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        reward = self.step_r if self.reward_params is None else np.random.normal(
            self.reward_params[old_agent_location, action, 0],
            self.reward_params[old_agent_location, action, 1]
        )
        for label, target_location in self._target_locations.items():
            if np.array_equal(self._agent_location, target_location):
                if label == "major":
                    reward = self.major_r
                else:
                    reward = self.minor_r

        return observation, reward, False, self.truncated, info

    def render(self):
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                self.truncated = True
                self.clock.tick(0)
                self.close()
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pg.init()
            pg.display.init()
            self.window = pg.display.set_mode( (self.window_size, self.window_size) )
            self.font = pg.font.SysFont(None, 24)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        #clear canvas
        canvas = pg.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        #define square size
        pix_square_size = self.window_size / self.size[0]
        
        board_height = self.num_rows * pix_square_size
        board_width = self.num_cols * pix_square_size

        #draw the rewards
        for label, target_location in self._target_locations.items():
            pg.draw.rect(
                canvas,
                (255, 0 if label == "major" else 165, 0),
                pg.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        #draw the agent
        agent_color = (0, 0, 255)
        if (self.agent is not None) and hasattr(self.agent, "recharge_mode") and self.agent.recharge_mode:
            agent_color = (255, 0, 0)
        pg.draw.circle(
            canvas,
            agent_color,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        #draw horizontal lines
        for y in range(self.size[1] + 1):
            pg.draw.line(
                canvas,
                (0, 0, 0),
                (0, pix_square_size * y),
                (board_width, pix_square_size * y),
                width=3,
            )

        #draw vertical lines
        for y in range(self.size[0] + 1):
            pg.draw.line(
                canvas,
                (0, 0, 0),
                (pix_square_size * y, 0),
                (pix_square_size * y, board_height),
                width=3,
            )

            

        if self.render_mode == "human":
            
            events = pg.event.get()
            for event in events:
                if event.type == pg.QUIT:
                    self.truncated = True
                    
            if self.truncated:
                self.clock.tick(0)
                self.close()
                
            else:
                
                #canvas
                self.window.blit(canvas, canvas.get_rect())
                
                #time
                img_time =   self.font.render(str(self.t), True, (0,0,0))
                self.window.blit(img_time, (20, 200))
                
                #budget
                if (self.agent is not None) and hasattr(self.agent, "budget"):
                    budget_color = (200, 0, 0) if self.agent.budget < 0 else (0, 200, 0)
                    img_budget = self.font.render(str(self.agent.budget), True, budget_color)
                    self.window.blit(img_budget, (20, 230))
                else:
                    budget_color = (200, 0, 0)
                    img_budget = self.font.render('None', True, budget_color)
                    self.window.blit(img_budget, (20, 230))

                #refresh
                pg.display.update()
                self.clock.tick(self.metadata["render_fps"])
        
        else:
            
            return np.transpose(
                np.array(pg.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        
    def close(self) -> None:
        self.truncated = True
        if self.window is not None:
            pg.display.quit()
            pg.quit()

    def get_target_states(self) -> List[Tuple[int, int]]:
        return dict((label, self.location_to_state(location)) for label, location in self._target_locations.items())

    def location_to_state(self, location):
        return location[0] * self.num_rows + location[1]