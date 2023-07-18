from typing import Tuple, List, Union, Callable

import numpy as np
import gymnasium as gym
import pygame as pg

#from gymnasium import error, spaces, utils
#from gymnasium.utils import seeding

from pyrl import Env

#class CustomEnv(gym.Env):
#    metadata = {'render.modes': ['human']}
#    def __init__(self):
#        ...
#    def step(self, action):
#        ...
#    def reset(self):
#        ...
#    def render(self, mode='human'):
#        ...
#    def close(self):
#        ...

    
class GridEnv(Env):

    metadata = {"render_modes": ["human", "rgb_array", "external"], "render_fps": 60}

    def __init__(self, render_mode: str=None, 
                 size: Union[None, int, Tuple[int, int]]=None,
                 num_rows=None, num_cols=None,
                 reward_matrix=None,
                 reward_targets=None,
                 default_reward:float=0.0,
                 reward_mode="s'",  # "sas'" , "as'" , "sa", "a" , "s'",
                 random=False) -> None:
        
        #define grid size from parameters
        if (num_rows is not None or num_cols is not None):
            if size is not None:
                print('Warning: GridEnv - ignoring size parameter, since num_rows or num_cols are given.')
            if num_rows is not None:
                self.num_rows = num_rows
                if num_cols is not None:
                    self.num_cols = num_cols
                else:
                    self.num_cols = num_rows
            else:
                self.num_cols = num_cols
                self.num_rows = num_cols
            self.size = np.array( (self.num_cols, self.num_rows) )
        else:
            if size is not None:
                if not isinstance(size, tuple) and not isinstance(size, list):
                    self.size = np.array([size, size])
                else:
                    self.size = np.array(size)
            else:
                self.size=np.array([20,20])
            self.num_cols = self.size[0]
            self.num_rows = self.size[1]
        
        state_space = gym.spaces.MultiDiscrete((self.num_rows, self.num_cols))

        self.num_actions = 4
        action_space = gym.spaces.Discrete(self.num_actions)

        self._action_to_direction = {
            0: np.array([1, 0]),  #right
            1: np.array([0, 1]),  #down
            2: np.array([-1, 0]), #left
            3: np.array([0, -1]), #up
        }

        self.reward_mode = reward_mode
        
        self.default_reward = default_reward

        if reward_matrix is not None:
            self.reward_matrix = np.array(reward_matrix)
            self.reward_targets = None
            if reward_targets is not None:
                print("WARNING: GridEnv cannot receive reward matrix and reward targets")
        else:
            if reward_targets is not None:
                self.reward_matrix = np.full((self.num_cols, self.num_rows), self.default_reward)
                self.reward_targets = reward_targets
                for r, pos_list in reward_targets.items():
                    for x, y in pos_list:
                        self.reward_matrix[x, y] = r
            else:
                self.reward_matrix = 2 * np.random.sample((self.num_cols, self.num_rows)) - 1
                self.reward_targets = None

        self._render_frame = None

        self._agent_location = np.array([0, 0])
        
        super().__init__(state_space=state_space, action_space=action_space, render_mode=render_mode)
        

    def _get_obs(self):
        #return int(self._agent_location[0] * self.num_rows + self._agent_location[1])
        return self._agent_location

    def _get_info(self) -> dict:
        return dict()


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        if self.num_rows == self.num_cols:
            self._agent_location = np.array([0, 0])
        else:
            self._agent_location = np.array([0, self.num_rows // 2])

        observation = self._get_obs()
        info = self._get_info()

        #self.render()
        if self.render_mode is not None:
            if self._render_frame is not None:
               self._render_frame()

        return observation, info


   
    def step(self, action):

        super().step(action)

        #old_agent_location = self._get_obs()
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, [0, 0], [self.num_cols - 1 , self.num_rows - 1]
        )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode is not None:
            self._render_frame()

        reward = self.reward_matrix.item(tuple(self._agent_location))

        return observation, reward, False, self.truncated, info

    def render(self):
        if self.render_mode is not None:
            if self._render_frame is not None:
               return self._render_frame()
        
    def close(self) -> None:
        #self.truncated = True
        pass

#    def get_target_states(self) -> List[Tuple[int, int]]:
#        #return dict((label, self.location_to_state(location)) for label, location in self._target_locations.items())
#        return self._target_locations
#
#    def location_to_state(self, location):
#        return location[0] * self.num_rows + location[1]

#########################################################################################################################
    
class GridEnvRender():

    def __init__(self, env, agent=None, fps=40, height=None, width=None, cell_size=None,
                 interruption_callback:Callable=None):

        self.env = env       #reference to the environment
        self.agent = agent   #reference to the active agent
        
        if cell_size is None:
            if height is None: 
                if width is None:
                    self.height = 600
                    self.cell_size = self.height // self.env.num_rows
                    self.width = self.env.num_cols * self.cell_size
                else:
                    self.width = width
                    self.cell_size = self.width // self.env.num_cols
                    self.height = self.env.num_rows * self.cell_size
            else: 
                if width is None:
                    self.height = height
                    self.cell_size = self.height // self.env.num_rows
                    self.width = self.env.num_cols * self.cell_size
                else:
                    self.width = width
                    self.cell_size = self.width // self.env.num_cols
                    self.height = self.env.num_rows * self.cell_size
        else:    
            self.cell_size = cell_size
            self.height = self.env.num_rows * self.cell_size
            self.width = self.env.num_cols * self.cell_size
        
        self.board_height = self.env.num_rows * self.cell_size + 10
        self.board_width = self.env.num_cols * self.cell_size
        
        self.height = 3 * self.board_height + 50
        
        self.fps = fps
        
        self.window = None
        self.clock = None
        
        pg.init()
        pg.display.init()
        self.window = pg.display.set_mode( (self.width, self.height) )
        self.clock = pg.time.Clock()
        
        self.font_size = int(self.cell_size * 0.8)
        self.font = pg.font.SysFont(None, self.font_size)
        
        self.max_r = self.env.reward_matrix.max()
        self.min_r = self.env.reward_matrix.min()
        self.delta_r = self.max_r - self.min_r
        
        #define colors for reward matrix
        self.color_matrix = np.zeros((self.env.num_cols, self.env.num_rows, 3), dtype=int)
        for x in range(self.env.num_cols):
            for y in range(self.env.num_rows):
                reward = self.env.reward_matrix[x, y]
                if reward == 0:
                    r = g = b = 200
                elif reward < 0:
                    r = 255
                    g = int(255 * (1.0 - (reward / self.min_r)) / 1.5)
                    b = g
                else:
                    g = 255
                    r = int(255 * (1.0 - (reward / self.max_r)) / 1.5)
                    b = r
                self.color_matrix[x, y] = [r, g, b]

#    def reset(self):
#        self._render_frame()

#    def step(self, action):
#        self._render_frame()

#    def render(self):
#        events = pg.event.get()
#        for event in events:
#            if event.type == pg.QUIT:
#                self.env.truncated = True
#                self.clock.tick(0)
#                self.close()
#        #if self.render_mode == "rgb_array":
#        #    return self._render_frame()

    def refresh(self):

        if self.env.interrupted:
            
            self.clock.tick(0)
            self.close()

        else:

            events = pg.event.get()
            for event in events:
               if event.type == pg.QUIT:
                   self.env.interrupted = True
           
            #clear canvas
            canvas = pg.Surface((self.width, self.height))
            canvas.fill((255, 255, 255))


            #draw the rewards
            for x in range(self.env.num_cols):
                for y in range(self.env.num_rows):
                    pg.draw.rect(
                        canvas,
                        self.color_matrix[x, y],
                        pg.Rect(
                            self.cell_size * np.array([x, y]),
                            (self.cell_size, self.cell_size),
                        ),
                    )
            
            #draw the agent
            agent_color = (0, 0, 255)
            if (self.agent is not None) and hasattr(self.agent, "recharge_mode") and self.agent.recharge_mode:
                agent_color = (255, 255, 0)
            pg.draw.circle(
                canvas,
                agent_color,
                (self.env._agent_location + 0.5) * self.cell_size,
                self.cell_size / 3,
            )

            #draw horizontal lines
            for y in range(self.env.num_rows + 1):
                pg.draw.line(
                    canvas,
                    (0, 0, 0),
                    (0, self.cell_size * y),
                    (self.board_width, self.cell_size * y),
                    width=3,
                )

            #draw vertical lines
            for x in range(self.env.num_cols + 1):
                pg.draw.line(
                    canvas,
                    (0, 0, 0),
                    (self.cell_size * x, 0),
                    (self.cell_size * x, self.board_height),
                    width=3,
                )

            #draw 1/(N+1) Matrix
            if hasattr(self.agent, 'N') and self.agent.N is not None:
               for x in range(self.env.num_cols):
                   for y in range(self.env.num_rows):
                       n_min = 255 - 255//(self.agent.N[x, y].min()+1) 
                       n_max = 255 - 255//(self.agent.N[x, y].max()+1) 
                       pg.draw.rect(
                           canvas,
                           (n_min, n_min, n_min),
                           pg.Rect(
                               (x * self.cell_size, y * self.cell_size + self.board_height),
                               (self.cell_size, self.cell_size),
                           )
                       )
                       pg.draw.circle(
                           canvas,
                           (n_max, n_max, n_max),
                           ((x+0.5) * self.cell_size, (y+0.5) * self.cell_size + self.board_height),
                           self.cell_size / 4,
                       )
                       
            
            #draw horizontal lines
            for y in range(self.env.num_rows + 1):
                pg.draw.line(
                    canvas,
                    (100, 100, 100),
                    (0, self.cell_size * y + self.board_height),
                    (self.board_width, self.cell_size * y + self.board_height),
                    width=3,
                )

            #draw vertical lines
            for x in range(self.env.num_cols + 1):
                pg.draw.line(
                    canvas,
                    (100, 100, 100),
                    (self.cell_size * x, self.board_height),
                    (self.cell_size * x, 2 * self.board_height),
                    width=3,
                )

            #draw Q Matrix
            if hasattr(self.agent, 'Q') and self.agent.Q is not None:
               max_q = self.agent.Q.max()
               min_q = self.agent.Q.min()
               if max_q > min_q:
                  for x in range(self.env.num_cols):
                      for y in range(self.env.num_rows):
                          q = self.agent.Q[x, y].max()
                          c = int(255 * (q - min_q) / (max_q - min_q))
                          pg.draw.rect(
                              canvas,
                              (c, c, c),
                              pg.Rect(
                                  (x * self.cell_size, y * self.cell_size + 2*self.board_height),
                                  (self.cell_size, self.cell_size),
                              )
                          )
                       
            
            #draw horizontal lines
            for y in range(self.env.num_rows + 1):
                pg.draw.line(
                    canvas,
                    (100, 100, 100),
                    (0, self.cell_size * y + 2 * self.board_height),
                    (self.board_width, self.cell_size * y + 2 * self.board_height),
                    width=3,
                )

            #draw vertical lines
            for x in range(self.env.num_cols + 1):
                pg.draw.line(
                    canvas,
                    (100, 100, 100),
                    (self.cell_size * x, 2 * self.board_height),
                    (self.cell_size * x, 3 * self.board_height),
                    width=3,
                )

            #budget bar
            if (self.agent is not None) and hasattr(self.agent, "b") and self.agent.b is not None:
                pg.draw.rect(
                   canvas,
                   (50, 50, 200),
                   pg.Rect(
                       (0, 3*self.board_height+30),
                       ( int(self.width * (self.agent.b / 1000)), self.cell_size),
                   )
                )

            #canvas
            self.window.blit(canvas, canvas.get_rect())

            #time
            img_time =   self.font.render(str(self.env.t), True, (0,0,0))
            self.window.blit(img_time, (20, 3*self.board_height+10))

            #budget
            if (self.agent is not None) and hasattr(self.agent, "b") and self.agent.b is not None:
                budget_color = (200, 0, 0) if self.agent.b < 0 else (0, 200, 0)
                img_budget = self.font.render(str(self.agent.b), True, budget_color)
                self.window.blit(img_budget, (20, 3*self.board_height+30))
            else:
                budget_color = (200, 0, 0)
                img_budget = self.font.render('None', True, budget_color)
                self.window.blit(img_budget, (20, 3*self.board_height+30))

            #refresh
            pg.display.update()
            self.clock.tick(self.fps)
                
        
    def close(self) -> None:
        if self.window is not None:
            pg.display.quit()
            pg.quit()
