from typing import Iterable, Callable, TypeVar, Generic, Tuple, List, Union

import math

import numpy as np
import gymnasium as gym
import pygame as pg


#from gymnasium import error, spaces, utils
#from gymnasium.utils import seeding

from pyrl import Env, PyGameRenderer, PyGameGUI

from pyrl.space import ensure_tuple

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

    metadata = {"render_modes": ["human", "rgb_array", "external"], "render_fps": 90}

    action_str = np.array([
            'right',  # 0
            'down',   # 1 
            'left',   # 2
            'up'])    # 3
         
         
    action_idx = {
            'right':0,
            'down':1,
            'left':2,
            'up':3
         }

    action_to_direction = np.array([
            [+1, 0],  #0: right
            [0, +1],  #1: down
            [-1, 0],  #2:left
            [0, -1]]) #3:up


    def __init__(self, render_mode: str=None, 
                 size: Union[None, int, Tuple[int, int]]=None,
                 num_rows=None, num_cols=None,
                 default_reward:float=0.0,
                 reward_variance:float=0.0,
                 reward_matrix=None,
                 reward_spots=None, reward_spread:float=0.0,
                 reward_mode="s'",  # "sas'" , "as'" , "sa", "a" , "s'",
                 initial_position=None,
                 default_initial_budget=None,
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
        
        state_space = gym.spaces.MultiDiscrete((self.num_cols, self.num_rows))

        self.num_actions = 4
        action_space = gym.spaces.Discrete(self.num_actions)

        self.reward_mode = reward_mode
        
        self.reward_variance = reward_variance
        self.reward_std_dev = math.sqrt(reward_variance)
        
        #rewards informed via matrix
        if reward_matrix is not None:
            self.reward_matrix = np.array(reward_matrix, dtype=float) + float(default_reward)
            #assert self.reward_matrix.shape == (self.num_cols, self.num_rows)
        
        #rewards not informed via matrix
        else:
           #nor via spots
           if reward_spots is None:
              #create randomized reward matrix with mean equivalent to default reward and variance to variance
              #self.reward_matrix = (2 * np.random.sample((self.num_cols, self.num_rows)) - 1) + float(default_reward) 
              self.reward_matrix = np.random.normal( loc = default_reward, scale = self.reward_std_dev, size = (self.num_cols, self.num_rows))
           #rewards will be informed via spots
           else:
              #create initial matrix full of default reward
              self.reward_matrix = np.full((self.num_cols, self.num_rows), float(default_reward))
           
        #rewards informed via spots
        if reward_spots is not None:
          #without reward spreading
          if reward_spread == 0.0:
             #each spot increments its value only at the exact position
             for (cx,cy), r in reward_spots.items():
                self.reward_matrix[cx,cy] += r
          #with reward spreading
          else:
             #each target is a center of reward, spreading the value around
             for (cx, cy), r in reward_spots.items():
                for x in range(self.num_cols):  
                   for y in range(self.num_rows):
                      #d = math.dist((cx, cy), (x, y))
                      d = abs(cx-x) + abs(cy-y)
                      self.reward_matrix[x, y] += r * (reward_spread ** d)
                   
        self._render_frame = None

        if initial_position is None:
           initial_position = [0,0]
        self._agent_location = np.array(initial_position)
        
        super().__init__(state_space, action_space, render_mode=render_mode, default_initial_budget=default_initial_budget)
        

    def _get_obs(self):
        #return int(self._agent_location[0] * self.num_rows + self._agent_location[1])
        return self._agent_location

    def _get_info(self) -> dict:
        return dict()

    def get_reward_matrix(self, reward_mode=None):  #"s'", "a" (MAB), "sa", "sas'", "ass'", "as'" (?) 
       if reward_mode is None  or reward_mode == self.reward_mode:
          R = self.reward_matrix
       else:
          print("TO DO: convert reward modes")
          R = None
       return R
       
    def print_reward_matrix(self, reward_mode=None):  #"s'", "a" (MAB), "sa", "sas'", as'" (?) 
       R = self.get_reward_matrix(reward_mode=reward_mode)
       if R is not None:
          #for x in range(self.num_cols):
          #   for y in range(self.num_rows):
          #      print(f"s'=(x,y)=({x},{y}), r={R[x,y]}")
          for y in range(self.num_rows):
             for x in range(self.num_cols):
                print(round(R[x,y],3), end=' ')
             print()
    
    def get_transition_matrix(self):
       P = np.array( [[[self._next_position(a, [x, y]) for a in range(4)] for y in range(self.num_rows)] for x in range(self.num_cols) ])
       return P


    def reset(self, seed=None, options=None):

        #observation, initial_budget, info = super().reset(seed=seed)
        observation, info = super().reset(seed=seed)

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

        #return observation, initial_budget, info
        return observation, info


    def _next_position(self, action, location=None):
        if location is None:
           location = self._agent_location
        direction = self.action_to_direction[action]
        return np.clip(location + direction, [0, 0], [self.num_cols-1 , self.num_rows-1])
   
    
    def step(self, action):

        super().step(action)

        self._agent_location = self._next_position(action)
        observation = self._get_obs()
        info = self._get_info()

        #if self.render_mode is not None:
        #    self._render_frame()

        if self.reward_variance is None or self.reward_variance == 0.0:
           reward = self.reward_matrix.item(tuple(self._agent_location))
        else:
           reward = np.random.normal(self.reward_matrix.item(tuple(self._agent_location)), self.reward_std_dev)
                    
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
    
class GridEnvGUI(PyGameGUI):

    #--------------------------------------------------------------    
    def __init__(self, sim, 
                 height=400, width=400,
                 cell_size=None,
                 fps=80,
                 batch_run=10000,
                 grid_elements=[
                    #{'pos':0, 'label':'agent', 'data':[{'source':'env', 'attr':'R', 'type':"s'"}]},
                    {'pos':1, 'label':'N_sa', 'source':'agent', 'attr':'N_sa', 'type':'sa', 'color_mode':'inversed_log_grayscale', 'backcolor':None},
                    #{'pos':2, 'label':'V', 'source':'agent', 'attr':'V', 'type':'s', 'color_mode':'grayscale', 'backcolor':None},
                    {'pos':2, 'label':'Q', 'source':'agent', 'attr':'Q', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
                    {'pos':3, 'label':'K', 'source':'agent', 'attr':'K', 'type':'sa', 'color_mode':'grayscale', 'backcolor':None},
                    {'pos':4, 'label':'policy', 'source':'agent', 'attr':'policy', 'type':'sa', 'color_mode':'grayscale', 'backcolor':(0,0,0)},
                 ],
                 on_close_listeners:Iterable[Callable]=[],
                 close_on_finish=True, finish_on_close=False):

        super().__init__(sim,
                         height=height, width=width,
                         fps=fps, batch_run=batch_run,
                         on_close_listeners=on_close_listeners,
                         close_on_finish=close_on_finish,
                         finish_on_close=finish_on_close)

        self.cell_size = cell_size
        
        self.grid_elements = grid_elements


    #--------------------------------------------------------------    
    def launch(self, give_first_step=True, start_running=True):
       
        if self.cell_size is None:
            self.cell_size = self.height // self.sim.env.num_rows
            self.width = self.sim.env.num_cols * self.cell_size
        else:    
            # self.cell_size = cell_size
            self.height = self.sim.env.num_rows * self.cell_size
            self.width = self.sim.env.num_cols * self.cell_size
        
        self.margin_size = 10 
        self.board_height = self.sim.env.num_rows * self.cell_size
        self.board_width = self.sim.env.num_cols * self.cell_size
        
        self.height = (len(self.grid_elements)+1) * (self.board_height + self.margin_size) + 3 * self.cell_size

        self.font_size = int(self.cell_size * 0.8)
        self.font = pg.font.SysFont(None, self.font_size)
        
        self.max_r = self.sim.env.reward_matrix.max()
        self.min_r = self.sim.env.reward_matrix.min()
        self.delta_r = self.max_r - self.min_r
        
        #define colors for reward matrix
        self.reward_color_matrix = np.zeros((self.sim.env.num_cols, self.sim.env.num_rows, 3), dtype=int)
        for x in range(self.sim.env.num_cols):
            for y in range(self.sim.env.num_rows):
                reward = self.sim.env.reward_matrix[x, y]
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
                self.reward_color_matrix[x, y] = [r, g, b]
                
        super().launch(give_first_step=give_first_step, start_running=start_running)
       
    #--------------------------------------------------------------    
    def _draw_grid(self, canvas, vertical_skip=None, vertical_position=0, line_color=(100, 100, 100)):

         if vertical_skip is None:
            vertical_skip = vertical_position * (self.board_height + self.margin_size)
         
         line_color = (0, 0, 0)
         #draw horizontal lines
         for y in range(self.sim.env.num_rows + 1):
             pg.draw.line(
                 canvas,
                 line_color,
                 (0, self.cell_size * y + vertical_skip),
                 (self.board_width, self.cell_size * y + vertical_skip),
                 width=3,
             )

         #draw vertical lines
         for x in range(self.sim.env.num_cols + 1):
             pg.draw.line( 
                 canvas,
                 line_color,
                 (self.cell_size * x, vertical_skip),
                 (self.cell_size * x, vertical_skip + self.board_height),
                 width=3,
             )
       
    #--------------------------------------------------------------    
    def _draw_rewards(self, canvas, vertical_position=0):

         vertical_skip = vertical_position * (self.board_height + self.margin_size)
       
         #draw the rewards
         for x in range(self.sim.env.num_cols):
             for y in range(self.sim.env.num_rows):
                 pg.draw.rect(
                     canvas,
                     self.reward_color_matrix[x, y],
                     pg.Rect(
                         (self.cell_size * x, self.cell_size * y + vertical_skip),
                         (self.cell_size, self.cell_size),
                     ),
                 )
                 
         self._draw_grid(canvas, vertical_skip=vertical_skip)

    #--------------------------------------------------------------    
    def _draw_agent(self, canvas, vertical_skip=None, vertical_position=0, agent_color=(0, 0, 255)):

         if vertical_skip is None:
            vertical_skip = vertical_position * (self.board_height + self.margin_size)
       
         if hasattr(self.sim.agent, "recharge_mode") and self.sim.agent.recharge_mode:
             agent_color = (255, 255, 0)
             
         pg.draw.circle(
             canvas,
             agent_color,
             (self.sim.env._agent_location + 0.5 + [0, vertical_skip]) * self.cell_size,
             self.cell_size / 3,
         )

    #--------------------------------------------------------------    
    def _draw_agent_position(self, canvas, vertical_skip=None, vertical_position=0, color=(255, 255, 50)):

         if vertical_skip is None:
            vertical_skip = vertical_position * (self.board_height + self.margin_size)
       
         pg.draw.rect(
             canvas,
             color,
             pg.Rect(
                 self.sim.env._agent_location[0] * self.cell_size, self.sim.env._agent_location[1] * self.cell_size + vertical_skip,
                 self.cell_size, self.cell_size
             ),
             width=3
         )

      #--------------------------------------------------------------    
    def _color(self, v, min_v, amplitude_v, color_mode='grayscale'):

         if color_mode == 'reward':

            if v == 0:
                r = g = b = 200
            elif v < 0:
                r = 255
                g = int(255 * (1.0 - (v / min_v)) / 1.5)
                b = g
            else:
                g = 255
                r = int(255 * (1.0 - (v / (min_v + amplitude_v))) / 1.5)
                b = r

            color = (r, g, b)
            
         else:
            
            c = int(255 * (v - min_v) / amplitude_v)
            
            if color_mode == 'inversed_grayscale':
               c = 255-c
            elif color_mode == 'log_grayscale':
               c = 255//(v+1)
            elif color_mode == 'inversed_log_grayscale':
               c = 255 - 255//(v+1)

            color = (c, c, c)
               
         return color

    #--------------------------------------------------------------    
    def _draw_s_matrix(self, canvas, matrix, min_value=None, max_value=None, vertical_skip=None, vertical_position=0, color_mode='grayscale', backcolor=None):

       if vertical_skip is None:
          vertical_skip = vertical_position * (self.board_height + self.margin_size)
       
       if max_value is None: 
          max_value = matrix.max()
       
       if min_value is None: 
          min_value = matrix.min()
       
       dif = max_value - min_value
       
       #draw the matrix
       for x in range(self.sim.env.num_cols):
          for y in range(self.sim.env.num_rows):
             
             value = matrix[x, y]

             if backcolor is None:
                color = self._color(value, min_value, dif, color_mode=color_mode)
             else:
                color = backcolor
                
             points = pg.Rect(
                       (x * self.cell_size, y * self.cell_size + vertical_skip),
                       (self.cell_size, self.cell_size),
                    )
             
             pg.draw.rect(canvas, color, points)
                 
       self._draw_grid(canvas, vertical_skip=vertical_skip)

    #--------------------------------------------------------------    
    def _draw_sa_matrix(self, canvas, matrix, min_q=None, max_q=None, vertical_skip=None, vertical_position=0, color_mode='grayscale', backcolor=None):
         
       if vertical_skip is None:
          vertical_skip = vertical_position * (self.board_height + self.margin_size)
       
       if max_q is None:
          max_q = matrix.max()
       
       if min_q is None:
          min_q = matrix.min()
       
       dif_q = max_q - min_q
       
       if max_q > min_q:
      
          for x in range(self.sim.env.num_cols):
             for y in range(self.sim.env.num_rows):
                   
                q = matrix[x, y].max()
                if backcolor is None:
                   color = self._color(q, min_q, dif_q, color_mode=color_mode)
                else:
                   color = backcolor
                points = pg.Rect(
                           (x * self.cell_size, y * self.cell_size + vertical_skip),
                           (self.cell_size, self.cell_size),
                       )
                pg.draw.rect(canvas, color, points)
                
                #0: np.array([1, 0]),  #right
                q = matrix[x, y, 0]
                color = self._color(q, min_q, dif_q, color_mode=color_mode)
                right_triangle_points = [
                     ((x+1) * self.cell_size, y * self.cell_size + self.cell_size//2 + vertical_skip), 
                     (x * self.cell_size + 2*self.cell_size//3, y * self.cell_size + self.cell_size//3 + vertical_skip), 
                     (x * self.cell_size + 2*self.cell_size//3, y * self.cell_size + 2*self.cell_size//3 + vertical_skip)
                  ]
                pg.draw.polygon(canvas, color, right_triangle_points)
                pg.draw.polygon(canvas, (0, 0, 0), right_triangle_points, width=1)


                #1: np.array([0, 1]),  #down
                q = matrix[x, y, 1]
                color = self._color(q, min_q, dif_q, color_mode=color_mode)
                down_triangle_points = [
                       (x * self.cell_size + self.cell_size//2, (y+1) * self.cell_size + vertical_skip), 
                       (x * self.cell_size + self.cell_size//3,   y * self.cell_size + 2*self.cell_size//3 + vertical_skip), 
                       (x * self.cell_size + 2*self.cell_size//3, y * self.cell_size + 2*self.cell_size//3 + vertical_skip)
                     ]
                pg.draw.polygon(canvas, color, down_triangle_points)
                pg.draw.polygon(canvas, (0, 0, 0), down_triangle_points, width=1)

                #2: np.array([-1, 0]), #left
                q = matrix[x, y, 2]
                color = self._color(q, min_q, dif_q, color_mode=color_mode)
                left_triangle_points = [
                       (x * self.cell_size, y * self.cell_size + self.cell_size//2 + vertical_skip), 
                       (x * self.cell_size + self.cell_size//3, y * self.cell_size + self.cell_size//3 + vertical_skip), 
                       (x * self.cell_size + self.cell_size//3, y * self.cell_size + 2*self.cell_size//3 + vertical_skip)
                     ]
                pg.draw.polygon(canvas, color, left_triangle_points)
                pg.draw.polygon(canvas, (0, 0, 0), left_triangle_points, width=1)
                   
                #3: np.array([0, -1]), #up                
                q = matrix[x, y, 3]
                color = self._color(q, min_q, dif_q, color_mode=color_mode)
                up_triangle_points = [
                       (x * self.cell_size + self.cell_size//2, y * self.cell_size + vertical_skip), 
                       (x * self.cell_size + self.cell_size//3,   y * self.cell_size + self.cell_size//3 + vertical_skip), 
                       (x * self.cell_size + 2*self.cell_size//3, y * self.cell_size + self.cell_size//3 + vertical_skip)
                     ]
                pg.draw.polygon(canvas, color, up_triangle_points)
                pg.draw.polygon(canvas, (0, 0, 0), up_triangle_points, width=1)
                
       #draw grid lines
       self._draw_grid(canvas, vertical_skip=vertical_skip)
       self._draw_agent_position(canvas, vertical_skip=vertical_skip)
       
       
    #--------------------------------------------------------------    
    def _draw_bar(self, canvas, v, max_v, vertical_skip=None, vertical_position=0, bar_position=0, color=(50, 50, 200) ):

       if vertical_skip is None:
          vertical_skip = vertical_position * (self.board_height + self.margin_size) + bar_position * self.cell_size

       points = pg.Rect(
                    (0, vertical_skip),
                    (int(self.width * (v / max_v)), self.cell_size),
                  )
       
       pg.draw.rect(canvas, color, points)

    #--------------------------------------------------------------    
    def _draw_bar_label(self, label, vertical_skip=None, vertical_position=0, bar_position=0, color=(0, 0, 0) ):

       if vertical_skip is None:
          vertical_skip = vertical_position * (self.board_height + self.margin_size) + bar_position * self.cell_size

       img =   self.font.render(str(label), True, color)
       self.window.blit(img, (20, vertical_skip+10))

    #--------------------------------------------------------------    
    def refresh(self):

         #clear canvas
         canvas = pg.Surface((self.width, self.height))
         canvas.fill((255, 255, 255))

         #draw the rewards
         self._draw_rewards(canvas, vertical_position=0)
         
         #draw the agent
         self._draw_agent(canvas, vertical_position=0)

         for grid_element in self.grid_elements:
            if grid_element['source'] == 'agent':
               source = self.sim.agent
            elif grid_element['source'] == 'env':
               source = self.sim.env
            else:
               source = self.sim
            if hasattr(source, grid_element['attr']):
               attr = getattr(source, grid_element['attr'])
               if attr is not None:
                  if grid_element['type'] == "sa":
                     self._draw_sa_matrix(canvas, matrix=attr, vertical_position=grid_element['pos'], color_mode=grid_element['color_mode'], backcolor=grid_element['backcolor'])
                  else:
                     self._draw_s_matrix(canvas, matrix=attr, vertical_position=grid_element['pos'], color_mode=grid_element['color_mode'], backcolor=grid_element['backcolor'])
            
         # #draw 1/(N+1) Matrix
         # #self._draw_exploration(canvas, vertical_position=1)
         # if hasattr(self.sim.agent, 'N') and self.sim.agent.N is not None:
         #    self._draw_sa_matrix(canvas, matrix=self.sim.agent.N, vertical_position=1, min_q=0, color_mode='inversed_log_grayscale')
                    
         # #draw Q Matrix
         # if hasattr(self.sim.agent, 'Q') and self.sim.agent.Q is not None:
         #    self._draw_sa_matrix(canvas, matrix=self.sim.agent.Q, vertical_position=2)

         # #draw K Matrix
         # if hasattr(self.sim.agent, 'K') and self.sim.agent.K is not None:
         #    self._draw_sa_matrix(canvas, matrix=self.sim.agent.K, vertical_position=3)

         # #draw Policy
         # if hasattr(self.sim.agent, 'policy') and self.sim.agent.policy is not None:
         #    self._draw_sa_matrix(canvas, matrix=self.sim.agent.policy, vertical_position=4, min_q=0, backcolor=(0,0,0))

         #budget bar
         if hasattr(self.sim.agent, "b") and self.sim.agent.b is not None:
            self._draw_bar(canvas, v=self.sim.agent.b, max_v=1000, vertical_position=5, bar_position=0)

         #time bar
         self._draw_bar(canvas, v=self.sim.t, max_v=self.sim.episode_horizon, vertical_position=5, bar_position=1, color=(0,150,0))
         
         #canvas
         self.window.blit(canvas, canvas.get_rect())

         #name label
         self._draw_bar_label(str(self.sim.agent.name), vertical_position=5, bar_position=2)

         #time label
         self._draw_bar_label("t = " + str(self.sim.env.t), vertical_position=5, bar_position=1)

         #budget label
         if hasattr(self.sim.agent, "b") and self.sim.agent.b is not None:
             #budget_color = (200, 0, 0) if self.sim.agent.b < 0 else (0, 200, 0)
             self._draw_bar_label("b = " + str(self.sim.agent.b), vertical_position=5, bar_position=0)

         #refresh
         #pg.display.update()
         
         ###self.clock.tick(self.fps)
         
         super().refresh()


###############################################################################

#UNIT TESTS
if __name__ == "__main__":

    print("\nGRID MODULE - UNIT TESTS\n")
    
    g = GridEnv(num_rows=2, num_cols=3)
    
    g.print_reward_matrix()

    P = g.get_transition_matrix()
    
    for x in range(g.num_cols):
       for y in range(g.num_rows):
          for a in range(4):
             print(f"(x,y)=({x},{y}),a={a}({g.action_str[a]}) {P[x,y,a]}")
    
    #reward_targets = {+7 : [(2, 1)],
    #                  +3 : [(4, 1)]}
    reward_spots = { (2, 1):+7, 
                     (4, 1):+3 }
    
    env = g
    
    #reward in the form "factored" + "s'"
    R = env.get_reward_matrix()
    print(R.shape)
    #reward in the form "flat" + "s'"
    R = R.reshape(env.observation_comb)
    print(R.shape)
    #reward in the form "flat" + "ass'"
    R = np.tile(R, (env.action_comb, env.observation_comb, 1))
    print(R.shape)

    #transition in the form "sas'" + "factored" + "deterministic"
    P = env.get_transition_matrix()
    print(P.shape)
    #transition in the form "sas'" + "flat(sa)/factored(s')" + "deterministic"
    P = P.reshape( (env.observation_comb, env.action_comb, 2) )
    print(P.shape)
    #transition in the form "ass'" + "flat(as)/factored(s')" + "deterministic"
    P = np.swapaxes(P,0,1)
    print(P.shape)
    #transition in the form "ass'" + "flat" + "deterministic"
    #P= np.multiply(P, [env.num_cols,1])
    P= np.multiply(P, [1,env.num_rows])
    print(P.shape)
    print(P)
    P = np.sum(P,axis=2)
    print(P.shape)
    print(P)
    #transition in the form "ass'" + "flat" + "stochastic"
    #P = np.expand_dims(P, axis=-1)
    #print(P.shape)
    #P = np.repeat(P, env.observation_comb, axis=-1)
    #print(P.shape)
    XP = np.zeros( (env.action_comb, env.observation_comb, env.observation_comb) , dtype=float)
    print(XP.shape)
    for act in range(env.action_comb):
       for obs in range(env.observation_comb):
          next_obs = P[act, obs]
          XP[act, obs, next_obs] = 1.0

    print(XP)


    g = GridEnv(num_rows=3, num_cols=5, reward_spots=reward_spots, default_reward=-1)

    g.print_reward_matrix()

    P = g.get_transition_matrix()

    for x in range(g.num_cols):
       for y in range(g.num_rows):
          for a in range(4):
             print(f"(x,y)=({x},{y}),a={a}({g.action_str[a]}) {P[x,y,a]}")

    g = GridEnv(num_rows=3, num_cols=5, reward_spots=reward_spots, reward_spread=0.5, default_reward=-1.)

    g.print_reward_matrix()

    P = g.get_transition_matrix()
