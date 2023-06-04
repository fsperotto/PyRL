import math
from math import sin, cos, atan, sqrt
import numpy as np



#rotate a point over another
def rotate(x, y, ang_rad=0.0, ox=0, oy=0):
    return [cos(ang_rad) * (x-ox) - sin(ang_rad) * (y-oy) + ox,
            sin(ang_rad) * (x-ox) + cos(ang_rad) * (y-oy) + oy]


class MovingBoxProblem:

    #constructor
    def __init__(self, body_height=40, body_width=80, 
                 art_sizes=[60, 40], art_num_states=[9, 19], art_angle_limits=[[-30,+30], [-150,0]],
                 accept_actions=[[-1,0],[+1,0],[0,-1],[0,+1]],
                 using_degrees=True):
        
        #articulation properties
        if not isinstance(art_sizes, list):
            art_sizes = [art_sizes]
        if not isinstance(art_num_states, list):
            art_num_states = [art_num_states]
        if not isinstance(art_angle_limits, list):
            art_angle_limits = [art_angle_limits]
        
        self.num_articulations = len(art_sizes)
        
        self.art_sizes = art_sizes
        self.art_num_states = art_num_states
        if using_degrees:
            self.art_angle_limits = [[math.radians(deg) for deg in art] for art in art_angle_limits]
        else:
            self.art_angle_limits = art_angle_limits

        #flat states from the combined articulation states
        self.num_flat_states = math.prod(art_num_states)
            
        #list of acceptable combined movements
        self.accept_actions = accept_actions
        if accept_actions is not None:
            self.num_flat_actions = len(accept_actions)
        else:
            self.num_flat_actions = 3 ** self.num_articulations
        
        self.art_angle_step = [((angle_max-angle_min) / (num_states-1)) for (angle_min, angle_max), num_states in zip(self.art_angle_limits, self.art_num_states)]
        
        self.body_height = body_height
        self.body_width = body_width
   
        self.reset()


    #reset running properties
    def reset(self):
        
        self.t = 0
        
        #action 0 is nop, +1 is up, -1 is down
        if self.accept_actions is not None:
            self.art_action = self.accept_actions[0].copy()
        else:
            self.art_action = [0] * self.num_articulations
        
        #self.art_cur_state = [(num_states // 2) for num_states in self.art_num_states]
        self.art_cur_state = [0] * self.num_articulations

        self.art_cur_angle = [0.0] * self.num_articulations
        
        self.art_coords = [None] * self.num_articulations
        
        self.pos_x = 0.0
        
        self.hand_pos_x = 0.0
        self.hand_pos_y = 0.0
        
        self.last_reward = 0.0
        
        self.update()
    
        
#//----------------------------------------------
    
    #update running properties
    def update(self, action=None):

        #update with a given action
        if action is not None:
            #action passed in articulated form (list of articulation movements)
            if isinstance(action, list):
                self.art_action = action
            #action passed as an index from the list of possible movements
            else:
                self.art_action = self.to_art_action(action)
            #move the arm
            for i in range(self.num_articulations):
                self.art_cur_state[i] = min(self.art_num_states[i]-1, max(0, self.art_cur_state[i] + self.art_action[i]))
    
        #define the angles corresponding to the new articulations states
        self.calc_angles(in_place=True)
        
        #position of the arm extremity
        hand_prev_x = self.hand_pos_x
        hand_prev_y = self.hand_pos_y
        self.hand_pos_x = sum(self.art_sizes[i] * math.cos(self.art_cur_angle[i]) for i in range(self.num_articulations)) + self.body_width
        self.hand_pos_y = sum(self.art_sizes[i] * math.sin(self.art_cur_angle[i]) for i in range(self.num_articulations)) + self.body_height

        #verify if it is elevating the box
        if (self.hand_pos_y < 0):
            self.body_angle = math.atan(-self.hand_pos_y/self.hand_pos_x)
        else:
            self.body_angle = 0.0
            
        #verify if the movement of the arm causes a box displacement
        d = 0.0
        if (self.hand_pos_y < 0):
            if (hand_prev_y < 0):
                d = sqrt(hand_prev_x**2 + hand_prev_y**2) - sqrt(self.hand_pos_x**2 + self.hand_pos_y**2)
            else:
                d = (hand_prev_x-hand_prev_y*(self.hand_pos_x-hand_prev_x)/(self.hand_pos_y-hand_prev_y)) - sqrt(self.hand_pos_x**2 + self.hand_pos_y**2)
        else:
            if (hand_prev_y < 0):
                d = -(self.hand_pos_x-self.hand_pos_y*(hand_prev_x-self.hand_pos_x)/(hand_prev_y-self.hand_pos_y)) + sqrt(hand_prev_x**2 + hand_prev_y**2)
            else:
                d = 0
        
        #the reward corresponds to the displacement
        self.last_reward = d
        
        self.pos_x += d
        
        self.avg_speed = self.pos_x / self.t  if  self.t > 0  else  0
        
        x = self.pos_x
        y = 0
        w = self.body_width
        h = self.body_height
        
        p_0 = [x, y]
        p_1 = rotate(x+w, y, self.body_angle, x, y)
        p_2 = rotate(x+w, y+h, self.body_angle, x, y)
        p_3 = rotate(x, y+h, self.body_angle, x, y)
        
        self.body_coords = p_0 + p_1 + p_2 + p_3

        p_base = [None, None] + p_2
        
        for i in range(self.num_articulations):
            self.art_coords[i] = [p_base[2],p_base[3],
                                  p_base[2]+self.art_sizes[i]*math.cos(self.art_cur_angle[i] + self.body_angle), 
                                  p_base[3]+self.art_sizes[i]*math.sin(self.art_cur_angle[i] + self.body_angle)]
            p_base = self.art_coords[i]
        
        #increment time
        if action is not None:
            self.t += 1
        
#//----------------------------------------------    

    def cur_flat_state(self):
        return self.to_flat_state(self.art_cur_state)
    
    def to_flat_state(self, art_state):
        v = 0
        m = 1
        for i in range(self.num_articulations-1, -1, -1):
            v += m * (art_state[i])
            m *= self.art_num_states[i]
        return v

    def to_art_state(self, v):
        art_state = [0] * self.num_articulations
        for i in range(self.num_articulations-1, -1, -1):
            art_state[i] = v % self.art_num_states[i]
            v = v // self.art_num_states[i]
        return art_state
    
    def cur_flat_action(self):
        if self.accept_actions is not None:
            return self.accept_actions.index(self.art_action)
        else:
            return self.to_flat_action(self.art_action)
    
    def to_flat_action(self, art_action):
        if self.accept_actions is not None:
            return self.accept_actions.index(art_action)
        else:
            v = 0
            m = 1
            for i in range(self.num_articulations-1, -1, -1):
                v += m * (art_action[i]+1)
                m *= 3
            return v
    
    def to_art_action(self, v):
        if self.accept_actions is not None:
            return self.accept_actions[v]
        else:
            art_state = [0] * self.num_articulations
            for i in range(self.num_articulations-1, -1, -1):
                art_state[i] = (v % 3) - 1
                v = v // 3
            return art_state

    def next_flat_state(self, state=None, action=None):
        if state is not None:
            state = self.to_art_state(state)
        if action is not None:
            action = self.to_art_action(action)
        return self.to_flat_state(self.next_art_state(art_state=state, art_action=action))
    
    def next_art_state(self, art_state=None, art_action=None):
        if art_state is None:
            art_state = self.art_cur_state
        if art_action is None:
            art_action = self.art_action
        #project arm movement
        art_next_state = art_state.copy()
        for i in range(self.num_articulations):
            art_next_state[i] = min(self.art_num_states[i]-1, max(0, art_state[i] + art_action[i]))
        return art_next_state

    #define the angles corresponding to the articulations state
    def calc_angles(self, art_state=None, in_place=False):
        #if state parameter is not given, take the current state
        if art_state is None:
            art_state = self.art_cur_state
        #if in_place flag is true, change the current angles
        if in_place:
            art_angles = self.art_cur_angle
        else:
            art_angles = self.art_cur_angle.copy()
        #remember the angle of the last articulation calculated
        prev_art_angle = 0.0
        #calculate each articulation angle from the base
        for i in range(self.num_articulations):
            art_angles[i] = art_state[i]*self.art_angle_step[i] + self.art_angle_limits[i][0] + prev_art_angle
            prev_art_angle = art_angles[i]
        #return the calculated angles
        return art_angles
        
    
    def expected_reward_flat(self, state=None, action=None, next_state=None):
        if state is not None:
            state = self.to_art_state(state)
        if action is not None:
            action = self.to_art_action(action)
        if next_state is not None:
            next_state = self.to_art_state(next_state)
        return self.expected_reward_art(art_state=state, art_action=action, art_next_state=next_state)
        

    def expected_reward_art(self, art_state=None, art_action=None, art_next_state=None):
        
        if art_state is None:
            art_state = self.art_cur_state
            art_angles = self.art_cur_angle
        else:
            art_angles = self.calc_angles(art_state=art_state)
        
        if art_action is None:
            art_action = self.art_cur_action
            
        if art_next_state is None:
            art_next_state = self.next_art_state(art_state, art_action)

        art_next_angles = self.calc_angles(art_state=art_next_state)
        
        #position of the arm extremity
        hand_prev_x = sum(self.art_sizes[i] * math.cos(art_angles[i]) for i in range(self.num_articulations)) + self.body_width
        hand_prev_y = sum(self.art_sizes[i] * math.sin(art_angles[i]) for i in range(self.num_articulations)) + self.body_height
        hand_pos_x = sum(self.art_sizes[i] * math.cos(art_next_angles[i]) for i in range(self.num_articulations)) + self.body_width
        hand_pos_y = sum(self.art_sizes[i] * math.sin(art_next_angles[i]) for i in range(self.num_articulations)) + self.body_height

        #verify if the movement of the arm causes a box displacement
        d = 0.0
        if (hand_pos_y < 0):
            if (hand_prev_y < 0):
                d = sqrt(hand_prev_x**2 + hand_prev_y**2) - sqrt(hand_pos_x**2 + hand_pos_y**2)
            else:
                d = (hand_prev_x-hand_prev_y*(hand_pos_x-hand_prev_x)/(hand_pos_y-hand_prev_y)) - sqrt(hand_pos_x**2 + hand_pos_y**2)
        else:
            if (hand_prev_y < 0):
                d = -(hand_pos_x-hand_pos_y*(hand_prev_x-hand_pos_x)/(hand_prev_y-hand_pos_y)) + sqrt(hand_prev_x**2 + hand_prev_y**2)
            else:
                d = 0
        
        #the reward corresponds to the displacement
        return d

    
    def getAvgSpeed(self):
        return (self.dDeplacement / self.t)  if  (self.t > 0)  else  0.0



    def getRewardModel(self):
        return np.array([[self.expected_reward_flat(s, a) for s in range(self.num_flat_states)] for a in range(self.num_flat_actions)])


    def getTransitionModel(self):
        return np.array([[self.next_flat_state(s, a) for s in range(self.num_flat_states)] for a in range(self.num_flat_actions)])


    def setObservationsAndUpdate(self, iObservedState, dReceivedReward):
        #set observations
        self.iCurrentState = iObservedState
        self.dLastReward = dReceivedReward
        self.update()



###############################################################


# imports every file form tkinter and tkinter.ttk
from tkinter import *
from tkinter.ttk import *

class MovingBoxViewer:
    
    def __init__(self, window=None, problem=None, agent=None, width=1000, height=800, show=True):
        
        # moving box model
        self.problem = problem

        self.agent = agent

        # tk window
        if window is not None:
            self.window = window
        else:
            self.window = Tk()
        
        self.window.title('MovingBox')
        
        self.width = width
        self.height = height
        
        self.wscale = 1.0
        self.hscale = 1.0
        
        # canvas object to create shape
        self.canvas = Canvas(window, width=width, height=height)
        
        self.hline = self.canvas.create_line(0, height//2, width, height//2, fill = "lightgrey")
        self.vline = self.canvas.create_line(width//2, 0, width//2, height, fill = "lightgrey")
        self.page_text = self.canvas.create_text(width//2+15, height//2+15, anchor="nw", text=str(0), fill="lightgrey", font=('Helvetica 15 bold'))
        self.state_text = self.canvas.create_text(30, 10, anchor="nw", text="current state: ", fill="black", font=('Helvetica 15 bold'))
        self.action_text = self.canvas.create_text(30, 30, anchor="nw", text="last action: ", fill="black", font=('Helvetica 15 bold'))
        self.dist_text = self.canvas.create_text(30, 50, anchor="nw", text="distance: ", fill="black", font=('Helvetica 15 bold'))
        self.speed_text = self.canvas.create_text(30, 70, anchor="nw", text="average speed: ", fill="black", font=('Helvetica 15 bold'))
        self.time_text = self.canvas.create_text(30, 90, anchor="nw", text="current time: ", fill="black", font=('Helvetica 15 bold'))
        self.state_text_value = self.canvas.create_text(200, 10, anchor="nw", text=str(0), fill="black", font=('Helvetica 15 bold'))
        self.action_text_value = self.canvas.create_text(200, 30, anchor="nw", text=str(0), fill="black", font=('Helvetica 15 bold'))
        self.dist_text_value = self.canvas.create_text(200, 50, anchor="nw", text=str(0), fill="black", font=('Helvetica 15 bold'))
        self.speed_text_value = self.canvas.create_text(200, 70, anchor="nw", text=str(0), fill="black", font=('Helvetica 15 bold'))
        self.time_text_value = self.canvas.create_text(200, 90, anchor="nw", text=str(0), fill="black", font=('Helvetica 15 bold'))
        
        x = self.problem.pos_x + width//2
        y = height//2
        h = self.problem.body_height
        w = self.problem.body_width
        
        # creating box
        self.box = self.canvas.create_polygon(x, y, x+w, y, x+w, y-h, x, y-h, fill = "black")
        x=x+w
        y=y-h
        self.art = []
        for i in range(self.problem.num_articulations):
            self.art += [self.canvas.create_line(x, y, x+self.problem.art_sizes[i], y, fill = "black")]
            x += self.problem.art_sizes[i]

        self.canvas.pack(fill="both", expand=True)
        #self.canvas.pack()
        
        # bind arrow keys to the tkinter
        self.window.bind("<Key>", self.on_keypress)
        
        n = max(0, self.problem.num_articulations-2)
        self.key_to_action = {"Right":[0,1]+[0]*n,"Left":[0,-1]+[0]*n,"Up":[1,0]+[0]*n,"Down":[-1,0]+[0]*n}
        
        self.window.bind("<Configure>", self.on_resize)

        self.playing = False
        
        # Toplevel object which will
        # be treated as a new window
#        newWindow = Toplevel(self.window)

        # sets the title of the
        # Toplevel widget
#        newWindow.title("Learner Parameters")

        # sets the geometry of toplevel
#        newWindow.geometry("200x200")

        # A Label widget to show in toplevel
#        Label(newWindow, text ="This is a new window").pack()        

        if hasattr(agent, 'epsilon'):
            self.eps_slider = Scale(self.window, from_=0.0, to=1.0, orient='horizontal', command=self.on_eps_slider_changed)
            self.eps_slider.set(agent.epsilon)
            self.eps_slider.pack()
        
        if show:
            self.show()
    
    
    #adjust problem coordinates to window coordinates
    def adjust(self, coords):
        coords = coords.copy()
        for i in range(len(coords)):
            if i%2==0:
                coords[i] += self.width//2
                coords[i] += self.translation
                coords[i] *= self.wscale
                #coords[i] %= self.width
            else:
                coords[i] = self.height//2 - coords[i]
                coords[i] *= self.hscale
        return coords
    
    def step(self, action=None, draw=True):
        if action is None:
            action = self.agent.choose()
        self.problem.update(action)
        self.agent.update()
        if draw:
            self.draw()

    def play(self):
        self.step()
        if self.playing:
            self.canvas.after(10, self.play)

    def batch(self):
        for _ in range(100000):
            self.step(draw=False)
        self.draw()
            
    def on_keypress(self, event):
        #print(event.keysym)
        if event.keysym in self.key_to_action:
            self.playing = False
            self.step(self.key_to_action[event.keysym])
        elif event.keysym == 'a':
            self.playing = False
            self.step()
        elif event.keysym == 'p':
            if not self.playing:
                self.playing = True
                self.play()
            else:
                self.playing = False
        elif event.keysym == 'b':
            self.playing = False
            self.window.config(cursor="watch")
            self.batch()
            self.window.config(cursor="arrow")
        elif event.keysym == 'r':
            self.playing = False
            self.problem.reset()
            self.agent.reset()
            self.draw()

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        self.wscale = float(event.width)/self.width
        self.hscale = float(event.height)/self.height
        self.canvas.coords(self.hline, 0, event.height//2, event.width, event.height//2)
        self.canvas.coords(self.vline, event.width//2, 0, event.width//2, event.height)
        self.canvas.coords(self.page_text, event.width//2+5, event.height//2+5)
        self.draw()
        
    def on_eps_slider_changed(self, event):
        self.agent.epsilon = self.eps_slider.get()
        #print(self.agent.epsilon)
            
    def show(self):
        # Infinite loop breaks only by interrupt
        self.window.mainloop()        
    

    def draw(self):
        
        self.page = int((self.problem.pos_x + self.problem.body_width + self.width//2) // self.width)
        self.translation = -self.page * self.width
        
        self.canvas.coords(self.box, self.adjust(self.problem.body_coords))
            
        for i in range(self.problem.num_articulations):
            self.canvas.coords(self.art[i], self.adjust(self.problem.art_coords[i]))

        self.canvas.itemconfig(self.page_text, text=str(self.page * self.width))
        self.canvas.itemconfig(self.state_text_value, text=str(self.problem.art_cur_state))
        self.canvas.itemconfig(self.action_text_value, text=str(self.problem.art_action))
        self.canvas.itemconfig(self.dist_text_value, text=str(round(self.problem.pos_x)))
        self.canvas.itemconfig(self.speed_text_value, text=str(round(self.problem.avg_speed, 2)))
        self.canvas.itemconfig(self.time_text_value, text=str(self.problem.t))

        

