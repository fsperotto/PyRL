# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:17:58 2023

@author: fperotto
"""

###################################################################

from typing import Iterable, Callable
import pygame as pg

###################################################################

class GUI():

    #--------------------------------------------------------------    
    def __init__(self, sim):

        self.sim = sim       #reference to the simulation
        
    #--------------------------------------------------------------    
    def reset(self):
        pass

    #--------------------------------------------------------------    
    def launch(self):
        pass

    #--------------------------------------------------------------    
    def refresh(self):
        pass
                
    #--------------------------------------------------------------    
    def close(self):
        pass


###################################################################

class PyGameGUI(GUI):

    #--------------------------------------------------------------    
    def __init__(self, sim,
                 height=400, width=400,
                 fps=80,
                 batch_run=10000,
                 on_close_listeners:Iterable[Callable]=[],
                 close_on_finish=True, finish_on_close=False):

        super().__init__(sim) 
        
        self.height = height
        self.width = width
        
        self.fps = fps
        self.refresh_interval_ms = max(10, 1000 // self.fps)
        
        self.batch_run = batch_run
        
        self.close_on_finish = close_on_finish
        self.finish_on_close = finish_on_close
        
        self.on_close_listeners = on_close_listeners
        
        self._is_closing = False
        
        pg.init()

        self.CLOCKEVENT = pg.USEREVENT+1
        #self.clock = pg.time.Clock()
        
        self.window = None
        
        #self.sim.add_listener('round_finished', self.on_clock)
        
    #--------------------------------------------------------------    
    def reset(self):
       
        self._is_closing = False

    #--------------------------------------------------------------    
    def set_timer_state(self, state:bool):
       
        if state == True:
           pg.time.set_timer(self.CLOCKEVENT, self.refresh_interval_ms)
        else:
           pg.time.set_timer(self.CLOCKEVENT, 0)

    #--------------------------------------------------------------    
    def launch(self, give_first_step=True, start_running=True):

        pg.display.init()
        self.window = pg.display.set_mode( (self.width, self.height) )
        #pg.display.set_caption('Exp')        
        #self.window.set_caption('Exp')
        
        if give_first_step:
           self.sim.step()
           self.refresh()
        
        if start_running:
           self.set_timer_state(True)
        
        #RUNNING
        try:

           while not self._is_closing:
              
              event = pg.event.wait()
              
              self.process_event(event)
              
              if self.close_on_finish and self.sim.finished:
                 self.close()

        except KeyboardInterrupt:
           self.close()
           print("KeyboardInterrupt: simulation interrupted by the user.")

        except:
           self.close()
           raise

     
    #--------------------------------------------------------------    
    def refresh(self):
        
        #refresh
        pg.display.update()
                
    #--------------------------------------------------------------    
    def close(self):
       
        self._is_closing = True
 
        if self.window is not None:

            #CLOSING
            for callback in self.on_close_listeners:
               callback(self)
               
            pg.display.quit()
            pg.quit()
           
        if self.finish_on_close:
           self.sim.run()

        if self.sim.env is not None:
           self.sim.env.close()

    #--------------------------------------------------------------    
    def process_event(self, event):
       
         if (event.type == pg.QUIT):
             self.close()
   
         elif event.type == pg.KEYDOWN:
             self.on_keydown(event.key)
             
         elif event.type == self.CLOCKEVENT:
             self.sim.step()
             self.refresh()
      
    #--------------------------------------------------------------    
    def on_keydown(self, key):
       
       #ESC = exit
       if key == pg.K_ESCAPE:
          self.close()
          
       #P = pause
       elif key == pg.K_p:
          self.set_timer_state(False)
          
       #R = run
       elif key == pg.K_r:
          self.set_timer_state(True)
          
       #S = step
       elif key == pg.K_s:
          self.sim.step()
          self.refresh()
            
       #B = batch run
       elif key == pg.K_b:
          self.sim.run(self.batch_run)
          self.refresh()

       #E = episode run
       elif key == pg.K_e:
          self.sim.run('episode')
          self.refresh()

       #Q = simulation run
       elif key == pg.K_q:
          self.sim.run('simulation')
          self.refresh()

       #Z = repetition run
       elif key == pg.K_z:
          self.sim.run('repetition')
          self.refresh()
          
    #--------------------------------------------------------------    
    #def on_clock(self, *args, **kwargs):
    #   self.refresh()
    #   self.clock.tick(self.fps)

    #--------------------------------------------------------------    
    #def step(self):
    #   self.sim.next_step()

    #--------------------------------------------------------------    
    # def process_events(self):

    #     keys = [] 

    #     events = pg.event.get()
        
    #     for event in events:

    #         if event.type == pg.QUIT:
    #             self._is_closing = True
      
    #         if event.type == pg.KEYDOWN:
    #             keys = keys + [event.key]
                
    #     if len(keys) > 0:
    #        self.on_keydown(keys)

    #--------------------------------------------------------------    
    # def refresh(self):

    #     self.process_events()
       
    #     self.render()
                   
            
