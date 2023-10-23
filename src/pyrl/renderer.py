# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:27:10 2023

@author: fperotto
"""

###################################################################

from typing import Callable
import pygame as pg

###################################################################

class Renderer():

    #--------------------------------------------------------------    
    def __init__(self, env=None, agent=None):

        self.env = env       #reference to the environment
        self.agent = agent       #reference to the environment
        
        self.ready = (self.env is not None)

    #--------------------------------------------------------------    
    def reset(self, env=None, agent=None):

        if env is not None:
           self.env = env       #reference to the environment
           
        if agent is not None:
           self.agent = agent       #reference to the environment
        
        self.ready = (self.env is not None)

    #--------------------------------------------------------------    
    def render(self):
        pass
     
      
    #--------------------------------------------------------------    
    def refresh(self):
        pass
                
    #--------------------------------------------------------------    
    def close(self):
        pass


###################################################################

class PyGameRenderer(Renderer):

    #--------------------------------------------------------------    
    def __init__(self, env=None, agent=None, 
                 height=400, width=400,
                 on_close_listeners:Callable=None):

        super().__init__(env, agent) 
        
        self.height = height
        self.width = width
        
        self.on_close_listeners = on_close_listeners
        
        self._is_closing = False
        
        pg.init()
        pg.display.init()

        self.window = pg.display.set_mode( (self.width, self.height) )
        

    #--------------------------------------------------------------    
    def reset(self):
       
        self._is_closing = False


    #--------------------------------------------------------------    
    def render(self):

        #refresh
        pg.display.update()

    #--------------------------------------------------------------    
    def process_events(self):

        keys = [] 

        events = pg.event.get()
        
        for event in events:

            if event.type == pg.QUIT:
                self._is_closing = True
      
            if event.type == pg.KEYDOWN:
                keys += event.key
                
        if len(keys) > 0:
           self.on_keydown(keys)

    #--------------------------------------------------------------    
    def on_keydown(self, keys):
       pass
        
      
    #--------------------------------------------------------------    
    def refresh(self):

        if self._is_closing:
            
            if self.on_close_listeners is not None:
               self.on_close_listeners()

            self.close()

        else:

            self.process_events()

            self.render()
                   
            
    #--------------------------------------------------------------    
    def close(self) -> None:
       
        if self.window is not None:
            pg.display.quit()
            pg.quit()

         
    #--------------------------------------------------------------    
    def add_close_listener(self, listener:Callable):
        self.on_close_listeners.append(listener)
        
        
        
###################################################################

