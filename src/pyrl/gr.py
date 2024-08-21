#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gambler's Ruin.

   This module implements GR methods.
"""

__version__ = "0.0.1"
__author__ = "Filipo S. Perotto, Matisse Roche"
__license__ = "MIT"
__status__ = "Development"

################

from typing import List, Union

from math import factorial, comb  #, sqrt

import numpy as np

#from scipy.special import binom   # from python 3.8 is preferable to use math.comb

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import RegularPolygon
from matplotlib.transforms import Affine2D
#from matplotlib.collections import PatchCollection

try:
    #import pandas as pd
    from pandas import DataFrame
    #from pandas.io.formats.style import Styler
    _PANDAS_IMPORTED = True
except:
    _PANDAS_IMPORTED = False

try:
    import networkx as nx
    _NX_IMPORTED = True
except:
    _NX_IMPORTED = False
    
try:
    from IPython.display import display, HTML, Latex
    _IPYTHON_IMPORTED = True
except:
    _IPYTHON_IMPORTED = False

##################################################################
# CLASSIC PASCAL'S TRIANGLE, BINOMIAL COEFFICIEN AND DISTRIBUTION
##################################################################

#pascal triangle element (binomial coefficient) over integer space
def zcomb(n:int, k:int):
    """
    Returns the given element of the Pascal's triangle over integer space

    binomial coefficient (combinations) extended with zeros over the entire Z space
    """
    #k and n must be naturals
    if (n >= k) and (k >= 0):
        return comb(n, k)
    else:
        return 0

pascal_element = zcomb

zbinom = zcomb


################

def pascal_triangle(max_n, recursive_construction=False, return_numpy=True):
    """
    Returns a squared matrix of size m containing a Pascal's Triangle.
    """
    
    #recursive
    if recursive_construction:
        matrix = [[1] + [0] * (max_n) for _ in range(max_n+1)]
        for n in range(1, max_n+1):
            for k in range(1, n+1):
                matrix[n][k] = matrix[n-1][k-1] + matrix[n-1][k]
        
    #binomial coefficient
    else:
        matrix = [[zcomb(n, k) for k in range(max_n+1)] for n in range(max_n+1)]
    
    if return_numpy:
        return np.array(matrix)
    else:
        return matrix

################

#from scipy.stats import binom
#mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
#rv = binom(n, p)
#prob = binom.cdf(x, n, p)
#p = binom.pmf(k, n, p, loc=0)

################

#the binomial distribution extended to Z space
def zprob(n:int, k:int):
    """the binomial distribution extended to Z space
    """
    return zcomb(n, k) / (2**n)
    #return zdist(n, k, p=0.5)

################

#returns the binomial distribution
# in fact, one can use scipy.spec.binom for that
def zdist(n:int, k:int, p:float=0.5):
    """binomial distribution
    """
    return zcomb(n, k) * p**k * (1-p)**(n-k)

################

#the sum of all elements in a same row of the pascal triangle
def ztotal(n:int):
    """the sum of all elements in a same row of the pascal triangle
    """
    return 2**n


###########################################
# TRANSLATED PASCAL'S TRIANGLE
###########################################

#shifted pascal triangle
def zscomb(n:int, k:int, b:int=0) : 
    """shifted pascal triangle
    """
    return zcomb(n, k-b)

zshift = zscomb

shift_pascal_triangle_element = zscomb

#delayed pascal triangle
def zdcomb(n:int, k:int, d:int=0) : 
    """delayed pascal triangle
    """
    return zcomb(n-d, k)

zdelay = zdcomb

delay_pascal_triangle_element = zdcomb

#translated pascal triangle
def zdbcomb(n:int, k:int, d:int=0, b:int=0): 
    """translated pascal triangle
    """
    return zcomb(n-d, k-b)

ztrans = zdbcomb

trans_pascal_triangle_element = zdbcomb


###########################################
# TRUNCATED PASCAL'S TRIANGLE (ONE FOLD)
###########################################

#truncated pascal triangle
def ztcomb(n:int, k:int, h:int=1): 
    """truncated pascal triangle
    """
    return zcomb(n, k) - zcomb(n, k-h)

ztrunc = ztcomb

trunc_pascal_triangle_element = ztcomb


###########################################
# CENTERED TRIANGLE
###########################################

#while the classical Pascal'S Triangle is like (+1,0) rewarded,
#the Centered triangle is equivalent to (+1,-1) rewarded
def ccomb(n:int, s:int):
    """
    Centered Pascal's triangle, equivalent to (+1,-1) rewarded
    """
    dk = n+s
    if dk%2==0: #even
        return zcomb( n, dk//2 )
    else:
        return 0

center_triangle_element = ccomb

#d delaying and centering
def cdcomb(n:int, s:int, d:int=0):
    """
    $d$ delayed and centered Pascal's Triangle

    """
    #return ccomb( n-d, s )
    dk = n-d+s
    if dk%2==0: #even
        return zcomb( n-d, dk//2 )
    else:
        return 0

delay_center_triangle_element = cdcomb


#b and d translation and centering
def cdbcomb(n:int, s:int, d:int=0, b:int=0): 
    """
    Centered Pascal's Triangle, $b$ shifted and $d$ translated
    """
    #return ccomb( n-d, n-d+s-b )
    dk = n-d+s-b
    if dk%2==0: #even
        return zcomb( n-d, dk//2 )
    else:
        return 0

trans_center_triangle_element = cdbcomb


###########################################
# BUDGETED TRIANGLE AND DISTRIBUTION
###########################################

#budgeted triangle is centered and b shifted
def bcomb(n:int, s:int, b:int=0): 
    """
    centering and $b$ shifting
    """
    #return ccomb(n, (n+s-b)//2)
    dk = n+s-b
    if dk%2==0: #even
        return zcomb( n, dk//2 )
    else:
        return 0
    
cbcomb = bcomb

budget_triangle_element = bcomb

shift_center_triangle_element = bcomb


def bdist(n:int, s:int, b:int=0, p:float=0.5):
    """
    The corresponding distribution of a centered and $b$ shifted Pascal's Triangle
    """ 
    #return budget_triangle(n, s, b) * p**k * (1-p)**(n-k)
    #return  zcomb(n, (n+s-b)//2) * p**((n+s-b)//2) * (1-p)**((n-s+b)//2)
    dk = n+s-b
    if dk%2==0: #even
        return zdist(n, dk//2, p)
    else:
        return 0

budget_dist = bdist


###########################################
# TRUNCATED CENTERED TRIANGLES
###########################################

#truncating
def tccomb(n:int, s:int, h:int=-1):
    """
    Truncated Centered Pascal's triangle
    """
    # h is the truncation position
    dk = n+s
    if dk%2==0: #even
        return zcomb(n, dk//2) - zcomb(n, (dk//2)-h)
    else:
        return 0

trunc_center_triangle_element = tccomb

################

def tcbcomb(n:int, s:int, b:int=0, h:int=-1):
    """
    Truncated and budgeted centered Pascal's triangle
    """
    # h is the truncation position
    # b is the budget
    dk = n+s-b 
    if dk%2==0: #even
        k = dk//2
        j = ((n+s+b)//2)-h     # ?????
        return zcomb(n, k) - zcomb(n, j)   #  ?????
        #return ztrunc(n, k, h)
        #return zcomb(n, k) - zcomb(n, k-h)
    else:
        return 0

trunc_budget_triangle_element = tcbcomb

################

def tcdcomb(n:int, s:int, d:int=0, h:int=-1):
    """
    Truncated and Delayed Pascal's triangle
    """
    dk = n-d+s 
    if dk%2==0: #even
        k = dk//2
        #return ztrunc(n, k, h)
        return zcomb(n, k) - zcomb(n, k-h)
    else:
        return 0

trunc_delay_triangle_element = tcdcomb

################

#n and d displacement
def tcdbcomb(n:int, s:int, b:int=0, d:int=0, h:int=-1):
    """
    $n$ and $d$ displaced centered triangle
    """
    dk = n-d+s-b 
    if dk%2==0: #even
        k = dk//2
        j = (n-d+s+b)//2 - h    # ????
        return zcomb(n-d, k) - zcomb(n-d, j)
    else:
        return 0

trunc_trans_triangle_element = tcdbcomb

###########################################
# RUIN TRIANGLE AND DISTRIBUTION
###########################################

# is a +b shifted triangle truncated on and under 0, then completed at 0
def ruin_budget_triangle_element(n:int, s:int, b:int=1):
    """
    A $+b$ shifted triangle, truncated on and under 0, then completed at 0
    """
    if s<0 or (n+s-b)%2: #odd
        return 0
    elif s>0:
        return zcomb(n, (n+s-b)//2) - zcomb(n, (n+s+b)//2)
    else:
        #return ruin_budget_triangle(n-1, s+1, b)
        return zcomb(n-1, (n+s-b)//2) - zcomb(n-1, (n+s+b)//2)
        #it is into the catalan triangle

################

def ruin_budget_dist(n:int, s:int, b:int=1, p:float=0.5):
    """
    The corresponding probability distribution of a budgeted ruinable triangle.
    """
    #return ruin_budget_triangle(n, s, b) * p**k * (1-p)**(n-k)    
    return ruin_budget_triangle_element(n, s, b) * p**((s+n-b)//2) * (1-p)**((n-s+b)//2)    


###########################################
# MIRRORED TRIANGLE
###########################################

def upper_mirrored_centered_triangle_element(n:int, s:int, m:int=0):
    """
    """
    if (n-m-abs(m-s))%2==0: #even
        return zcomb(n, (n-m-abs(m-s))//2)
    else:
        return 0

def lower_mirrored_centered_triangle_element(n:int, s:int, m:int=0): 
    """
    """
    return zcomb(n, (n+m-abs(m-s))/2)
    
def mirrored_centered_triangle_element(n:int, s:int, m:int=0):
    """
    """
    return zcomb(n, (n-abs(m)-abs(m-s))/2 ) #mirrored towards exterior

#################

#mirrored shifted
def mirrored_budgeted_triangle_element(n:int, s:int, b:int=0, m:int=0): 
    """
    """
    return zcomb(n, (n-abs(m-b)-abs(m-s))/2 )


###########################################
# REWARDED TRIANGLE
###########################################

#rewarded triangle
def rew_triangle_element(n:int, s:int, r:int=+1, c:int=0): 
    """
    Rewarded Triangle
    """
    if (s-(n*c))%(r-c)==0: #integer
        return zcomb(n, (s-(n*c))//(r-c))
    else:
        return 0

###########################################
# VALUED TRIANGLE
###########################################

def w_triangle_element(n:int, k:int, w:int=1): 
    """
    """
    return w * zcomb(n, k)

def vw_triangle_element(n:int, k:int, v:int=1, w:int=1):
    """
    """
    if (n==0 and k==0):
        return w
    else:
        return w * zcomb(n-1, k-1) + v * zcomb(n-1, k)  


###########################################
# GENERAL TRIANGLE
###########################################
#(d,b)translated (r,c)rewarded (v)valued pascal triangle
# n = number of total rounds/trials
# s = sum of rewards
# r = reward on success
# c = reward on failure
# d = delay on time (displacement)
# b = initial budget (shift)
# w = the value placed at the origin 
#general_triangle =   lambda n, s, w=1, r=1, c=0, d=0, b=0 : w * zbinom(n-d, (s-b-((n-d)*c))/(r-c)) if (s-b-((n-d)*c))/(r-c)).is_integer() else 0
#general_triangle =   lambda n, s, w=1, r=1, c=0, d=0, b=0 : w * rewarded_pascal_triangle(n-d, s-b, r, c)
#
def gen_triangle_element (n:int, s:int, w:int=1, r:int=1, c:int=0, d:int=0, b:int=0):
    """
    GENERAL TRIANGLE ELEMENT
    $(d,b)$-translated $(r,c)$-rewarded $(v)$-valued Pascal's Triangle
    n = number of total rounds/trials
    s = sum of rewards
    r = reward on success
    c = reward on failure
    d = delay on time (displacement)
    b = initial budget (shift)
    w = the value placed at the origin 
    
    """
    if (s-b-c*n+d*c)%(r-c)==0: #integer
        return w * zcomb(n-d, (s-b-c*n+d*c)//(r-c))
    else:
        return 0


################

def gen_triangle(origin=(0,0), parents=[(-1,-1),(-1,0)], factors=[+1, +1],
                 min_corner=(0,0), max_corner=(10,10),
                 gen_element_func = gen_triangle_element, origin_value=1,
                 recursive_construction=False):
    """
    Returns a squared matrix containing a Triangle.
    """
    
    #get budget and delay, corresponding to the origin coordinates of the triangle
    (initial_b, initial_t) = origin
    
    #get the size of the matrix based on the corners coordinates
    shape = tuple(p2-p1+1 for p1, p2 in zip(min_corner, max_corner))
    (amplitude_t, amplitude_b) = shape

    max_t = amplitude_t - initial_t - 1
    max_b = amplitude_b - initial_b - 1
        
    #recursive
    if recursive_construction:
        matrix = np.zeros(shape, dtype=int)
        matrix[origin] = origin_value
        #for t in range(initial_t, max_t+1):
        for t in range(amplitude_t):
            #for b in range(initial_b, max_b+1):
            for b in range(amplitude_b):
                for parent in parents:
                    (parent_t, parent_b) = tuple(p1+p2 for p1, p2 in zip((t,b), parent))
                    if parent_t>=0 or parent_b>=0:
                        matrix[t,b] += matrix[parent_t, parent_b]
        
    #using coefficient function
    else:
        matrix = np.array([[gen_element_func(t, b) for b in range(initial_b, max_b+1)] for t in range(initial_t, max_t+1)])
    
    return matrix

###########################################

def trunc_gen_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, h=0):
    """
    """
    return gen_triangle_element(n, s, w=w, r=r, c=c, d=d, b=b) - gen_triangle(n, s, w=w, r=r, c=c, d=d, b=b+h-(h*c//r))


###########################################

def piv_gen_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, m=0):
    """
    """
    return gen_triangle_element(n, 2*m-s, w, r, c, d, b)


###########################################

def mirror_gen_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, m=0):
    """
    """
    if m>b:
        return gen_triangle_element(n, m+abs(m-s), w=w, r=r, c=c, d=d, b=b)
    elif m<b:
        return gen_triangle_element(n, m-abs(m-s), w=w, r=r, c=c, d=d, b=b)
    else:
        return gen_triangle_element(n, s, w=w, r=r, c=c, d=d, b=b)

###########################################

def mirror_budget_triangle_element(n, s, b=0, m=0):
    """
    """
    if m>b:
        return budget_triangle_element(n, m+abs(m-s), b)
    elif m<b:
        return budget_triangle_element(n, m-abs(m-s), b)
    else:
        return budget_triangle_element(n, s, b)

###########################################

def bound_budget_triangle_element(n, s, b=0, h=0):
    """
    """
    return budget_triangle_element(n, s, b) - mirror_budget_triangle_element(n, s, b=b, m=h)
    
###########################################

def bound_budget_dist(n, s, b=0, h=0, p=0.5): 
    """
    """
    k = (s+n-b)/2
    return bound_budget_triangle_element(n, s, b, h) * p**k * (1-p)**(n-k)    

###########################################

#mirrored general triangle
# n = number of total rounds/trials
# s = sum of rewards
# r = reward on success
# c = reward on failure
# d = delay on time (displacement)
# b = initial budget (shift)
# w = the value placed at the origin
# m = mirror position 

def lower_mirrored_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, m=0):
    """
    """     
    return gen_triangle_element(n, m+abs(m-s), w, -c, -r, d, b)
    
def upper_mirrored_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, m=0): 
    """
    """
    return gen_triangle(n, m-abs(m-s), w, -c, -r, d, b)
    
def inner_mirrored_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, m=0): 
    """
    """
    if m!=b:
        return gen_triangle_element(n, m+(abs(m-s)*((m-b)//abs(m-b))), w, -c, -r, d, b)  
    else:
        return 0
        
mirrored_triangle_element = inner_mirrored_triangle_element
    
def outer_mirrored_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, m=0): 
    """
    """
    if m!=b:
        return gen_triangle_element(n, m-((m-b)*abs(m-s)//abs(m-b)), w, -c, -r, d, b)  
    else:
        return 0

################

def bound_gen_triangle_element(n, s, w=1, r=1, c=0, d=0, b=0, h=0): 
    """
    """
    return gen_triangle_element(n, s, w, r, c, d, b) - mirror_gen_triangle_element(n, s, w, r, c, d, b, h)

###########################################

# returns a string that can be used as a label for the triangle given parameters
def tri_str(w=1, r=1, c=0, d=0, b=0, m=0, h=0, mirror_mode=None, bound_mode=None):
    """
    returns a string that can be used as a label for the triangle given parameters
    """
    
    s = []
    
    if bound_mode is not None and type(bound_mode) == str:
        s += [f'${h}$-{bound_mode}']

    if mirror_mode is not None and type(mirror_mode) == str:
        s += [f'${m}$-{mirror_mode}']
    
    if w!=1:
        s += [f'${w}$-valued']
    
    if r==1 and c==0:
        s += ['classical']
    elif r==1 or c==-1:
        s += ['centered']
    elif r==-c:
        s += [f'${r}$-balanced']
    else:
        s += [f'$({r},{c})$-rewarded']
    
    if d!=0 and b!=0:
        s += [f'$({d},{b})$-displaced']
    elif b!=0:
        s += [f'${b}$-budgeted']
    elif d!=0:
        s += [f'${d}$-delayed']
    
    s += ['triangle']
    
    return ' '.join(s)



###########################################
# CATALAN TRIANGLE
###########################################

def catnum(n):
    """
    Catalan's Number
    """
    #return comb(2*n, n) // (n+1)
    #return (1 / (n+1)) * zcomb(2*n, n)
    return zcomb(2*n, n) // (n+1)

catalan_number = catnum

################

def catalan_triangle(n, m): 
    """
    Catalan's Triangle
    """
    return max(0, zcomb(n+m, m) - zcomb(n+m, m-1))

################

def center_catalan_triangle(n, s):
    """
    Centered Catalan's Triangle
    """
    if (abs(s)>n) or ((n+s)%2):
        return 0
    else:
        return catalan_triangle( ((n+s)//2), (n-s)//2 )

def exact_ruin_triangle(n, b):
    """
    """
    return center_catalan_triangle(n-1, b-1)
    
################

def exact_ruin_dist(n, b, p=0.5):
    """
    Distribution of probabilities of being ruined at a given exact round $n$, in a Gambler's Ruin game with infinite gain $g$, starting with budget $b$. 
    """
    k = (b+n)/2
    return exact_ruin_triangle(n, b) * p**(n-k) * (1-p)**k    




###########################################

        
class PascalTriangle:

    
   def __init__(self, max_n:int=10, name=None):

      if max_n < 0:
          raise ValueError()  

      #max_n is simply use to limit when extracting matrices, but all the elements can be created without constraint
      self.max_n = max_n
      
      if name is not None:
          self.name = name
      else:
          self.name = "Pascal's Triangle"


   #allow to get elements using triangle[n, k] or triangle[n][k]
   def __getitem__(self, n, k=None):
   
      #if k is None and n is a tuple, the method was called using [n, k], otherwise [n][k] is expected
      if k is None and isinstance(n, tuple):  (n, k) = n
          
      return zcomb(n, k)

   get_item = __getitem__


   def to_list(self):
      #return [[self[n, k] for k in range(self.max_n+1)] for n in range(self.max_n+1)]
      return pascal_triangle(self.max_n, return_numpy=False)

        
   def to_numpy(self):
      #return np.array(self.to_list())
      return pascal_triangle(self.max_n, return_numpy=True)


   def latex_formula(self):
       return r"$\binom{n}{k}$"


###########################################

class PascalMatrix(PascalTriangle):

   def __init__(self, max_n:int, transpose=False, name=None, recursive_construction=False):

      if max_n < 0:
          raise ValueError()  

      self.max_n = max_n
      
      self.values = pascal_triangle(max_n, recursive_construction=recursive_construction)
      self.transposed = False
      self.transpose(transpose)
      #self.reversed = False
      #self.reverse(reverse)
        
      if name is not None:
          self.name = name
      else:
          self.name = "Pascal's Matrix"
          if self.transposed:
              self.name += " (transposed)"
          #if self.reversed:
          #    self.name += " (reversed)"


   def transpose(self, transposed=None):
   
      if transposed is None:
          transposed = not self.transposed
          
      if transposed != self.transposed:
          self.values = self.values.transpose()

      self.transposed = transposed


   #def reverse(self, reversed=None):
   # 
   #   if reversed is None:
   #       reversed = not self.reversed
   #       
   #   if reversed != self.reversed:
   #       self.values = self.values[::-1]
   #
   #   self.reversed = reversed


   #allow to get elements using triangle[n, k] or triangle[n][k]
   def __getitem__(self, n, k=None):
   
      #if k is None and n is a tuple, the method was called using [n, k], otherwise [n][k] is expected
      if k is None and isinstance(n, tuple):
          (n, k) = n
          
      return self.values[n, k]

   get_item = __getitem__


   def to_str(self):
      return self.name + ":\n" + str(self.to_numpy())

   def to_list(self):
      return self.values.tolist()
        
   def to_numpy(self):
      return self.values

   
   def _prepare_plot(self, fig=None, ax=None, title=None, tight_layout=None):

       if fig is None and ax is None:
          fig, ax = plt.subplots()

       if title is not None:
          ax.set_title(title)
       else:
          ax.set_title(self.name)
       
       if tight_layout:
           fig.tight_layout()
       
       return fig, ax
       
   
   #print a matrix using pandas dataframe and adapting presentation
   def plot_table(self, title=None, fontsize=None, show=True, block=None, fig=None, ax=None, tight_layout=None):

       fig, ax = self._prepare_plot(fig=fig, ax=ax, title=title, tight_layout=tight_layout)

       ax.axis('off')

       if fontsize is None: fontsize = 100//self.max_n

       table = ax.table(cellText=self.values, colLabels=range(self.max_n+1), rowLabels=range(self.max_n+1), loc='center')
       table.auto_set_font_size(False)
       table.set_fontsize(fontsize)
       ##table.scale(2, 2)

       if show:
           plt.show(block=block)

       return fig, ax

    
   def plot(self, title=None, show=True, block=None, fig=None, ax=None, tight_layout=None, 
            positive_color='blue', zero_color='lightgray', negative_color='red',
            hide_zeros = False,
            pad_left=0.7, pad_right=0.7, pad_top=0.5, pad_bottom=0.9,
            fontsize=None, fontfamily='STIXGeneral',
            reversed = False, obliquous = False):

       fig, ax = self._prepare_plot(fig=fig, ax=ax, title=title, tight_layout=tight_layout)

       ax.xaxis.set_major_locator(MaxNLocator(integer=True))
       
       if fontsize is None: fontsize = 180//self.max_n

       for n in range(self.max_n+1):
          for k in range(self.max_n+1):
             v = self[n,k]
             if not hide_zeros or v != 0:
                 color = positive_color if v > 0 else (zero_color if v==0 else negative_color)
                 if not obliquous:
                     x = k
                 else:
                     x = 2*k-n
                 y = n+0.2
                 ax.text(x, y, str(v), ha='center', va='center', fontsize=fontsize, fontfamily=fontfamily, color=color)

       if self.transposed:
           ax.set_xlabel("$n$")
           ax.set_ylabel("$k$")
       else:
           ax.set_xlabel("$k$")
           ax.set_ylabel("$n$")

       if not obliquous:
           ax.set_xlim(-pad_left, self.max_n + pad_right)
           ax.set_ylim(-pad_top, self.max_n + pad_bottom)
       else:
           #ax.get_xaxis().set_visible(False)
           ax.set_xticks([])
           ax.set_ylim(-pad_top, self.max_n + pad_bottom)
           if not reversed:
               ax.set_xlim(-self.max_n - pad_left, self.max_n + pad_right)
           else:
               ax.set_xlim(-pad_left, 2*self.max_n + pad_right)
       
       if reversed:
           #ax.set_yticklabels(ax.get_yticklabels()[::-1]) 
           ax.set_title(ax.get_title() + " (reversed)")
       else:
           ax.xaxis.set_ticks_position('top')
           ax.xaxis.set_label_position('top')
           ax.invert_yaxis()

       if tight_layout:
           fig.tight_layout()
       if show:
           plt.show(block=block)
           
       return fig, ax


   def plot_hex(self, 
             min_diam: float = 1.0,
             align_to_origin: bool = True,
             face_color: Union[List[float], str] = None,
             edge_color: Union[List[float], str] = None,
             line_width=1.,
             plotting_gap: float = 0.,
             fontsize=None, 
             fontfamily='STIXGeneral',
             show: bool = True,
             fig=None,
             ax: plt.Axes = None,
             figsize=None,
             frame : bool = True):

      # Computes the coordinates of the hexagon centers, given the size rotation and layout specifications

      ratio = np.sqrt(3) / 2

      value_k = []
      value_n = []
      coord_x = []
      coord_y = []
      for i in range(self.max_n+1):
         for j in range(i+1):
            value_k += [j]
            value_n += [i]
            coord_x += [(j+(self.max_n-i)/2)]
            #coord_x += [2*j-i]
            coord_y += [i * ratio]

      value_k = np.array(value_k)
      value_n = np.array(value_n)

      coord_x = np.array(coord_x, dtype='float')
      coord_y = np.array(coord_y, dtype='float')
    
      coord_x = coord_x.reshape(-1, 1)
      coord_y = coord_y.reshape(-1, 1)

      if self.transposed:
         coord_x, coord_y = coord_y, coord_x
         orientation = np.deg2rad(90.0)
      else:
         orientation = 0.0
    
      origin_x = coord_x[0]  # Pick center of first hexagon as origin for rotation or crop...
      origin_y = coord_y[0]  # np.median() averages center 2 values for even arrays :\

      if ax is None:
         fig = plt.figure(figsize=figsize)
         ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            

      for i, (curr_x, curr_y, curr_n, curr_k) in enumerate(zip(coord_x, coord_y, value_n, value_k)):

            edgecolor = 'black'
            if edge_color is not None  and isinstance(edge_color, list):
                edgecolor = edge_color[i]
            facecolor = (1, 1, 1, 0)  # Make the face transparent
            if face_color is not None  and isinstance(face_color, list):
                facecolor = face_color[i]
            linewidth=1.0
            if line_width is not None and isinstance(line_width, list):
                linewidth = line_width[i]


            if self.transposed:
                transf = Affine2D().scale(1.0/ratio, 1.0) + ax.transData
            else:
                transf = Affine2D().scale(1.0, 1.0/ratio) + ax.transData
                orientation = np.deg2rad(0)
            
            polygon = RegularPolygon((curr_x, curr_y), numVertices=6,
                                              radius=min_diam / np.sqrt(3) * (1 - plotting_gap),
                                              orientation=orientation,
                                              edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth,
                                              transform=transf)
            ax.add_artist(polygon)

            if fontsize is None: fontsize = 140//self.max_n

            ax.text(curr_x, curr_y, str(comb(curr_n,curr_k)), ha='center', va='center', fontsize=fontsize, fontfamily=fontfamily, transform=transf)

      if self.transposed:
         ax.set_aspect(1.0/ratio)
      else:
         ax.set_aspect(ratio)
        
      #ax.axis([coord_x.min() - 1.0 * min_diam, coord_x.max() + 1.0 * min_diam, 
      #         coord_y.min() - 1.0 * min_diam, coord_y.max() + 2.5 * min_diam])

      if self.transposed:
            ax.set_ylabel('$k$')
            ax.set_xlabel('$n$')
            ax.set_ylim(-0.9, self.max_n+0.9)
            ax.set_xlim(-0.9, self.max_n+0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(True)
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
      else:
            ax.set_xlabel('$k$')
            ax.set_ylabel('$n$')
            ax.set_xlim(-0.9, self.max_n+0.9)
            ax.set_ylim(-0.9, self.max_n+0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()
        
      if not frame:    
            ax.axis('off')
            
      # plot using matplotlib
      if show:
         plt.show()

      return fig, ax


   def plot_digraph(self, color_node=False, 
                    title=None, show=True, block=None, fig=None, ax=None, tight_layout=None,
                    node_size=1000):
        """plot graph, set fig_size and draw nodes, edges and labels"""

        if not _NX_IMPORTED:
            raise RuntimeError("Networkx cannot be imported.")

        digraph = nx.DiGraph()
        
        #create_root_node
        #digraph.add_node(f"{0}_{0}", weight=1, pos=np.array([0,0]))

        for y, row in enumerate(self.values):
            for x, w in enumerate(row):
                digraph.add_node(f"{y}_{x}", weight=w, pos=np.array([x, -y-1]))

        labels = {n:int(w['weight']) for n, w in digraph.nodes(data=True)}
        #pos = nx.drawing.nx_pydot.pydot_layout(digraph, prog='dot')
        pos = {node[0]: node[1]['pos'] for node in digraph.nodes(data=True)}
        
        #fill graph enumerating throw previously generated list respecting to weight attribute
        for y, l in enumerate(self.values[1:]):
            row_len = len(l)
            for x, w in enumerate(l):
                #digraph.add_node(f"{y+1}_{x}", weight=w, pos=np.array([x, -y-1]))
                if x == 0:
                    digraph.add_edge(f"{y}_{x}",   f"{y+1}_{x}")
                elif x == row_len-1:
                    digraph.add_edge(f"{y}_{x-1}", f"{y+1}_{x}")
                else:
                    digraph.add_edge(f"{y}_{x}",   f"{y+1}_{x}")
                    digraph.add_edge(f"{y}_{x-1}", f"{y+1}_{x}")                    

        node_colors='#eecccc'
        if color_node:
            node_colors = [w for r in digraph.pt for w in r]

        
        box_size = len(self.values) + 1
        box_size = 9 if box_size < 9 else box_size
        
        fig, ax = self._prepare_plot(fig=fig, ax=ax, title=title, tight_layout=tight_layout)

        #plt.figure(1,figsize=(box_size,box_size)) 
        
        nx.draw_networkx_nodes(digraph, 
                               pos, 
                               node_size=node_size, 
                               #node_color=color_node,
                               edgecolors='#000000', linewidths=1, 
                               #node_color=node_colors, vmin=min(node_colors), vmax=max(node_colors),
                               #cmap='Set3'
                               )
        nx.draw_networkx_edges(digraph, pos, edgelist=digraph.edges(), arrows=False, width=1)
        nx.draw_networkx_labels(digraph, pos, labels=labels, font_size=18);
        
        plt.show()


       
   def to_latex_matrix(self, hide_zeros=True):
        
       s = r'\begin{matrix}' + '\n'
       for n in range(self.max_n+1):
          s += ' '
          #ss = ' & '.join(str(self[n, k] for k in range(max_n+1))
          for k in range(self.max_n+1):
             v = self[n, k]
             if not hide_zeros or v!=0:
                s += str(self[n, k])
             s += " & "
          s += r' \\ ' + '\n'
       s += '\end{matrix}'
       return s
            
    
   def __str__(self):
        return self.to_str() 


   #convert a matrix to a pandas dataframe
   def to_df(self, 
              label_rows=None, label_cols=None, 
              label_axis_cols=None, label_axis_rows=None, 
              transpose=False, reverse=False,
              hide_zeros=False):
    
        if not _PANDAS_IMPORTED:
            raise RuntimeError("Pandas cannot be imported.")
            
        df = DataFrame(self.values, index=label_rows, columns=label_cols)
        df = df.rename_axis(label_axis_rows)
        df = df.rename_axis(label_axis_cols, axis="columns")
        
        if transpose:
            df = df.T
        
        if reverse:
            df = df.iloc[::-1]
        
        if hide_zeros:
            df = df.replace(0, "")
        
        return df
    
        
   def to_dfs(self, title=None, heatmap=None, precision=None, 
              label_rows=None, label_cols=None, 
              label_axis_cols=None, label_axis_rows=None, 
              transpose=False, reverse=False,
              hide_zeros=False):
    
        df = self.to_df(label_rows=label_rows, label_cols=label_cols, 
        label_axis_cols=label_axis_cols, label_axis_rows=label_axis_rows, 
        transpose=transpose, reverse=reverse,
        hide_zeros=hide_zeros)

        dfs = df.style
        
        if title is not None:
            dfs = dfs.set_caption(title)    
        #df.style.applymap(lambda v : "color: lightgray" if v==0 else ("color: yellow" if v<0 else "color: black") )
        dfs = dfs.map(lambda v : "color: lightgray" if v==0 else ("color: yellow" if isinstance(v, int) and v<0 else "color: black") )
        dfs = dfs.set_table_styles([ 
                {"selector":"th.row_heading", "props": [("border-right", "1px solid black")]},
                {"selector":"th.index_name", "props": [("border-right", "1px solid black")]},
                #{"selector":"td", "props": [("padding", "2px"), ("margin", "0")]},
                #{"selector":"th", "props": [("padding", "2px"), ("margin", "0")]},
            ], overwrite=False)
        dfs = dfs.set_table_styles({0:[ 
                {"selector":"td", "props": [("border-top", "1px dotted grey"), ("border-bottom", "1px dotted grey")]},
            ]}, axis=1, overwrite=False)
        dfs = dfs.set_table_styles({0:[ 
                {"selector":"td", "props": [("border-left", "1px dotted grey"), ("border-right", "1px dotted grey")]},
            ]}, overwrite=False)
        if heatmap is not None:
            dfs = dfs.background_gradient(axis=None, cmap=heatmap)
            dfs = dfs.map(lambda v : "color: lightgray" if v==0 else "")
        if precision is not None:
            dfs = dfs.set_properties(precision=precision)
            
        return dfs
    
###############################################################################

def printdf(data, label_rows=None, label_cols=None, label_axis_cols="s", label_axis_rows="n",  title=None, 
            transpose=False, reverse=False, hide_zeros=False,
            heatmap=None, precision=None,
            show=True):

    if not _PANDAS_IMPORTED:
        raise RuntimeError("Pandas cannot be imported.")

    df = DataFrame(data, index=label_rows, columns=label_cols)
    df = df.rename_axis(label_axis_rows)
    df = df.rename_axis(label_axis_cols, axis="columns")
    
    if transpose:
        df = df.T
    
    if reverse:
        df = df.iloc[::-1]
    
    if hide_zeros:
        df = df.replace(0, "")
    
    dfs = df.style
    
    if title is not None:
        dfs = dfs.set_caption(title)    
    #df.style.applymap(lambda v : "color: lightgray" if v==0 else ("color: yellow" if v<0 else "color: black") )
    dfs = dfs.map(lambda v : "color: lightgray" if v==0 else ("color: yellow" if isinstance(v, int) and v<0 else "color: black") )
    dfs = dfs.set_table_styles([ 
            {"selector":"th.row_heading", "props": [("border-right", "1px solid black")]},
            {"selector":"th.index_name", "props": [("border-right", "1px solid black")]},
            #{"selector":"td", "props": [("padding", "2px"), ("margin", "0")]},
            #{"selector":"th", "props": [("padding", "2px"), ("margin", "0")]},
        ], overwrite=False)
    dfs = dfs.set_table_styles({0:[ 
            {"selector":"td", "props": [("border-top", "1px dotted grey"), ("border-bottom", "1px dotted grey")]},
        ]}, axis=1, overwrite=False)
    dfs = dfs.set_table_styles({0:[ 
            {"selector":"td", "props": [("border-left", "1px dotted grey"), ("border-right", "1px dotted grey")]},
        ]}, overwrite=False)
    if heatmap is not None:
        dfs = dfs.background_gradient(axis=None, cmap=heatmap)
        dfs = dfs.map(lambda v : "color: lightgray" if v==0 else "")
    if precision is not None:
        dfs = dfs.set_properties(precision=precision)
        
    
    if _IPYTHON_IMPORTED and show:
        display(dfs)
    else:
        print(dfs)
    

###############################################################################

def print_latex_matrix(data, hide_zeros=True, hide_negative=True,
                        transpose=False, reverse=False, obliquous=False,
                        show=True):

    data = np.array(data)
    
    if transpose:
        data = data.transpose()
    
    if reverse:
        data = data[::-1]
    
    s = r'\begin{matrix}' + '\n'
    for n in range(len(data)):
       s += ' '
       #ss = ' & '.join(str(self[n, k] for k in range(max_n+1))
       for k in range(len(data[n])):
          v = data[n, k]
          if not hide_zeros or v!=0:
             s += str(v)
          s += " & "
       s += r' \\ ' + '\n'
    s += '\end{matrix}'
    
    if _IPYTHON_IMPORTED and show:
        display(Latex(s))
    else:
        print(s)
    
        
###############################################################################


class TriMatrix:
    
    def __init__(self, n:int, init=0, f=None):
        self.n = n    #number of rows (= cols)
        self.length = n*(n+1)//2   #total number of elements
        self.init = init
        self.data = [init]*self.length
        if f is not None:
            self.fill(f)
        
    def __getitem__(self, index):
        if isinstance(index, list) or isinstance(index, tuple):
            i, j = index
            if i >= j and j >= 0 and i<self.n:
                return self.data[i*(i+1)//2 + j]
            else:
                raise IndexError('index out of range')
        else:
            return self.data[index]
        
    def __setitem__(self, index, value):
        if isinstance(index, list) or isinstance(index, tuple):
            i, j = index
            if i >= j and j >= 0 and i<self.n:
                self.data[i*(i+1)//2 + j] = value
            else:
                raise IndexError('index out of range')
        else:
            self.data[index] = value
        
    def __str__(self):
        padsize = len(str(max(self.data)))
        return '\n'.join([' '.join([str(self[i,j]).rjust(padsize) for j in range(i+1)]) for i in range(self.n)]) + '\n'

    to_str = __str__
    
    def latex(self, centered:bool=True):
        s = r'\begin{matrix} '
        for i in range(self.n):
            if centered:
                s += r' ' + r'& '*(self.n-i)
            for j in range(i+1):
                s += str(self[i,j]) + r' & '
                if centered:
                    s += r' & '
            s += r' \\ '
        s += '\end{matrix}'
        return s

    def plot(self, ax=None, centered:bool=True, invert_yaxis:bool=True, set_limits:bool=True, fontsize=None, equalize=False):
        if fontsize is None:
            fontsize = 300//self.n
        for i in range(self.n):
            for j in range(i+1):
                if centered:
                    x, y = 2*j-i, i+0.2
                else:
                    x, y = j, i+0.2
                ax.text(x, y, str(self[i,j]), ha='center', va='center', size=fontsize )
        if set_limits:
            if centered:
                ax.set_xlim(-self.n-0.7, self.n+0.3)
            else:
                ax.set_xlim(-0.7, self.n-0.3)
            ax.set_ylim(-0.5, self.n-0.1)
        if invert_yaxis:
            ax.invert_yaxis()
        if equalize:
            ax.set_aspect('equal')
            
    
    def __repr__(self):
        return f"Triangular Matrix of rank {str(self.n)}"
    
    def reset(self):
        """ Reset the matrix data """
        self.fill(self.init)
        
    def fill(self, f):
        if callable(f):
            for i in range(self.n):
                for j in range(i+1):
                    self[i,j] = f(i,j)
        elif isinstance(f, list) or isinstance(f, tuple):
            for i in range(self.n):
                for j in range(i+1):
                    self[i,j] = f[i,j]
        else:
            for i in range(self.n):
                for j in range(i+1):
                    self[i,j] = f
            
    def __eq__(self, other):
        """ Test equality """
        return (self.data == other.data)
        
    def __add__(self, other=None, shift=0):
        """ Add a matrix to this matrix and
        return the new matrix. Doesn't modify
        the current matrix """
        pass

    def __sub__(self, other=None, shift=0):
        """ Subtract a matrix from this matrix and
        return the new matrix. Doesn't modify
        the current matrix """
        pass
                          
    def __mul__(self, other=None, shift=0):
        """ Multiple a matrix with this matrix and
        return the new matrix. Doesn't modify
        the current matrix """
        pass


###############################################################################

class SomePascalTriangle():

    def __init__(self, rowcount):
        self.rowcount = rowcount
        self.pt = self._create()

    def _create(self):
        """Create an empty list and then append lists of 0s, each list one longer than the previous"""
        return [[0] * r for r in range(1, self.rowcount + 1)]

    def populate(self):
        """Populate an uninitialized list with actual values"""
        for r in range(0, len(self.pt)):
            for c in range(0, len(self.pt[r])):
                self.pt[r][c] = factorial(r) / (factorial(c) * factorial(r - c))

    def print_left(self):
        """Prints the triangle in a left-aligned format to demonstrate data structure"""
        for r in range(0, len(self.pt)):
            for c in range(0, len(self.pt[r])):
                print('{:>4}'.format(int(self.pt[r][c])), end="")
            print()

    def print_center(self):
        """Prints the triangle in a conventional centred format"""
        inset = int(((((len(self.pt) * 2) - 1) / 2) * 3))
        for r in range(0, len(self.pt)):
            print(" " * inset, end="")
            for c in range(0, len(self.pt[r])):
                print('{:>3}   '.format(int(self.pt[r][c])), end="")
            print()
            inset -= 3


###############################################################################

#UNIT TESTS
if __name__ == "__main__":

    max_n = 10
    
    #create a numpy matrix for the Pascal's Triangle using the pyrl.gr.zcomb function and print it to the console 
    triang1 = pascal_triangle(max_n)
    print("\n", triang1)
    print(triang1.dtype)

    #create a numpy matrix for the Pascal's Triangle recursivelly and print it to the console 
    triang2 = pascal_triangle(max_n=max_n, recursive_construction=True)
    print("\n", triang2)
    print(triang2.dtype)
    print("Equivalency:", np.all(triang1 == triang2))

    #create a Pascal's Triangle and print some rows to the console 
    triang3_obj = PascalTriangle(max_n=max_n)
    triang3 = triang3_obj.to_numpy()
    print("\n", triang3)
    print(triang3.dtype)
    print("Equivalency:", np.all(triang1 == triang3))

    triang4 = gen_triangle(max_corner=(max_n, max_n))
    print("\n", triang4)
    print(triang4.dtype)
    print("Equivalency:", np.all(triang1 == triang4))

    triang5 = gen_triangle(max_corner=(max_n, max_n), recursive_construction=True)
    print("\n", triang5)
    print(triang5.dtype)
    print("Equivalency:", np.all(triang1 == triang5))
    
    #create a Pascal's matrix 
    triang6_obj = PascalMatrix(max_n=max_n)
    triang6 = triang6_obj.to_numpy()
    print("\n", triang6)
    print(triang6.dtype)
    print("Equivalency:", np.all(triang1 == triang6))

    #print it to the console 
    print()
    print(triang6_obj)

    #print it to the console as a latex matrix
    print()
    print(triang6_obj.to_latex_matrix())
    
    #plot it 
    triang6_obj.plot(tight_layout=True, block=False)
    triang6_obj.plot(tight_layout=True, hide_zeros=True, block=False)
    triang6_obj.plot(tight_layout=True, hide_zeros=True, obliquous=True, block=False)
    
    triang6_obj.plot_digraph()
    
    print()
    triang7_obj = PascalMatrix(max_n=max_n, transpose=True)
    print(triang7_obj)

    triang7_obj.plot(tight_layout=True, block=False)
    triang7_obj.plot(tight_layout=True, reversed=True, block=False)
    triang7_obj.plot(tight_layout=True, reversed=True, hide_zeros=True, block=False)
    triang7_obj.plot(tight_layout=True, reversed=True, hide_zeros=True, obliquous=True, block=False)

    triang7_obj.plot_table(tight_layout=True, block=False)
    
    triang7_obj.plot_hex()
