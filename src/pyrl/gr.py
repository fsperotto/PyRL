#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gambler's Ruin.

   This module implements GR methods.
"""

__version__ = "0.0.1"
__author__ = "Filipo S. Perotto, Aymane Ouhabi, Melvine Nargeot"
__license__ = "MIT"
__status__ = "Development"

################

from math import sqrt, factorial, comb
#from scipy.special import binom   # from python 3.8 is preferable to use math.comb
import numpy as np

    
###########################################
# CLASSIC TRIANGLE
###########################################

#pascal triangle over integer space
def zcomb(n:int, k:int):
"""pascal triangle over integer space

   binomial coefficient (combinations) extend with zeros over the entire Z space
"""
    #k and n must be naturals
    if (n >= k) and (k >= 0):
        return comb(n, k)
    else:
        return 0

pascal_triangle = zcomb

zbinom = zcomb

################

#shifted pascal triangle
def zshift(n:int, k:int, b:int=0) : 
"""shifted pascal triangle
"""
    return zcomb(n, k-b)

shift_pascal_triangle = zshift

#delayed pascal triangle
def zdelay(n:int, k:int, d:int=0) : 
"""delayed pascal triangle
"""
    return zcomb(n-d, k)

delay_pascal_triangle = zdelay

#translated pascal triangle
def ztrans(n:int, k:int, d:int=0, b:int=0): 
"""translated pascal triangle
"""
    return zcomb(n-d, k-b)

trans_pascal_triangle = ztrans

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

################

#truncated pascal triangle
def ztrunc(n:int, k:int, h:int=1): 
"""truncated pascal triangle
"""
    return zcomb(n, k) - zcomb(n, k-h)

trunc_pascal_triangle = ztrunc


###########################################
# CENTERED TRIANGLE
###########################################

#centered, equivalent to (+1,-1) rewarded
def ccomb(n:int, s:int):
"""
"""
    dk = n+s
    if dk%2==0: #even
        return zcomb( n, dk//2 )
    else:
        return 0

center_triangle = ccomb

#centering and b shifting
def bcomb(n:int, s:int, b:int=0): 
"""
"""
    #return ccomb(n, (n+s-b)//2)
    dk = n+s-b
    if dk%2==0: #even
        return zcomb( n, dk//2 )
    else:
        return 0
    
budget_triangle = bcomb

shift_center_triangle = bcomb

def bdist(n:int, s:int, b:int=0, p:float=0.5):
"""
""" 
    #return budget_triangle(n, s, b) * p**k * (1-p)**(n-k)
    #return  zcomb(n, (n+s-b)//2) * p**((n+s-b)//2) * (1-p)**((n-s+b)//2)
    dk = n+s-b
    if dk%2==0: #even
        return zdist(n, dk//2, p)
    else:
        return 0

budget_dist = bdist

#d delaying and centering
def delay_center_triangle(n:int, s:int, d:int=0):
"""
"""
    #return ccomb( n-d, s )
    dk = n-d+s
    if dk%2==0: #even
        return zcomb( n-d, dk//2 )
    else:
        return 0

#b and d translation and centering
def trans_center_triangle(n:int, s:int, d:int=0, b:int=0): 
"""
"""
    #return ccomb( n-d, n-d+s-b )
    dk = n-d+s-b
    if dk%2==0: #even
        return zcomb( n-d, dk//2 )
    else:
        return 0

################

#truncating
def trunc_center_triangle(n:int, s:int, h:int=-1):
"""
"""
    # h is the truncation position
    dk = n+s
    if dk%2==0: #even
        return zcomb(n, dk//2) - zcomb(n, (dk//2)-h)
    else:
        return 0

################

def trunc_budget_triangle(n:int, s:int, b:int=0, h:int=-1):
"""
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

################

def trunc_delay_triangle(n:int, s:int, d:int=0, h:int=-1):
"""
"""
    dk = n-d+s 
    if dk%2==0: #even
        k = dk//2
        #return ztrunc(n, k, h)
        return zcomb(n, k) - zcomb(n, k-h)
    else:
        return 0

################

#n and d displacement
def trunc_trans_triangle(n:int, s:int, b:int=0, d:int=0, h:int=-1):
"""
"""
    dk = n-d+s-b 
    if dk%2==0: #even
        k = dk//2
        j = (n-d+s+b)//2 - h    # ????
        return zcomb(n-d, k) - zcomb(n-d, j)
    else:
        return 0

#################

# is a +b shifted triangle truncated on and under 0, then completed at 0
def ruin_budget_triangle(n:int, s:int, b:int=1):
"""
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
"""
    #return ruin_budget_triangle(n, s, b) * p**k * (1-p)**(n-k)    
    return ruin_budget_triangle(n, s, b) * p**((s+n-b)//2) * (1-p)**((n-s+b)//2)    

###########################################
# MIRRORED TRIANGLE
###########################################

def upper_mirrored_centered_triangle(n:int, s:int, m:int=0):
"""
"""
    if (n-m-abs(m-s))%2==0: #even
        return zcomb(n, (n-m-abs(m-s))//2)
    else:
        return 0

def lower_mirrored_centered_triangle(n:int, s:int, m:int=0): 
"""
"""
    return zcomb(n, (n+m-abs(m-s))/2)
    
def mirrored_centered_triangle(n:int, s:int, m:int=0):
"""
"""
    return zcomb(n, (n-abs(m)-abs(m-s))/2 ) #mirrored towards exterior

#################

#mirrored shifted
def mirrored_budgeted_triangle(n:int, s:int, b:int=0, m:int=0): 
"""
"""
    return zcomb(n, (n-abs(m-b)-abs(m-s))/2 )


###########################################
# REWARDED TRIANGLE
###########################################

#rewarded triangle
def rew_triangle(n:int, s:int, r:int=+1, c:int=0): 
"""
"""
    if (s-(n*c))%(r-c)==0: #integer
        return zcomb(n, (s-(n*c))//(r-c))
    else:
        return 0

###########################################
# VALUED TRIANGLE
###########################################

def w_triangle(n:int, k:int, w:int=1): 
"""
"""
    return w * zcomb(n, k)

def vw_triangle(n:int, k:int, v:int=1, w:int=1):
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
def gen_triangle (n:int, s:int, w:int=1, r:int=1, c:int=0, d:int=0, b:int=0):
"""
"""
    if (s-b-c*n+d*c)%(r-c)==0: #integer
        return w * zcomb(n-d, (s-b-c*n+d*c)//(r-c))
    else:
        return 0

###########################################

def trunc_gen_triangle(n, s, w=1, r=1, c=0, d=0, b=0, h=0):
"""
"""
    return gen_triangle(n, s, w=w, r=r, c=c, d=d, b=b) - gen_triangle(n, s, w=w, r=r, c=c, d=d, b=b+h-(h*c//r))


###########################################

def piv_gen_triangle(n, s, w=1, r=1, c=0, d=0, b=0, m=0):
"""
"""
    return gen_triangle(n, 2*m-s, w, r, c, d, b)


###########################################

def mirror_gen_triangle(n, s, w=1, r=1, c=0, d=0, b=0, m=0):
"""
"""
    if m>b:
        return gen_triangle(n, m+abs(m-s), w=w, r=r, c=c, d=d, b=b)
    elif m<b:
        return gen_triangle(n, m-abs(m-s), w=w, r=r, c=c, d=d, b=b)
    else:
        return gen_triangle(n, s, w=w, r=r, c=c, d=d, b=b)

###########################################

def mirror_budget_triangle(n, s, b=0, m=0):
"""
"""
    if m>b:
        return budget_triangle(n, m+abs(m-s), b)
    elif m<b:
        return budget_triangle(n, m-abs(m-s), b)
    else:
        return budget_triangle(n, s, b)

###########################################

def bound_budget_triangle(n, s, b=0, h=0):
"""
"""
    return budget_triangle(n, s, b) - mirror_budget_triangle(n, s, b=b, m=h)
    
###########################################

def bound_budget_dist(n, s, b=0, h=0, p=0.5): 
"""
"""
    k = (s+n-b)/2
    return bound_budget_triangle(n, s, b, h) * p**k * (1-p)**(n-k)    

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

def lower_mirrored_triangle(n, s, w=1, r=1, c=0, d=0, b=0, m=0):
"""
""" 
    return gen_triangle(n, m+abs(m-s), w, -c, -r, d, b)
    
def upper_mirrored_triangle(n, s, w=1, r=1, c=0, d=0, b=0, m=0): 
"""
"""
    return gen_triangle(n, m-abs(m-s), w, -c, -r, d, b)
    
def inner_mirrored_triangle(n, s, w=1, r=1, c=0, d=0, b=0, m=0): 
"""
"""
    if m!=b:
        return gen_triangle(n, m+(abs(m-s)*((m-b)//abs(m-b))), w, -c, -r, d, b)  
    else:
        return 0
        
mirrored_triangle = inner_mirrored_triangle
    
def outer_mirrored_triangle(n, s, w=1, r=1, c=0, d=0, b=0, m=0): 
"""
"""
    if m!=b:
        return gen_triangle(n, m-((m-b)*abs(m-s)//abs(m-b)), w, -c, -r, d, b)  
    else:
        return 0

################

def bound_gen_triangle(n, s, w=1, r=1, c=0, d=0, b=0, h=0): 
"""
"""
    return gen_triangle(n, s, w, r, c, d, b) - mirror_gen_triangle(n, s, w, r, c, d, b, h)

###########################################

# returns a string that can be used as a label for the triangle given parameters
def tri_str(w=1, r=1, c=0, d=0, b=0, m=0, h=0, mirror_mode=None, bound_mode=None):
"""
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
"""
    #return comb(2*n, n) // (n+1)
    #return (1 / (n+1)) * zcomb(2*n, n)
    return zcomb(2*n, n) // (n+1)

catalan_number = catnum

################

def catalan_triangle(n, m): 
"""
"""
    return max(0, zcomb(n+m, m) - zcomb(n+m, m-1))

################

def center_catalan_triangle(n, s):
"""
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
"""
    k = (b+n)/2
    return exact_ruin_triangle(n, b) * p**(n-k) * (1-p)**k    

###############
