
from pyrl.gr import *
from pyrl.utils import printdf

#from gr import *
#from utils import printdf

from math import comb
import math

k_arr = range(0,10)
n_arr = range(0,10)

printdf([[comb(n, k) for n in n_arr] for k in k_arr], label_rows=n_arr, label_cols=k_arr, label_axis_cols="k", label_axis_rows="n", title="Classical Pascal's Triangle", transpose=True, reverse=False, mode='pyplot')
printdf([[catalan_triangle(n, k) for n in n_arr] for k in k_arr], label_rows=n_arr, label_cols=k_arr, label_axis_cols="k", label_axis_rows="n", title="Catalan's Triangle", transpose=True, reverse=False, mode='pyplot')

h = 3

printdf([[ztrunc(n, k, h) for n in n_arr] for k in k_arr], label_rows=n_arr, label_cols=k_arr, label_axis_cols="k", label_axis_rows="n", title="+3 Truncated Pascal's Triangle", transpose=True, reverse=False, mode='pyplot')

# h-Truncated b-Budgeted Triangle

max_s = 15
max_n = 20
s_arr = range(-max_s,max_s+1)
n_arr = range(max_n+1)

b=0  #initial budget (budget shift)
h=-2 #lower bound

printdf([[bound_budget_triangle(n, s, b, h) for s in s_arr] for n in n_arr], label_rows=n_arr, label_cols=s_arr, label_axis_cols="s", label_axis_rows="n", title=f'${h}$-bounded ${b}$-budgeted centered triangle:', transpose=True, reverse=True, mode='pyplot')

w=1  #value at origin
r=1  #reward (succes)
c=-1 #cost (failure)
d=0  #delay (time shift)

printdf([[bound_gen_triangle(n, s, w, r, c, d, b, h) for s in s_arr] for n in n_arr], label_rows=n_arr, label_cols=s_arr, label_axis_cols="s", label_axis_rows="n", title=f'${h}$-bounded ${b}$-budgeted centered triangle:', transpose=True, reverse=True, mode='pyplot')
printdf([[trunc_gen_triangle(n, s, w, r, c, d, b, h) for s in s_arr] for n in n_arr], label_rows=n_arr, label_cols=s_arr, label_axis_cols="s", label_axis_rows="n", title=f'${h}$-truncated ${b}$-budgeted centered triangle:', transpose=True, reverse=True, mode='pyplot')

w=1; r=1; c=-1; d=0; b=5; h=0

printdf([[bound_gen_triangle(n, s, w, r, c, d, b, h) for s in s_arr] for n in n_arr], label_rows=n_arr, label_cols=s_arr, label_axis_cols="s", label_axis_rows="n", title=f'${h}$-bounded ${b}$-budgeted centered triangle:', transpose=True, reverse=True, mode='pyplot')

def bitrunc_gen_triangle(n, s, w=1, r=1, c=-1, d=0, b=0, h=-2, g=+2):
    if s>=g or s<=h:
        return 0
    else:
        result =  gen_triangle(n, s, w, r, c, d, b)
        for i in range(1,100):
            result -= mirror_gen_triangle(n, s, w, r, c, d, b+(2*i-1)*(g-h), g)
            result -= mirror_gen_triangle(n, s, w, r, c, d, b-(2*i-1)*(g-h), h)
            result += mirror_gen_triangle(n, s, w, r, c, d, b+(2*i)*(g-h), g)
            result += mirror_gen_triangle(n, s, w, r, c, d, b-(2*i)*(g-h), h)
        return result
    
b=0   #initial budget
h=-3  #lower bound
g=+5  #upper bound
    
printdf([[bitrunc_gen_triangle(n, s, b=0, h=h, g=g) for s in s_arr] for n in n_arr], label_rows=n_arr, label_cols=s_arr, label_axis_cols="s", label_axis_rows="n", title=f"{h} {g} Truncated Pascal's Triangle:", transpose=True, reverse=True, mode='pyplot')
