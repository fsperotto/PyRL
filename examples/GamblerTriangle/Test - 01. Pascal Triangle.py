#!/usr/bin/env python
# coding: utf-8

# # Pascal Triangle and the Binomial Coefficient

# ## Brief History
# 
# What is called "Pascal's triangle" in the Occident was known centuries before Blaise Pascal. But Pascal published the earliest known mathematical treatise about the triangle (Traité du triangle arithmétique, 1654), where he proved several of its properties (including the binomial correspondence) by inductive reasoning, and linked it to problems in probability theory.
# 
# <img src="img/pascal_triangle.png" width=600 height=auto />
# 
# ## Definition
# 
# The Pascal's triangle is classically defined as a triangular matrix on the space $\mathbb{N} \times \mathbb{N}$ of naturals numbers.
# 
# In the most common representation, the value $1$ is written into the first column, $k = 0$, and the value $0$ into the cells of the first row, $n=0$, except the first cell. 
# The cells on the other rows are calculated recursivelly by adding the immediate upper and upper-left neighbor cells: 
# 
# $\forall \ n, k \in \mathbb{N} :$
# 
# $$ C(n, k) \ = \ C^k_n \ = \ {}_nC_k \ = \ \begin{cases} 
#   0 & \text{if } n=0 \text{ and } k>0 \\
#   1 & \text{if } k=0 \\
#   C(n-1, k-1) + C(n-1, k) & \text{otherwise}
# \end{cases}$$
# 
# It forms a lower/left triangular matrix, then the definition can be simplified, as follows:
# 
# $$ C(n, k) \ = \ C^k_n \ = \ {}_nC_k \ = \ \begin{cases} 
#   0 & \text{if } k>n \\
#   1 & \text{if } k=0 \\
#   C(n-1, k-1) + C(n-1, k) & \text{otherwise}
# \end{cases}$$
# 
# Each non-zero element can be calculated directly using the binomial coefficient:
# 
# $ \forall \ n, k \in \mathbb{N} \mid n \geq k \geq 0 $ :
# 
# $$ C(n, k) 
# \ = \ \binom{n}{k} 
# \ = \ \frac{n!}{k!(n-k)!} 
# \ = \ \frac{\Gamma(n+1)}{\Gamma(k+1)\Gamma(n-k+1)}
# \ = \ \frac{n^{\underline{k}}}{k!} $$
# 

# In[1]:

from srl.gr import *
from srl.utils import printdf

################

max_n = 20
k_arr = range(0,max_n+1)
n_arr = range(0,max_n+1)

printdf([[zcomb(n, k) for k in k_arr] for n in n_arr], label_rows=n_arr, label_cols=k_arr, label_axis_cols="k", label_axis_rows="n", title='Classic Pascal Triangle (lower triangular matrix), $C(n,k)$:', html=False)
printdf([[zcomb(n, k) for k in k_arr] for n in n_arr], label_rows=n_arr, label_cols=k_arr, label_axis_cols="k", label_axis_rows="n", title='Gambling Pascal Triangle (transposed reversed presentation):', transpose=True, reverse=True, html=False)





# ## Meaning
# 
# The Pascal's Triangle indicates:
# 
#  - the number of combinations of $n$ elements, $k$ by $k$ ($n$ chose $k$);
#  
#  - the number of possible paths to reach node $(n,k)$, starting from the root node in a binary tree;
#  
#  - at each row, the coefficients of the developped power of the sum of two terms (binomial coefficients): $(a + b)^n = \sum^n_{k=0} \binom{n}{k} a^{n-k} b^{k}$;
#  
#  - at each row, divided by its sum $2^n$, the binomial distribution in the symmetric case where $p = 1/2$. 
# 
# 
# See: https://www.johndcook.com/blog/binomial_coefficients/
# 
# See: https://en.wikipedia.org/wiki/Binomial_coefficient
# 
# See: https://fr.wikipedia.org/wiki/Triangle_de_Pascal
# 
# See: https://mathworld.wolfram.com/PascalsTriangle.html
# 
# See: https://www.cut-the-knot.org/arithmetic/combinatorics/BinomialTheorem.shtml
# 
# See: https://www.mathsisfun.com/pascals-triangle.html
# 
# See: https://www.math10.com/en/algebra/probabilities/binomial-theorem/binomial-theorem.html
# 
# 
# ## Main Properties
# 
# The Pascal triangle ensure the folowing properties:
# 
# $ \binom{n}{k} = 0 $  when  $k>n$.
# 
# $ \binom{n}{k} = 1 $  when  $k=0$ or $k=n$.
# 
# $ \binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \ , \ \forall n, k > 0$
# 
# ## Gambling Context
# 
# It is the base of combinatorics, and can be seen, in the gambling context, as the number of ways a gambler can observe $k$ successes after playing $n$ rounds.
# 
# ## Sum of a row in Pascal's Triangle
# 
# The sum of all possible paths in a given level is the sum of all terms of the binomial coefficient for a given number of rounds $t$, i.e. considering sequences of "success or failure", the number of possible sequences:
# 
# $ \sum\limits_{k=0}^n \binom{n}{k} \ = \ 2^n$

# In[ ]:




