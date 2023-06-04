#Import Dependencies
import numpy as np
from numpy.random import binomial, randint, uniform, rand, normal, choice
from random import choices
import numpy.ma as ma
from math import sqrt, log
from scipy.stats import beta, norm, binom, truncnorm
from scipy.integrate import quadrature as integral
from collections import Iterable
from math import sqrt, log
from numba import jit
from itertools import accumulate as acc
    

""" partially copied from SMPyBandits"""

################################################################################
#        self.r_min = r_min  #: Lower values for rewards
#        self.r_max = r_max  #: Higher values for rewards
#        self.r_amp = r_max - r_min  #: Amplitude
#        self.r_0_1 = ((self.r_max==1.0) and (self.r_min==0.0))
################################################################################

    
################################################################################

class RandomArm():
    """ Base class for an arm class.
        return uniformly distributed random values between 0 and 1
    """
    
    def __str__(self):
        return f"Random Arm"

    def __init__(self, minr=0.0, maxr=1.0):
        """ Base class for an arm class."""
        self.minr = minr
        self.maxr = maxr
        if (minr is None) or (minr == float('-inf')) or (maxr is None) or (maxr == float('+inf')):
            self.scalable = False
            self.ampl = float('+inf')
            self.mean = None
        else:
            self.scalable = True
            self.ampl = maxr-minr
            self.mean = (minr+maxr) / 2

    def scale(self, x):
        """ scale a success value (between 0 and 1) to the correct reward range"""
        if self.scalable:
            return x * self.ampl + self.minr
        else:
            return x
    
    def draw(self, shape=None):
        """ Draw a numpy array of random samples, of a certain shape. If shape is None, return a single sample"""
        return uniform(low=self.minr, high=self.maxr, size=shape)

    def convert(self, chances_arr=None):
        """ Draw a numpy array of random samples, using already sampled seeds from 0 .. 1"""
        return self.scale(chances_arr)
    
    
################################################################################

class BernoulliArm(RandomArm):
    """ Bernoulli distributed arm."""

    def __str__(self):
        return f"Bernoulli Arm ($p={self.p}$)"
    
    def __init__(self, p, minr=0.0, maxr=1.0):
        """New arm."""
        super().__init__(minr=minr, maxr=maxr)
        #assert 0.0 <= p <= 1.0, "Error, the parameter probability for Bernoulli class has to be in [0, 1]."  # DEBUG
        if p > 1.0:
            print("SMAB warning: parameter p cannot be greater than 1.0; fixing it to 1.0")
            p = 1.0
        if p < 0.0:
            print("SMAB warning: parameter p cannot be negative; fixing it to 0.0")
            p = 0.0
        self.p = p  #: Parameter p for this Bernoulli arm
        self.mean = self.scale(p)

    # --- Random samples
    def draw(self, shape=None):
        """ Draw a numpy array of random samples, of a certain shape. If shape is None, return a single sample"""
        return self.scale(binomial(n=1, p=self.p, size=shape))
        #return np.asarray(binomial(1, self.p, shape) * self.ampl + self.minr, dtype=float)

    def convert(self, chances_arr=None):
        """ Draw a numpy array of random samples, using already sampled seeds from 0 .. 1"""
        return self.scale(binom.ppf(q=chances_arr, n=1, p=self.p))
    
################################################################################

class GaussianArm(RandomArm):
    """ Gaussian distributed arm."""

    def __str__(self):
        return f"Gaussian (Normal) Arm ($\\mu={self.mean}, \\sigma^2={self.stddev}$)"
    
    def __init__(self, mean=0.0, stddev=1.0):
        """New arm."""
        super().__init__()
        self.mean = mean
        self.stddev = stddev
        self.variance = stddev**2

    # --- Random samples
    def draw(self, shape=None):
        """ Draw a numpy array of random samples, of a certain shape. If shape is None, return a single sample"""
        return normal(loc=self.mean, scale=self.stddev, size=shape)

    def convert(self, chances_arr=None):
        """ Draw a numpy array of random samples, using already sampled seeds from 0 .. 1"""
        return norm.ppf(q=chances_arr, loc=self.mean, scale=self.stddev)
    
################################################################################

class TruncGaussianArm(GaussianArm):
    """ Truncated Gaussian distributed arm."""

    def __str__(self):
        return f"Truncated Gaussian (Normal) Arm ($a={self.r_min}, b={self.r_max}, \mu={self.mean}, \sigma^2={self.stddev}$)"
    
    def __init__(self, a=-1.0, b=+1.0, mean=0.0, stddev=1.0):
        """New arm."""
        super().__init__()
        self.mean = mean
        self.stddev = stddev
        self.variance = stddev**2
        self.r_min = r_min
        self.r_max = r_max
        self.r_amp = r_max - r_min
        self.a = (r_min - mean) / stddev
        self.b = (r_max - mean) / stddev

    # --- Random samples
    def draw(self, shape=None):
        """ Draw a numpy array of random samples, of a certain shape. If shape is None, return a single sample"""
        return truncnorm.rvs(a=self.a, b=self.b, loc=self.mean, scale=self.stddev, size=shape)

    def convert(self, chances_arr=None):
        """ Draw a numpy array of random samples, using already sampled seeds from 0 .. 1"""
        return truncnorm.ppf(q=chances_arr, a=self.a, b=self.b, loc=self.mean, scale=self.stddev)
    
    