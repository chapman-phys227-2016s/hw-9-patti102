#! /usr/bin/env python

"""
File: MCint_class.py

Copyright (c) 2016 Taylor Patti

License: MIT

This class provides a Monte Carlo integrating technique to a
set of classes of an integration superclass.

"""

import numpy as np
import math
import random

class Integrator(object):
    """Contains an array of integration techniques. The one which was added
    for this exercise is MCint_vec, which is the one which is otherwise documented
    and which has its own test function."""
    def __init__(self, a, b, n):
        self.a, self.b, self.n = a, b, n
        self.points, self.weights = self.construct_method()

    def construct_method(self):
        raise NotImplementedError('no rule in class %s' %
                                  self.__class__.__name__)

    def integrate(self, f):
        s = 0
        for i in range(len(self.weights)):
            s += self.weights[i]*f(self.points[i])
        return s

    def vectorized_integrate(self, f):
        return np.dot(self.weights, f(self.points))


class Midpoint(Integrator):
    def construct_method(self):
        a, b, n = self.a, self.b, self.n  # quick forms
        h = (b-a)/float(n)
        x = np.linspace(a + 0.5*h, b - 0.5*h, n)
        w = np.zeros(len(x)) + h
        return x, w

class Trapezoidal(Integrator):
    def construct_method(self):
        x = np.linspace(self.a, self.b, self.n)
        h = (self.b - self.a)/float(self.n - 1)
        w = np.zeros(len(x)) + h
        w[0] /= 2
        w[-1] /= 2
        return x, w

class Simpson(Integrator):
    def construct_method(self):
        if self.n % 2 != 1:
            print 'n=%d must be odd, 1 is added' % self.n
            self.n += 1
        x = np.linspace(self.a, self.b, self.n)
        h = (self.b - self.a)/float(self.n - 1)*2
        w = np.zeros(len(x))
        w[0:self.n:2] = h*1.0/3
        w[1:self.n-1:2] = h*2.0/3
        w[0] /= 2
        w[-1] /= 2
        return x, w

class GaussLegendre2(Integrator):
    def construct_method(self):
        if self.n % 2 != 0:
            print 'n=%d must be even, 1 is subtracted' % self.n
            self.n -= 1
        nintervals = int(self.n/2.0)
        h = (self.b - self.a)/float(nintervals)
        x = np.zeros(self.n)
        sqrt3 = 1.0/math.sqrt(3)
	for i in range(nintervals):
            x[2*i]   = self.a + (i+0.5)*h - 0.5*sqrt3*h
            x[2*i+1] = self.a + (i+0.5)*h + 0.5*sqrt3*h
        w = np.zeros(len(x)) + h/2.0
        return x, w

class GaussLegendre2_vec(Integrator):
    def construct_method(self):
        if self.n % 2 != 0:
            print 'n=%d must be even, 1 is added' % self.n
            self.n += 1
        nintervals = int(self.n/2.0)
        h = (self.b - self.a)/float(nintervals)
        x = np.zeros(self.n)
        sqrt3 = 1.0/math.sqrt(3)
        m = np.linspace(0.5*h, (nintervals-1+0.5)*h, nintervals)
        x[0:self.n-1:2] = m + self.a - 0.5*sqrt3*h
        x[1:self.n:2]   = m + self.a + 0.5*sqrt3*h
        w = np.zeros(len(x)) + h/2.0
        return x, w
    
class MCint_vec(Integrator):
    """Uses the Monte Carlo random number generation to integrate the function. This function
    returns an array of function analysis points as well as an array of weight values for those points."""
    def construct_method(self):
        h = float(self.b - self.a) / self.n
        x = np.random.uniform(self.a, self.b, self.n)
        w = np.zeros(len(x)) + h
        return x, w


# A linear function will be exactly integrated by all
# the methods, so such an f is the candidate for testing
# the implementations

def test_Integrate():
    """Check that linear functions are integrated exactly."""
    def f(x):
        return x + 2

    def F(x):
        """Integral of f."""
        return 0.5*x**2 + 2*x

    a = 2; b = 3; n = 4     # test data
    I_exact = F(b) - F(a)
    tol = 1E-15

    methods = [Midpoint, Trapezoidal, Simpson, GaussLegendre2,
               GaussLegendre2_vec]
    for method in methods:
        integrator = method(a, b, n)

        I = integrator.integrate(f)
        assert abs(I_exact - I) < tol

        I_vec = integrator.vectorized_integrate(f)
        assert abs(I_exact - I_vec) < tol
        
def test_MC():
    """Ensures that the Monte Carlo method obtains a rather accurate value for the integration of
    the sin function on the interval 0 to pi. In order to provide for a speedy integration, small levels
    of precision were required in order to permit relatively sparse point sampling for a random function like the
    Monte Carlo method here utilized. This is reflected in the tolerance given."""
    apt = math.fabs(MCint_vec(0, np.pi, 1000000).integrate(np.sin) - 2) < 1E-3
    msg = 'Monte Carlo random integration technique is not close enough.'
    assert apt, msg