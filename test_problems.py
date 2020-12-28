#!/usr/bin/env python
import numpy as np
import time
from mo_utils import ND_Front
import math

class ZDT1:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional Ackley function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        J[0] = x[0]
        t = 0
        for i in range(1, self.dim):
            t = t + x[i]
        g = 1 + 9 * (t / (self.dim - 1))
        J[1] = g * (1 - np.sqrt(J[0] / g))
        return J

    def paretofront(self):
        g = 1
        J = np.zeros([1001, 2])
        for i in range(1001): # true pareto front if known
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = g * (1 - np.sqrt(J[i, 0] / g))
        return J

class ZDT2:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional Ackley function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        J[0] = x[0]
        t = 0
        for i in range(1, self.dim):
            t = t + x[i]
        g = 1 + 9 * (t / (self.dim - 1))
        J[1] = g * (1 - (J[0] / g)**2)
        return J

    def paretofront(self):
        g = 1
        J = np.zeros([1001, 2])
        for i in range(1001): # true pareto front if known
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = g * (1 - (J[i, 0] / g)**2)
        return J

class ZDT3:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional Ackley function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        J[0] = x[0]
        t = 0
        for i in range(1, self.dim):
            t = t + x[i]
        g = 1 + 9 * (t / (self.dim - 1))
        J[1] = g * (1 - np.sqrt(J[0] / g) - (J[0] / g) * np.sin(10 * np.pi * J[0]))
        return J

    def paretofront(self):
        g = 1
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = g * (1 - np.sqrt(J[i, 0] / g) - (J[i, 0] / g) * np.sin(10 * np.pi * J[i, 0]))

        (ndf_index, df_index) = ND_Front(np.transpose(J))
        return J[ndf_index, :]

class ZDT4:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional Ackley function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        J[0] = x[0]
        t = 0
        xnew = np.copy(x)
        for i in range(1, self.dim):
            xnew[i] = -5 + 10*x[i]
            t = t + (xnew[i]**2 - 10*np.cos(4*np.pi*xnew[i]))
        g = 1 + 10*(self.dim-1) + t
        J[1] = g * (1 - np.sqrt(J[0] / g))
        return J

    def paretofront(self):
        g = 1
        J = np.zeros([1001, 2])
        for i in range(1001): # true pareto front if known
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = g * (1 - np.sqrt(J[i, 0] / g))
        return J

class ZDT6:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional Ackley function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        J[0] = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]))**6
        t = 0
        for i in range(1, self.dim):
            t = t + x[i]
        g = 1 + 9 * ((t / (self.dim - 1))**0.25)
        h = 1 - (J[0] / g)**2
        J[1] = g * h
        return J

    def paretofront(self):
        g = 1
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = g * (1 - (J[i, 0] / g)**2)
        return J


class LZF1:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional UF1 function (CEC 2009) and F2 in Zhang 2009 \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        t_even = 0
        count_even = 0
        t_odd = 0
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            xnew[i-1] = -1 + 2 * xnew[i - 1]
            if((i % 2) == 0):
                t_even = t_even + (xnew[i - 1] - np.sin(6 * np.pi * xnew[0] + i * np.pi / self.dim))**2
                count_even = count_even + 1
            else:
                t_odd = t_odd + (xnew[i - 1] - np.sin(6 * np.pi * xnew[0] + i * np.pi / self.dim))**2
                count_odd = count_odd + 1
            i = i + 1
        J[0] = xnew[0] + 2 * t_odd / count_odd
        J[1] = 1 - np.sqrt(xnew[0]) + 2 * t_even / float(count_even)
        return J

    def paretofront(self):
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = 1 - np.sqrt(J[i, 0])
        return J

class LZF2:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional UF2 function (CEC 2009) and F5 in Zhang 2009 \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        t_even = 0
        count_even = 0
        t_odd = 0
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            xnew[i - 1] = -1 + 2 * xnew[i - 1]
            if((i % 2) == 0):
                t_even = t_even + (xnew[i - 1] - (0.3 * xnew[0]**2 * np.cos(24 * np.pi * xnew[0] + 4 * i * np.pi / self.dim) + 0.6 * xnew[0]) * np.sin(6 * np.pi * xnew[0] + i * np.pi / self.dim))**2
                count_even = count_even+1
            else:
                t_odd = t_odd + (xnew[i-1] -(0.3 * xnew[0]**2 * np.cos(24 * np.pi * xnew[0] + 4 * i * np.pi / self.dim) + 0.6 * xnew[0]) * np.cos(6 * np.pi * xnew[0] + i * np.pi / self.dim))**2
                count_odd = count_odd + 1
            i = i + 1
        J[0] = xnew[0] + 2 * t_odd / float(count_odd)
        J[1] = 1 - np.sqrt(xnew[0]) + 2 * t_even / float(count_even)
        return J

    def paretofront(self):
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = 1 - np.sqrt(J[i, 0])
        return J


class LZF3:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional UF3 function (CEC 2009) and F8 in Zhang 2009 \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        sum_even = 0
        prod_even = 1
        count_even = 0
        sum_odd = 0
        prod_odd = 1
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            if((i % 2) == 0):
                y = xnew[i-1] - xnew[0]**(0.5 * (1.0 + 3.0 * (i-2.0) / (self.dim-2)))
                sum_even = sum_even + (y)**2
                prod_even = prod_even * (np.cos(20 * y * np.pi / np.sqrt(i)))
                count_even = count_even + 1
            else:
                y = xnew[i - 1] - xnew[0]**(0.5 * (1.0 + 3.0 * (i-2.0) / (self.dim-2)))
                sum_odd = sum_odd + (y)**2
                prod_odd = prod_odd * (np.cos(20 * y * np.pi / np.sqrt(i)))
                count_odd = count_odd + 1
            i = i+1
        J[0] = xnew[0] + 2 * (4 * sum_odd - 2 * prod_odd + 2) / float(count_odd)
        J[1] = 1 - np.sqrt(xnew[0]) + 2 * (4 * sum_even - 2 * prod_even + 2) / float(count_even)
        return J

    def paretofront(self):
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = 1 - np.sqrt(J[i, 0])
        return J

class LZF4:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional UF4 function (CEC 2009) \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        sum_even = 0
        count_even = 0
        sum_odd = 0
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            xnew[i - 1] = -2 + 4 * xnew[i - 1]
            if((i % 2) == 0):
                y = xnew[i-1] - np.sin(6*np.pi*xnew[0] + i*np.pi/self.dim)
                h = np.abs(y)/(1.0 + np.exp(2*np.abs(y)))
                sum_even = sum_even + h
                count_even = count_even + 1
            else:
                y = xnew[i-1] - np.sin(6*np.pi*xnew[0] + i*np.pi/self.dim)
                h = np.abs(y)/(1.0 + np.exp(2*np.abs(y)))
                sum_odd = sum_odd + h
                count_odd = count_odd + 1
            i = i+1
        J[0] = xnew[0] + 2 * sum_odd / float(count_odd)
        J[1] = 1 - xnew[0]**2 + 2 * sum_even / float(count_even)
        return J

    def paretofront(self):
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = 1 - J[i, 0]**2
        return J

class LZF5:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional F3 Function in Zhang 2009 \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        t_even = 0
        count_even = 0
        t_odd = 0
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            xnew[i - 1] = -1 + 2 * xnew[i - 1]
            if((i % 2) == 0):
                t_even = t_even + (xnew[i-1] - 0.8*xnew[0]*np.sin(6*np.pi*xnew[0] + i*np.pi/self.dim))**2
                count_even = count_even + 1
            else:
                t_odd = t_odd + (xnew[i-1] - 0.8*xnew[0]*np.cos(6*np.pi*xnew[0] + i*np.pi/self.dim))**2
                count_odd = count_odd + 1
            i = i + 1
        J[0] = xnew[0] + 2 * t_odd / float(count_odd)
        J[1] = 1 - np.sqrt(xnew[0]) + 2 * t_even / float(count_even)
        return J

    def paretofront(self):
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = 1 - np.sqrt(J[i, 0])
        return J

class LZF6:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=8, nobj=2):
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional F4 Function in Zhang 2009 \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = self.paretofront()

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        J = np.zeros(2)
        t_even = 0
        count_even = 0
        t_odd = 0
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            xnew[i - 1] = -1 + 2 * xnew[i - 1]
            if((i % 2) == 0):
                t_even = t_even + (xnew[i-1] - 0.8*xnew[0]*np.sin(6*np.pi*xnew[0] + i*np.pi/self.dim))**2
                count_even = count_even + 1
            else:
                t_odd = t_odd + (xnew[i-1] - 0.8*xnew[0]*np.cos((6*np.pi*xnew[0] + i*np.pi/self.dim)/3.0))**2
                count_odd = count_odd + 1
            i = i + 1
        J[0] = xnew[0] + 2 * t_odd / float(count_odd)
        J[1] = 1 - np.sqrt(xnew[0]) + 2 * t_even / float(count_even)
        return J

    def paretofront(self):
        J = np.zeros([1001, 2])
        for i in range(1001):
            J[i, 0] = np.double(i) / 1000
            J[i, 1] = 1 - np.sqrt(J[i, 0])
        return J

################################################################################
# DTLZ Problems
################################################################################

class DTLZ1:
    def __init__(self, nobj = 2):
        dim = nobj + 4
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional DTLZ Function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = None

    def objfunction(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum([math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim-k:]]))
        f = [0.5 * (1.0 + g)]*self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj-i-1):
                f[i] *= solution[j]
            if i > 0:
                f[i] *= 1 - solution[self.nobj-i-1]
        f = np.asarray(f)
        return f

class DTLZ2:
    def __init__(self, nobj = 2):
        dim = nobj + 9
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional DTLZ Function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = None

    def objfunction(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g =  sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim-k:]])
        f = [1.0 + g]*self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi *solution[self.nobj-i-1])
        f = np.asarray(f)
        return f

class DTLZ3:
    def __init__(self, nobj = 2):
        dim = nobj + 9
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional DTLZ Function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = None

    def objfunction(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum([math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim-k:]]))
        f = [1.0 + g]*self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi *solution[self.nobj-i-1])
        f = np.asarray(f)
        return f

class DTLZ4:
    def __init__(self, nobj = 2):
        dim = nobj + 9
        self.xlow =  np.zeros(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.nobj = nobj
        self.info = str(dim)+"-dimensional DTLZ Function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.pf = None

    def objfunction(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        alpha = 100.0
        g =  sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim-k:]])
        f = [1.0 + g]*self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi * math.pow(solution[j], alpha))
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi *math.pow(solution[self.nobj-i-1],alpha))
        f = np.asarray(f)
        return f

# class DTLZ7:
#     def __init__(self, nobj = 2):
#         dim = nobj + 19
#         self.xlow =  np.zeros(dim)
#         self.xup = np.ones(dim)
#         self.dim = dim
#         self.nobj = nobj
#         self.info = str(dim)+"-dimensional DTLZ Function \n" +\
#                              "Global optimum: f(0,0,...,0) = 0"
#         self.integer = []
#         self.continuous = np.arange(0, dim)
#         self.pf = None
#
#     def objfunction(self, solution):
#         if len(solution) != self.dim:
#             raise ValueError('Dimension mismatch')
#         k = self.dim - self.nobj + 1
#         solution = list(solution)
#
#         g =  1.0 + (sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim-k:]]))
#         f = [1.0 + g]*self.nobj
#
#         for i in range(self.nobj):
#             for j in range(self.nobj-i-1):
#                 f[i] *= math.cos(0.5 * math.pi * solution[j])
#             if i > 0:
#                 f[i] *= math.sin(0.5 * math.pi *solution[self.nobj-i-1])
#         f = np.asarray(f)
#         return f

# data = DTLZ2()
# sol = 0.5*np.ones((13))
# print(sol)
# f = data.objfunction(sol)
# print(f)

#sol = [0.5, 0.57143, 0.35714, 0.64286, 0.21429, 0.64286]
# sol = [1, 0.0, 0.0, 0.0, 0.0, 0.0]
# f = data.objfunction(sol)
# print(f)
