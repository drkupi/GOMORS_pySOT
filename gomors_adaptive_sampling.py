"""
.. module:: mo_adaptive_sampling
   :synopsis: Ways of finding the next point to evaluate in the adaptive phase of MoPySOT

.. moduleauthor:: Taimoor Akhtar
                David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>
"""

import math
from pySOT.utils import *
import scipy.spatial as scp
from pySOT.heuristic_methods import GeneticAlgorithm as GA
from scipy.optimize import minimize
import scipy.stats as stats
import types
from mo_utils import *
import random
from hv import HyperVolume
import numpy as np
import time
from platypus.algorithms import NSGAII, NSGAIII, MOEAD, EpsMOEA, EpsNSGAII
from gomors_moea_problem import GlobalProblem, GapProblem, CustomGenerator
from selection_rules import OspaceDistanceSelection, DspaceDistanceSelection, HyperVolumeSelection, EpsilonSelection


def __fix_docs(cls):
    """Help function for stealing docs from the parent"""
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls



class MultiSampling(object):
    """Maintains a list of adaptive sampling methods

    A collection of adaptive sampling methods and weights so that the user
    can use multiple adaptive sampling methods for the same optimization
    problem. This object keeps an internal list of proposed points
    in order to be able to compute the minimum distance from a point
    to all proposed evaluations. This list has to be reset each time
    the optimization algorithm restarts

    :param strategy_list: List of adaptive sampling methods to use
    :type strategy_list: list
    :param cycle: List of integers that specifies the sampling order, e.g., [0, 0, 1] uses
        method1, method1, method2, method1, method1, method2, ...
    :type cycle: list
    :raise ValueError: If cycle is incorrect

    :ivar sampling_strategies: List of adaptive sampling methods to use
    :ivar cycle: List that specifies the sampling order
    :ivar nstrats: Number of adaptive sampling strategies
    :ivar current_strat: The next adaptive sampling strategy to be used
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, strategy_list, cycle):
        if cycle is None:
            cycle = range(len(strategy_list))
        if (not all(isinstance(i, int) for i in cycle)) or \
                np.min(cycle) < 0 or \
                np.max(cycle) > len(strategy_list)-1:
            raise ValueError("Incorrect cycle!!")
        self.sampling_strategies = strategy_list
        self.nstrats = len(strategy_list)
        self.cycle = cycle
        self.current_strat= 0
        self.proposed_points = None
        self.data = strategy_list[0].data
        self.fhat = None
        self.budget = None
        self.n0 = None

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.fhat = fhat
        self.n0 = start_sample.shape[0]
        for i in range(self.nstrats):
            self.sampling_strategies[i].init(self.proposed_points, fhat, budget)

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            for i in range(self.nstrats):
                self.sampling_strategies[i].remove_point(x)
            return True
        return False

    def make_points(self, npts, xbest, sigma, front, subset=None, proj_fun=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box
        :type sigma: float
        :param subset: Coordinates to perturb
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array

        .. todo:: Change the merit function from being hard-coded
        """

        new_points = np.zeros((npts, self.data.dim))

        # Figure out what we need to generate
        npoints = np.zeros((self.nstrats,), dtype=int)
        for i in range(npts):
            npoints[self.cycle[self.current_strat]] += 1
            self.current_strat = (self.current_strat + 1) % len(self.cycle)

        # Now generate the points from one strategy at the time
        count = 0
        for i in range(self.nstrats):
            if npoints[i] > 0:
                new_points[count:count+npoints[i], :] = \
                    self.sampling_strategies[i].make_points(npts=npoints[i], xbest=xbest,
                                                            sigma=sigma, front=front, subset=subset,
                                                            proj_fun=proj_fun)

                count += npoints[i]
                # Update list of proposed points
                for j in range(self.nstrats):
                    if j != i:
                        self.sampling_strategies[j].proposed_points = \
                            self.sampling_strategies[i].proposed_points

        return new_points

class EvolutionaryAlgorithm(object):
    """An implementation of the Surrogate MOEA Search also used in GOMORS

    This is an implementation of the surrogate-assisted MOEA search
    that was also described in GOMORS. An MOEA is used to solve the
    embedded surrogate MO problem (Steps 2.2 and 2.3) with population
    retention.

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, epsilons, numcand=None, cand_flag=None):
        self.data = data
        self.fhat = None
        self.xrange = self.data.xup - self.data.xlow
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.proposed_points = None
        self.previous_points = None # Only GOMORS
        self.dmerit = None
        self.xcand = None
        self.fhvals = None
        self.next_weight = 0
        self.numcand = numcand
        self.cand_flag = cand_flag
        self.n0 = None
        if self.numcand is None:
            self.numcand = min([5000, 100*data.dim])
        self.budget = None
        self.epsilons = epsilons


        # Check that the inputs make sense
        if not(isinstance(self.numcand, int) and self.numcand > 0):
            raise ValueError("The number of candidate points has to be a positive integer")

    def init(self, start_sample, fhat, budget, prev_sample=None):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.previous_points = prev_sample
        self.budget = budget
        self.fhat = fhat
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            return True

        idx = np.sum(np.abs(self.previous_points - x), axis=1).argmin()
        if np.sum(np.abs(self.previous_points[idx, :] - x)) < 1e-10:
            self.previous_points = np.delete(self.previous_points, idx, axis=0)
            return True

        return False

    def __generate_cand__(self, xfront, xgap=None):
        if xgap is None:
            problem = GlobalProblem(self.data.dim, self.data.nobj, self.fhat)
            generator = CustomGenerator(100)
            generator.create(problem, xfront)
            #algorithm = NSGAIII(problem, divisions_outer=100,generator=generator)
            #algorithm = NSGAII(problem=problem, population_size=100, generator=generator)
            algorithm = EpsNSGAII(problem=problem, epsilons=self.epsilons, population_size=100, generator=generator)
            #algorithm = EpsMOEA(problem=problem, epsilons=self.epsilons, population_size=100, generator=generator)
        else:
            rgap = 0.1
            problem = GapProblem(self.data.dim, self.data.nobj, self.fhat,  xgap, rgap)
            #algorithm = NSGAIII(problem, divisions_outer=100)
            #algorithm = EpsNSGAII(problem=problem, epsilons=self.epsilons, population_size=100)
            algorithm = EpsMOEA(problem=problem, epsilons=self.epsilons, population_size=100)
        algorithm.run(2500)
        self.xcand = np.asarray([s.variables for s in algorithm.result])
        self.fhvals = np.asarray([s.objectives for s in algorithm.result])
        #print(self.fhvals.shape)


    def generate_candidates(self, xgap, sigma):
        # Step 1 - Compute DDS Probability
        if self.budget < 2:
            ddsprob = 0
        else:
            ddsprob = min([20.0/self.data.dim, 1.0]) * (1.0 - (np.log(self.proposed_points.shape[0] - self.n0 + 1.0) / np.log(self.budget - self.n0)))
        minprob = np.min([1.0, 1.0/self.data.dim])
        ddsprob = np.max([minprob, ddsprob])

        nlen = len(xgap)
        # Step 2 - Generate candidates
        # Fix when nlen is 1
        # Todo: Use SRBF instead
        if nlen == 1:
            ar = np.ones((self.numcand, 1))
        else:
            ar = (np.random.rand(self.numcand, nlen) < ddsprob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        self.xcand = np.ones((self.numcand, self.data.dim)) * xgap
        scalefactors = sigma * self.xrange
        for i in range(nlen):
            lower, upper = self.data.xlow[i], self.data.xup[i]
            ssigma = scalefactors[i]
            ind = np.where(ar[:, i] == 1)[0]
            self.xcand[ind, i] = stats.truncnorm.rvs(
                (lower - xgap[i]) / ssigma, (upper - xgap[i]) / ssigma,
                loc=xgap[i], scale=ssigma, size=len(ind))

        # Compute surrogate approximations for each objective for each candidate
        self.fhvals = np.zeros((self.numcand, self.data.nobj))
        i = 0
        for fhat in self.fhat:
            fvals = fhat.evals(self.xcand)
            fvals = fvals.flatten()
            self.fhvals[:,i] = fvals
            i = i+1


    def make_points(self, npts, xbest, xfront, front, subset=None, proj_fun=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box
        :type sigma: float
        :param subset: Coordinates to perturb, the others are fixed
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array

        .. todo:: Change the merit function from being hard-coded
        """

        # if subset is None:
        #     subset = np.arange(0, self.data.dim)
        # scalefactors = sigma * self.xrange
        # # Make sure that the scale factors are correct for
        # # the integer variables (at least 1)
        # ind = np.intersect1d(self.data.integer, subset)
        # if len(ind) > 0:
        #     scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

        # ------- GAP OPTIMIZATION ----------------------------------

        # Find most crowded point
        xgap = xbest
        # Run Gap Optimization Algorithm to Generate Candidates
        start = time.clock()
        if self.cand_flag is None:
            self.__generate_cand__(xfront, xgap)
        else:
            self.generate_candidates(xgap,sigma=0.2)
        if proj_fun is not None:
            self.xcand = proj_fun(self.xcand)

        # Choose Point for Evaluation, using Hypervolume Contribution OR Epsilon-ND Progress
        if self.epsilons is None:
            (ndf_index, df_index) = ND_Front(np.transpose(self.fhvals))
            self.xcand_nd = self.xcand[ndf_index,:]
            self.fhvals_nd = self.fhvals[ndf_index,:]
            print(self.fhvals_nd.shape)
            rule1 = HyperVolumeSelection(self.data)
        else:
            (nd, dominated, box_dominated) = epsilon_ND_front(np.transpose(self.fhvals), self.epsilons)
            self.xcand_nd = self.xcand[nd,:]
            self.fhvals_nd = self.fhvals[nd,:]
            print(self.fhvals_nd.shape)
            rule1 = EpsilonSelection(self.data, self.epsilons)
        index = rule1.select_points(np.copy(front),np.copy(self.xcand_nd),np.copy(self.fhvals_nd))
        xnew_gap = self.xcand_nd[index,:]
        fh_new_gap = self.fhvals_nd[index,:]
        end = time.clock()
        totalTime = end - start
        print('Gap Optimization Time: ' + str(totalTime))
        # -----------------------------------------------------------

        # Generate candidate points
        start = time.clock()
        self.__generate_cand__(xfront)
        if proj_fun is not None:
            self.xcand = proj_fun(self.xcand)
        end = time.clock()
        totalTime = end - start
        print('CAndidate Generation Time: ' + str(totalTime))

        # Candidate Selection
        if self.epsilons is None:
            (ndf_index, df_index) = ND_Front(np.transpose(self.fhvals))
            self.xcand_nd = self.xcand[ndf_index,:]
            self.fhvals_nd = self.fhvals[ndf_index,:]
            print(self.fhvals_nd.shape)
            rule1 = HyperVolumeSelection(self.data)
        else:
            (nd, dominated, box_dominated) = epsilon_ND_front(np.transpose(self.fhvals), self.epsilons)
            self.xcand_nd = self.xcand[nd,:]
            self.fhvals_nd = self.fhvals[nd,:]
            rule1 = EpsilonSelection(self.data, self.epsilons)

        index = rule1.select_points(np.copy(front),np.copy(self.xcand_nd),np.copy(self.fhvals_nd))
        rule2 = OspaceDistanceSelection(self.data)
        index = rule2.select_points(np.copy(self.xcand_nd),np.copy(self.fhvals_nd),np.copy(front), index)
        rule3 = DspaceDistanceSelection(self.data)
        index = rule3.select_points(np.copy(self.xcand_nd),np.copy(self.previous_points), index)
        #index = np.random.randint(len(self.xcand))
        xnew = self.xcand_nd[index,:]
        fh_new = self.fhvals_nd[index,:]
        xnew = np.vstack((np.asmatrix(xnew), np.asmatrix(xnew_gap)))
        fh_new = np.vstack((np.asmatrix(fh_new), np.asmatrix(fh_new_gap)))

        # Use Gap Optimization to Choose 4th Point


        # update list of proposed and previous points
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asmatrix(xnew)))
        self.previous_points = np.vstack((self.previous_points,
                                          np.asmatrix(xnew)))

        return xnew, fh_new, self.fhvals_nd
