"""
.. module:: selection_rules
   :synopsis: Acquisition functions / merit rules for selecting new points from candidates

.. moduleauthor:: Taimoor Akhtar <erita@nus.edu.sg>

"""

import scipy.stats as stats
import types
from mo_utils import *
import random
from hv import HyperVolume
import numpy as np

class MultiRuleSelection(object):
    """ This is a multi-rule selection methodology for cycling
        between different rules.
    """
    def __init__(self, rule_list, cycle=None):
        if cycle is None:
            cycle = range(len(rule_list))
        if (not all(isinstance(i, int) for i in cycle)) or \
                np.min(cycle) < 0 or \
                np.max(cycle) > len(rule_list)-1:
            raise ValueError("Incorrect cycle!!")
        self.selection_rules = rule_list
        self.nrules = len(rule_list)
        self.cycle = cycle
        self.current_rule= 0
        self.current_iter = 0
        self.data = rule_list[0].data

    def select_points(self, npts, xcand_nd, fhvals_nd, front, proposed_points, fvals):

        new_points = np.zeros((npts, self.data.dim))

        # Figure out what we need to generate
        npoints = np.zeros((self.nrules,), dtype=int)
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


class HyperVolumeSelection(object):
    """ This is the rule for hypervolume based selection of new points
    """
    def __init__(self, data, npts=1):
        """
        :param data:
        :param npts:
        """
        self.data = data
        self.npts = npts

    def select_points(self, front, xcand_nd, fhvals_nd, indices=None):

        # Use hypervolume contribution to select the next best
        # Step 1 - Normalize Objectives
        (M, l) = xcand_nd.shape
        temp_all = np.vstack((fhvals_nd, front))
        minpt = np.zeros(self.data.nobj)
        maxpt = np.zeros(self.data.nobj)
        for i in range(self.data.nobj):
            minpt[i] = np.min(temp_all[:,i])
            maxpt[i] = np.max(temp_all[:,i])
        normalized_front = np.asarray(normalize_objectives(front, minpt, maxpt))
        (N, temp) = normalized_front.shape
        normalized_cand_fh = np.asarray(normalize_objectives(fhvals_nd.tolist(), minpt, maxpt))

        # Step 2 - Make sure points already selected are not included in new points list
        if indices is not None:
            nd = range(N)
            dominated = []
            for index in indices:
                fvals = np.vstack((normalized_front, normalized_cand_fh[index,:]))
                (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
            normalized_front = fvals[nd,:]
            N = len(nd)

        # Step 3 - Compute Hypervolume Contribution
        hv = HyperVolume(1.1*np.ones(self.data.nobj))
        xnew = np.zeros((self.npts, l))
        if indices is None:
            indices = []
        hv_vals = -1*np.ones(M)
        hv_vals[indices] = -2
        for j in range(self.npts):
            # 3.1 - Find point with best HV improvement
            base_hv = hv.compute(normalized_front)
            for i in range(M):
                if hv_vals[i] != 0 and hv_vals[i] != -2:
                    nd = range(N)
                    dominated = []
                    fvals = np.vstack((normalized_front, normalized_cand_fh[i,:]))
                    (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
                    if dominated and dominated[0] == N: # Record is dominated
                        hv_vals[i] = 0
                    else:
                        new_hv = hv.compute(fvals[nd,:])
                        hv_vals[i] = new_hv - base_hv
            # vals = np.zeros((M,2))
            # vals[:,0] = xcand_nd[:,0]
            # vals[:,1] = hv_vals
            # print(vals)
            # 3.2 - Update selected candidate list
            index = np.argmax(hv_vals)
            xnew[j,:] = xcand_nd[index,:]
            indices.append(index)
            # 3.3 - Bar point from future selection and update non-dominated set
            hv_vals[index] = -2
            nd = range(N)
            dominated = []
            fvals = np.vstack((normalized_front, normalized_cand_fh[index,:]))
            (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
            normalized_front = fvals[nd,:]
            N = len(nd)
        return indices


class DspaceDistanceSelection(object):
    """
    Implementation of the Decision-Space Selection
    Rule in GOMORS that chooses new points based
    on max-min decision space distance from
    evaluated points
    """
    def __init__(self, data, npts=1):
        """
        :param data:
        :param npts:
        """
        self.data = data
        self.npts = npts

    def select_points(self, xcand_nd, proposed_points, indices=None):
        if indices is not None:
            selected_points = np.vstack((proposed_points, xcand_nd[indices,:]))
        else:
            selected_points = np.copy(proposed_points)
        xnew = np.zeros((self.npts, self.data.dim))
        for i in range(self.npts):
            dists = scp.distance.cdist(xcand_nd, selected_points)
            dmerit = np.amin(np.asmatrix(dists), axis=1)
            if indices is not None:
                dmerit[indices] = -1
            index = np.argmax(dmerit)
            if indices is None:
                indices = []
            indices.append(index)
            xnew[i,:] = xcand_nd[index,:]
            selected_points = np.vstack((selected_points, xnew[i,:]))
        return indices


class OspaceDistanceSelection(object):
    """
    Implementation of the Objective-Space Selection
    Rule in GOMORS that chooses new points based
    on max-min approximate obj space distance from
    evaluated points
    """
    def __init__(self, data, npts=1):
        """
        :param data:
        :param npts:
        """
        self.data = data
        self.npts = npts

    def select_points(self, xcand_nd, fhvals_nd, fvals, indices=None):

        # Step 1 - Normalize Objectives
        (M, l) = xcand_nd.shape
        temp_all = np.vstack((fhvals_nd, fvals))
        minpt = np.zeros(self.data.nobj)
        maxpt = np.zeros(self.data.nobj)
        for i in range(self.data.nobj):
            minpt[i] = np.min(temp_all[:,i])
            maxpt[i] = np.max(temp_all[:,i])
        normalized_fvals = np.asarray(normalize_objectives(fvals, minpt, maxpt))
        (N, l) = normalized_fvals.shape
        normalized_cand_fh = np.asarray(normalize_objectives(fhvals_nd.tolist(), minpt, maxpt))

        # Step 2 - Make sure points already selected are not included in new points list
        if indices is not None:
            selected_fvals = np.vstack((normalized_fvals, normalized_cand_fh[indices,:]))
        else:
            selected_fvals = np.copy(normalized_fvals)

        # Step 3 - Find point(s) with max-min distance in objective space
        dists = scp.distance.cdist(normalized_cand_fh, selected_fvals)
        dmerit = np.amin(np.asmatrix(dists), axis=1)
        xnew = np.zeros((self.npts, self.data.dim))
        for i in range(self.npts):
            if indices is not None:
                dmerit[indices] = -1
            index = np.argmax(dmerit)
            if indices is None:
                indices = []
            indices.append(index)
            xnew[i,:] = xcand_nd[index,:]
            selected_fvals = np.vstack((selected_fvals, normalized_cand_fh[index,:]))
        return indices

class EpsilonSelection(object):
    """ This is the rule for epsilon-progress based selection of new points
    """
    def __init__(self, data, epsilon, npts=1):
        """
        :param data:
        :param npts:
        """
        self.data = data
        self.npts = npts
        self.epsilon = epsilon

    def select_points(self, front, xcand_nd, fhvals_nd, indices=None):

        # Randomly select a point from points with epsilon progress
        (M, l) = xcand_nd.shape
        (N, l) = front.shape
        # Step 1 - Add older points already selected to the eps_front
        if indices is not None:
            ndf_index = range(N)
            df_index = []
            box_index = []
            for index in indices:
                fvals = np.vstack((front, fhvals_nd[index,:]))
                (ndf_index, df_index, box_index, F_box) = epsilon_ND_Add(np.transpose(fvals), df_index, ndf_index, box_index, self.epsilon)
            front = fvals[ndf_index,:]
            N = len(ndf_index)

        # Step 2 - Check if there is Epsilon Progress and add those points to a list
        xnew = np.zeros((self.npts, l))
        if indices is None:
            indices = []
        ep_indices = []
        for i in range(M):
            if i not in indices:
                nd = range(N)
                dominated = []
                box_dominated = []
                fvals = np.vstack((front, fhvals_nd[i,:]))
                (nd, dominated, box_dominated, F_box) = epsilon_ND_Add(np.transpose(fvals), dominated, nd, box_dominated, self.epsilon)
                if dominated == [] and box_dominated == []: # Record is in new box on front and all other records are also non-dominated
                    ep_indices.append(i)
                elif len(dominated)>0 and dominated[0] != N: # Record is Not this front
                    ep_indices.append(i)

        for j in range(self.npts):
            if ep_indices != []:
                index = random.randint(0,len(ep_indices)-1)
                indices.append(ep_indices[index])
                ep_indices.remove(ep_indices[index])
            else:
                index = random.randint(0,M-1)
                while index in indices:
                    index = random.randint(0,M-1)
                indices.append(index)
        return indices
