"""
.. module:: archiving_methods
   :synopsis: Parallel synchronous MO optimization strategy - GOMORS

.. moduleauthor:: David Bindel <bindel@cornell.edu>,
                David Eriksson <dme65@cornell.edu>,
                Taimoor Akhtar <erita@nus.edu.sg>

"""

from __future__ import print_function
import numpy as np
import math
from mo_utils import *
from hv import HyperVolume
from copy import deepcopy
from matplotlib import pyplot as plt
import random

POSITIVE_INFINITY = float("inf")

class MemoryRecord():
    "Record that Represents Memory of Optimization Progress Attained Around this Center Point"

    def __init__(self, x, fx, sigma=0.2, nfail=0, ntabu=0, rank=POSITIVE_INFINITY, fitness=POSITIVE_INFINITY):
        """Initialize the record.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.x = x
        self.fx = fx
        self.nfail = nfail
        self.ntabu = ntabu
        self.rank = rank
        self.fitness = fitness
        self.sigma_init = sigma
        self.sigma = sigma
        self.noffsprings = 1
        self.offsprings = []
        self.fhat_pts = []

    def reset(self):
        self.ntabu = 0
        self.nfail = 0
        self.sigma = self.sigma_init


class MemoryArchive():

    def __init__(self, size_max):
        """Initialize the record.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.contents = []
        self.size_max = size_max
        self.num_records = 0

    def add(self, record, cur_rank=None):
        if cur_rank == None:
            cur_rank = 1
        if self.contents: # if Archive is not Empty
            ranked = False
            while cur_rank <= len(self.contents): # Traverse through all front to find front where record is to be inserted
                front = self.contents[cur_rank-1]
                dominated_records = []
                fvals = [rec.fx for rec in front]
                num_front = len(fvals)
                nd = range(num_front)
                dominated = []
                fvals.append(record.fx)
                fvals = np.asarray(fvals)
                (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
                if dominated == []: # Record is in front and all other records are also non-dominated
                    ranked = True
                    # 1 - Update Add Record to Current Front in Memory Archive
                    record.rank = cur_rank
                    front.append(record)
                    for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                        item.fitness = POSITIVE_INFINITY
                    self.num_records+=1
                    break
                if dominated[0] == num_front: # Record is Not this front
                    fvals = None
                else: # this is the front and it dominates other points already on the front
                    ranked = True
                    # 1 - Update Add Record to Current Front in Memory Archive
                    record.rank = cur_rank
                    front.append(record)
                    self.num_records+=1
                    # 2 - Remove dominated solutions from current front and add them later
                    dominated = sorted(dominated, reverse=True)
                    for i in dominated:
                        dominated_record = deepcopy(front[i])
                        front.remove(front[i])
                        self.num_records-=1
                        self.add(dominated_record,cur_rank)
                    for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                        item.fitness = POSITIVE_INFINITY
                    break
                cur_rank+=1

            if ranked == False:
                record.rank = len(self.contents) + 1
                record.fitness = POSITIVE_INFINITY
                self.contents.append([record])
                self.num_records+=1

        else:
            self.contents.append([record])
            self.num_records+=1
            record.rank = 1
            record.fitness = POSITIVE_INFINITY

        # Make Sure that number of records in archive is less than size_max
        if self.num_records > self.size_max:
            self.contents[-1].remove(self.contents[-1][-1])
            if self.contents[-1] == []:
                self.contents.remove(self.contents[-1])
            self.num_records -=1

    def compute_hv_fitness(self, cur_rank):
        # Step 0 - Obtain fevals of front
        front = deepcopy(self.contents[cur_rank-1])
        nrec = len(front)
        if nrec == 1:
            self.contents[cur_rank-1][0].fitness = 1
        else:
            fvals = [rec.fx for rec in front]
            # Step 1 - Normalize Objectives
            nobj = len(front[0].fx)
            normalized_fvals = normalize_objectives(fvals)
            # Step 2 - Compute Hypervolume Contribution
            hv = HyperVolume(1.1*np.ones(nobj))
            base_hv = hv.compute(np.asarray(normalized_fvals))
            for i in range(nrec):
                fval_without = deepcopy(normalized_fvals)
                fval_without.remove(fval_without[i])
                new_hv = hv.compute(np.asarray(fval_without))
                hv_contrib = base_hv - new_hv
                self.contents[cur_rank-1][i].fitness = hv_contrib

    def select_center_population(self, npts, d_thresh=1.0):
        center_pts = []
        count = 1
        nfronts = len(self.contents)
        cur_rank = 1
        flag_tabu = False  # Only true if all points in archive are tabu
        while count <= npts: # Traverse through Memory Archive to Select Center Population
            front = self.contents[cur_rank-1] # Iterate through fronts
            if front[0].fitness == POSITIVE_INFINITY:
                self.compute_hv_fitness(cur_rank)
            front.sort(key=lambda x: x.fitness, reverse=True)
            for rec in front: # Traverse through sorted front (by fitness)
                if flag_tabu == True: # If we cycled through all fronts and did not get enough pts
                    rec.reset()
                    center_pts.append(rec)
                    count +=1
                    if count > npts:
                        break
                elif rec.ntabu == 0:
                    # Radius Rule Check goes first
                    flag_radius = radius_rule(rec, center_pts, d_thresh)
                    if flag_radius == True:
                        center_pts.append(rec)
                        count +=1
                        if count > npts:
                            break
            cur_rank = int((cur_rank % nfronts) + 1)
            if cur_rank == 1:
                flag_tabu = True
        return center_pts


class NonDominatedArchive():

    def __init__(self, size_max):
        """Initialize the record.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.contents = []
        self.size_max = size_max
        self.num_records = 0

    def add(self, record):
        if self.contents: # if Archive is not Empty
            ranked = False
            front = self.contents
            fvals = [rec.fx for rec in front]
            num_front = len(fvals)
            nd = range(num_front)
            dominated = []
            fvals.append(record.fx)
            fvals = np.asarray(fvals)
            (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
            if dominated == []: # Record is in front and all other records are also non-dominated
                ranked = True
                # 1 - Update Add Record to Current Front in Memory Archive
                record.rank = 1
                front.append(record)
                for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                    item.fitness = POSITIVE_INFINITY
                self.num_records+=1
            elif dominated[0] == num_front: # Record is Not this front
                fvals = None
            else: # this is the front and it dominates other points already on the front
                ranked = True
                # 1 - Update Add Record to Current Front in Memory Archive
                record.rank = 1
                front.append(record)
                self.num_records+=1
                # 2 - Remove dominated solutions from current front and add them later
                dominated = sorted(dominated, reverse=True)
                for i in dominated:
                    dominated_record = deepcopy(front[i])
                    front.remove(front[i])
                    self.num_records-=1
                for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                    item.fitness = POSITIVE_INFINITY

        else:
            self.contents.append(record)
            self.num_records+=1
            record.rank = 1
            record.fitness = POSITIVE_INFINITY

        # Make Sure that number of records in archive is less than size_max
        if self.num_records > self.size_max:
            self.contents.remove(self.contents[-1])
            self.num_records -=1

    def compute_fitness(self):
        # Step 0 - Obtain fevals of front
        front = deepcopy(self.contents)
        nrec = len(front)
        if nrec == 1:
            self.contents[0].fitness = 1
        else:
            fvals = [rec.fx for rec in front]
            nobj = len(front[0].fx)
            # Step 1 - Normalize Objectives
            normalized_fvals = normalize_objectives(fvals)
            # Step 2 - Compute Hypervolume Contribution
            hv = HyperVolume(1.1*np.ones(nobj))
            base_hv = hv.compute(np.asarray(normalized_fvals))
            for i in range(nrec):
                fval_without = deepcopy(normalized_fvals)
                fval_without.remove(fval_without[i])
                new_hv = hv.compute(np.asarray(fval_without))
                hv_contrib = base_hv - new_hv
                self.contents[i].fitness = hv_contrib

class EpsilonArchive():
    "The archiving method that uses the concept of epsilon-box dominance "
    def __init__(self, size_max, epsilon):
        """Initialize the archive.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.contents = []
        self.size_max = size_max
        self.num_records = 0
        self.epsilon = epsilon
        self.F_box = None
        self.improvement = 0

    def add(self, record):
        F_box = None
        self.improvement = 0
        if self.contents: # if Archive is not Empty
            front = self.contents
            fvals = [rec.fx for rec in front]
            num_front = len(fvals)
            nd = range(num_front)
            dominated = []
            box_dominated = []
            fvals.append(record.fx)
            fvals = np.asarray(fvals)
            (nd, dominated, box_dominated, F_box) = epsilon_ND_Add(np.transpose(fvals), dominated, nd, box_dominated, self.epsilon)
            if dominated == [] and box_dominated == []: # Record is in front and all other records are also non-dominated
                # 1 - Update Add Record to Current Front in Memory Archive
                record.rank = 1
                front.append(record)
                for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                    item.fitness = POSITIVE_INFINITY
                self.num_records+=1
                self.improvement = 1
            elif dominated==[] and box_dominated[0] == num_front: # Record is Not this front
                fvals = None
            elif box_dominated==[] and dominated[0]==num_front:
                fvals = None
            else: # this is the front and it dominates other points already on the front
                # 1 - Update Add Record to Current Front in Memory Archive
                record.rank = 1
                front.append(record)
                self.num_records+=1
                self.improvement = 1
                # 2 - Remove dominated solutions from current front and add them later
                dominated = sorted(dominated, reverse=True)
                for i in dominated:
                    front.remove(front[i])
                    self.num_records-=1
                    self.improvement = 1
                # 2 - Remove dominated solutions from current front and add them later
                box_dominated = sorted(box_dominated, reverse=True)
                for i in box_dominated:
                    front.remove(front[i])
                    self.num_records-=1
                for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                    item.fitness = POSITIVE_INFINITY

        else:
            self.contents.append(record)
            self.num_records+=1
            record.rank = 1
            record.fitness = POSITIVE_INFINITY
            self.improvement = 1
        self.F_box = F_box

        # Make Sure that number of records in archive is less than size_max
        if self.num_records > self.size_max:
            self.contents.remove(self.contents[-1])
            self.num_records -=1

    def reset(self):
        self.contents = []
        self.num_records = 0
        self.F_box = None
        self.improvement = 0

    def compute_fitness(self):
        ref_vector = np.zeros(10)
        # Step 0 - Compute Reference Point
        # front = np.copy(self.F_box)
        # ndim, nrec = front.shape
        # ref_vector = np.zeros(ndim)
        # F_step = np.zeros((ndim, nrec))
        # F_check = np.zeros(nrec)
        # for i in range(ndim):
        #     ref_vector[i] = np.max(front[i,:]) + 2*self.epsilon[i]
        # if nrec == 1:
        #     self.contents[0].fitness = 1
        # else:
        #     for j in range(nrec):
        #         for i in range(ndim):
        #             sub_idx = np.where(front[i,:] > front[i,j])[0]
        #             if sub_idx.size == 0:
        #                 F_step[i,j] = ref_vector[i]
        #             else:
        #                 idx = sub_idx[front[i,sub_idx].argmin()]
        #                 F_step[i,j] = front[i,idx]
        #
        #         for k in range(nrec):
        #             if k != j:
        #                 if domination(front[:, k], F_step[:, j], ndim):
        #                     F_check[j] = 1
        #                     break
        #         if F_check[j] == 0:
        #             print(j)
        #             print(front)
        #             print(F_step[:,j])
        # print(F_check)
        # return F_step

    def compute_hv_fitness(self):
        # Step 0 - Obtain fevals of front
        front = deepcopy(self.F_box)
        nobj, nrec = front.shape
        if nrec == 1:
            self.contents[0].fitness = 1
        else:
            fvals = np.transpose(front)
            fvals = fvals.tolist()
            # Step 1 - Normalize Objectives
            normalized_fvals = normalize_objectives(fvals)
            # Step 2 - Compute Hypervolume Contribution
            hv = HyperVolume(1.1*np.ones(nobj))
            base_hv = hv.compute(np.asarray(normalized_fvals))
            for i in range(nrec):
                fval_without = deepcopy(normalized_fvals)
                fval_without.remove(fval_without[i])
                new_hv = hv.compute(np.asarray(fval_without))
                hv_contrib = base_hv - new_hv
                self.contents[i].fitness = hv_contrib


def main():
    """Main test routine"""
    # Generate points for adding to archive
    archive = NonDominatedArchive(200)
    eps_archive = EpsilonArchive(200, [0.05, 0.05])
    for i in range(20):
        xvals = np.random.rand(1,2)
        yvals = []
        yvals.append(np.random.rand())
        if np.random.rand() < 1:
            yvals.append(1 - np.sqrt(yvals[0]) + 0.5*np.random.rand())
        else:
            yvals.append(np.random.rand())
        srec1 = MemoryRecord(xvals[0],np.asarray(yvals))
        srec2 = MemoryRecord(xvals[0],np.asarray(yvals))
        archive.add(srec1)
        eps_archive.add(srec2)
    F_box = eps_archive.F_box
    eps_archive.compute_fitness()
    archive.compute_fitness()
    af = [rec.fitness for rec in archive.contents]
    eaf = [rec.fitness for rec in eps_archive.contents]
    print(random.randint(0,2))
    print(af)
    print(eaf)
    #F_step = eps_archive.compute_fitness()

    # Plot Non-Dominated Points
    front = archive.contents
    front2 = eps_archive.contents
    fvals = [rec.fx for rec in front]
    fvals2 = [rec.fx for rec in front2]
    fvals = np.asarray(fvals)
    fvals2 = np.asarray(fvals2)
    plt.figure(1)
    plt.plot(fvals[:,0], fvals[:,1], 'b*')
    plt.plot(fvals2[:,0], fvals2[:,1], 'gd')
    plt.plot(F_box[0,:], F_box[1,:], 'ro')
    #plt.plot(F_step[0,:], F_step[1,:], 'yp')
    plt.show()


if __name__ == "__main__":
    main()
