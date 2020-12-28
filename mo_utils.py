"""
.. module:: metric_tools
   :overview: Includes methods for calculating performance metrics / indicators for MO analysis

.. moduleauthor:: Taimoor Akhtar <taimoor.akhtar@gmail.com>,

"""

import numpy as np
import scipy.spatial as scp


def reduce_bounds(F, bound):
    (M, l) = F.shape
    F_new = []
    for i in range(M):
        if all(np.greater_equal(bound, F[i,:])):
            F_new.append(F[i,:])

    if F_new:
        return np.asarray(F_new)

def nd_sorting(F):
    (M, l) = F.shape
    nd_ranks = np.ones((l,), dtype=np.int)
    P = np.ones((l,), dtype=np.int)
    for i in range(0,l):
        P[i] = i
    i=1
    while len(P) > 0:
        (ndf_index, df_index) = ND_Front(F[:,P])
        for j in range(0,len(ndf_index)):
            nd_ranks[P[ndf_index[j]]] = i
        P_new = np.ones((len(df_index),), dtype=np.int)
        for j in range(0,len(df_index)):
            P_new[j] = P[df_index[j]]
        P = P_new
        i = i+1
    return nd_ranks

def ND_Front(F):
    (M, l) = F.shape
    df_index = []
    ndf_index = [int(0)]
    for i in range(1, l):
        (ndf_index, df_index) = ND_Add(F[:,0:i+1], df_index, ndf_index)
    return (ndf_index, df_index)

def ND_Add(F, df_index, ndf_index):
    (M, l) = F.shape
    l = int(l - 1)
    ndf_count = len(ndf_index)
    ndf_index.append(l)
    ndf_count += 1
    j = 1
    while j < ndf_count:
        if domination(F[:,l],F[:,ndf_index[j-1]],M):
            df_index.append(ndf_index[j-1])
            ndf_index.remove(ndf_index[j-1])
            ndf_count -= 1
        elif domination(F[:,ndf_index[j-1]],F[:,l],M):
            df_index.append(l)
            ndf_index.remove(l)
            ndf_count -= 1
            break
        else:
            j += 1
    return (ndf_index, df_index)

def epsilon_ND_front(F, e):

    M, l = F.shape
    df_index = []
    box_index = []
    ndf_index = [int(0)]
    for i in range(1, l):
        (ndf_index, df_index, box_index, F_box) = epsilon_ND_Add(F[:,0:i+1], df_index, ndf_index, box_index, e)

    return (ndf_index, df_index, box_index)

def epsilon_ND_Add(F, df_index, ndf_index, box_index, e):
    (M, l) = F.shape
    l = int(l - 1)
    ndf_count = len(ndf_index)
    ndf_index.append(l)
    ndf_count += 1
    j = 1
    F_box = np.transpose(compute_epsilon_precision(np.transpose(F), e))
    while(j < ndf_count):
            if domination(F_box[:, l], F_box[:, ndf_index[j - 1]], M):
                df_index.append(ndf_index[j - 1])
                ndf_index.remove(ndf_index[j-1])
                ndf_count = ndf_count - 1
            elif domination(F_box[:,ndf_index[j - 1]], F_box[:, l], M):
                df_index.append(l)
                ndf_index.remove(l)
                ndf_count = ndf_count - 1
                break
            elif np.array_equal(F_box[:, l], F_box[:, ndf_index[j - 1]]):
                d1 = np.linalg.norm((F[:, l] - F_box[:, l]) / e)
                d2 = np.linalg.norm((F[:, ndf_index[j - 1]] - F_box[:, l]) / e)
                if(d1 < d2):
                    box_index.append(ndf_index[j - 1])
                    ndf_index.remove(ndf_index[j - 1])
                    ndf_count = ndf_count - 1
                else:
                    box_index.append(l)
                    ndf_index.remove(l)
                    ndf_count = ndf_count - 1
                    break
            else:
                j = j + 1
    return (ndf_index, df_index, box_index, F_box[:, ndf_index])

def compute_epsilon_precision(F, e):
# This function comnputes epsilon precise values of all elements in F
    M, l = F.shape
    F_box = np.multiply(np.floor(F / (e * np.ones(l))), (e * np.ones(l)))
    return F_box

def domination(fA, fB, M):
    d = False
    for i in range(0,M):
        if fA[i] > fB[i]:
            d = False
            break
        elif fA[i] < fB[i]:
            d = True
    return d

def weakly_dominates(fA, fB, M):
    d = False
    for i in range(0,M):
        if fA[i] > fB[i]:
            d = False
            break
        elif fA[i] <= fB[i]:
            d = True
    return d

def front_3d(front, min_point, bound):
    M, nobj = front.shape

    nsamples = 100000
    precision = 0.005
    samples = np.random.rand(nsamples, nobj)
    eps = np.zeros(nobj)
    for i in range(nobj):
        samples[:,i] = np.ones(nsamples)*min_point[i] + (bound[i] - min_point[i])*samples[:,i]
        eps[i] = precision*(bound[i] - min_point[i])

    front_surf = []
    for i in range(nsamples):
        curPt = samples[i,:]
        j = 0
        check = 1
        while j < M:
            if domination(front[j,:], curPt, nobj):
                check = 0
                j = M
            else:
                j=j+1

        if check == 1:
            final_check = 1
            j=0
            curPt = curPt + eps
            while j < M:
                if domination(front[j,:], curPt, nobj):
                    final_check = 0
                    j = M
                else:
                    j=j+1
        else:
            j=0
            final_check = 0
            curPt = curPt - eps
            while j < M:
                if domination(front[j,:], curPt, nobj):
                    j = M
                    final_check = 1
                else:
                    j=j+1

        if final_check == 0:
            front_surf.append(samples[i,:])

    front_surf = np.asarray(front_surf)
    front_surf = np.vstack((front_surf, front))
    return front_surf

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def normalize_objectives(fvals, minpt=None, maxpt=None):
    nobj = len(fvals[0])
    if maxpt is None:
        maxpt = [max([rec[i] for rec in fvals]) for i in range(nobj)]
    if minpt is None:
        minpt = [min([rec[i] for rec in fvals]) for i in range(nobj)]
    normalized_fvals = []
    for item in fvals:
       normalized_fvals.append([(item[i] - minpt[i]) / (maxpt[i] - minpt[i]) if (maxpt[i] - minpt[i]) > 0 else 0 for i in range(nobj)])
    return normalized_fvals


def radius_rule(rec, center_pts, d_thresh):
    flag = True
    if center_pts == []:
        flag = True
    else:
        X_c = np.asarray([record.x for record in center_pts])
        sigmas = [record.sigma for record in center_pts]
        nc = len(center_pts)
        X = np.asarray(rec.x)
        for i in range(nc):
            d = scp.distance.euclidean(X,X_c[i,:]) # Todo - Divide by Square Root of Dim
            if d < sigmas[i]*d_thresh/np.sqrt(len(X)):   #/np.sqrt(len(X))
                flag = False
                break
    return flag
