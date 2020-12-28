from gomors_sync_strategies import MoSyncStrategyNoConstraints
from gomors_adaptive_sampling import *
from test_problems import *
from pySOT import SymmetricLatinHypercube, RBFInterpolant, CubicKernel, \
    LinearTail
from poap.controller import SerialController, ThreadController, BasicWorkerThread
from archiving_strategies import NonDominatedArchive, EpsilonArchive
import numpy as np
import os.path
import logging
from Townbrook import *
from Cville import *

def main():
    pnames = ['Canflownse']
    data_list = [CANFLOWNSE()]
    epsilons = []
    #epsilon = [1000,1000]
    #epsilons.append(epsilon)
    epsilon =[0.0001, 0.0001, 0.0001]
    epsilons.append(epsilon)
    num_trials = 8
    nthreads = 4
    i = 0
    for data in data_list:
        pname = pnames[i]
        epsilon = epsilons[i]
        maxeval = 1000
        experiment(pname, data, nthreads, maxeval, num_trials, epsilon)
        i=i+1


def experiment(pname, data, nthreads, maxeval, num_trials, epsilon):
    print('Problem being solved: ' + pname)
    print('Number of Threads: ' + str(nthreads))
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/" + pname + '_' + str(data.dim) + '_' + str(nthreads) + ".log"):
        os.remove("./logfiles/" + pname + '_' + str(data.dim) + '_' + str(nthreads) + ".log")
    logging.basicConfig(filename="./logfiles/" + pname + '_' + str(data.dim) + '_' + str(nthreads) + ".log",
                        level=logging.INFO)
    for i in range(num_trials):
        optimization_trial(pname, data, epsilon, nthreads, maxeval, i+1)

def optimization_trial(pname, data, epsilon, nthreads, maxeval, num):
    nsamples = nthreads
    print("Trial Number:" + str(num))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        MoSyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval),
            sampling_method=EvolutionaryAlgorithm(data,epsilons=epsilon, cand_flag=1),
            archiving_method=EpsilonArchive(size_max=200,epsilon=epsilon))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    def merit(r):
        return r.value[0]
    result = controller.run(merit=merit)

    # Save results to File
    X = np.loadtxt('final.txt')
    controller.strategy.save_plot(num)
    fname = pname + '_' + str(data.dim) + '_EGOMORS_'  + str(maxeval) + '_'  + str(num) + '_' + str(nthreads) + '.txt'
    np.savetxt(fname, X)


if __name__ == '__main__':
    main()
