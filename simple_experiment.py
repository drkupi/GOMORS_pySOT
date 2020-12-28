
from gomors_sync_strategies import MoSyncStrategyNoConstraints
from gomors_adaptive_sampling import EvolutionaryAlgorithm
from test_problems import *
from pySOT import SymmetricLatinHypercube, RBFInterpolant, CubicKernel, LinearTail
from poap.controller import SerialController, ThreadController, BasicWorkerThread
from archiving_strategies import NonDominatedArchive, EpsilonArchive
import numpy as np
import os.path
import logging
#from Townbrook import *


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_simple.log"):
        os.remove("./logfiles/test_simple.log")
    logging.basicConfig(filename="./logfiles/test_simple.log",
                        level=logging.INFO)

    nthreads = 4
    maxeval = 100
    nsamples = nthreads

    print("\nNumber of threads: " + str(nthreads))
    print("Maximum number of evaluations: " + str(maxeval))
    print("Sampling method: Mixed")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    #data = LZF3()
    data = DTLZ4(nobj=2)
    num = 1
    epsilons = [0.05, 0.05]
    # Create a strategy and a controller
    controller = ThreadController()
    #controller = SerialController(data.objfunction)
    controller.strategy = \
        MoSyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval),
            sampling_method=EvolutionaryAlgorithm(data,epsilons=epsilons, cand_flag=1), archiving_method=EpsilonArchive(size_max=200,epsilon=epsilons))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    def merit(r):
        return r.value[0]
    result = controller.run(merit=merit)

    controller.strategy.save_plot(num)

if __name__ == '__main__':
    main()
