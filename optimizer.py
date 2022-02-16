## Source code from EvoloPy package

import Optimizers.GWO as gwo
import Optimizers.AGWO as agwo
import functions as benchmarks 
import csv
import numpy as np 
import time
import warnings
import os

warnings.simplefilter(action="ignore")

def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]
    print(algo)

    if algo == "GWO":
        x = gwo.GWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "AGWO":
        x = agwo.AGWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    else:
        return null
    return x

def run(optimizer, objectivefunc, NumOfRuns, params, export_flags):
    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)

    Returns
    -----------
    N/A
    """
    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]
    # Export results ?
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Flag = False
    Flag_details = False

    # CSV Header for the convergence
    CnvgHeader = []
    results_directory = "Results/"
    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(objectivefunc)):
            convergence = [0] * NumOfRuns
            stopiter = [0] * NumOfRuns
            for k in range(0, NumOfRuns):
                func_details = benchmarks.getFunctionDetails(objectivefunc[j])
                x = selector(optimizer[i], func_details, PopulationSize, Iterations)
                convergence[k] = x.convergence
                optimizerName = x.optimizer
                objfname = x.objfname
                ###### FOR GWO , AGWO: 
                stopiter[k] = x.stopiter
                if Export_details == True:
                    ExportToFile = results_directory + "experiment_details.csv"
                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag_details == False
                        ):  # just one time to write the header of the CSV file
                            header = np.concatenate(
                                [["Optimizer", "objfname","STOP Iteration"], CnvgHeader]
                            )
                            writer.writerow(header)
                            Flag_details = True  # at least one experiment
                        a = np.concatenate(
                            [[x.optimizer, x.objfname, x.stopiter], x.convergence]
                        )
                        writer.writerow(a)
                    out.close()

            if Export == True:
                ExportToFile = results_directory + "experiment.csv"
                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        Flag == False
                    ):  # just one time to write the header of the CSV file
                        header = np.concatenate(
                            [["Optimizer", "objfname","STOP Iteration"], CnvgHeader]
                        )
                        writer.writerow(header)
                        Flag = True
                    avgstopiter = float("%0.2f" % (sum(stopiter) / NumOfRuns))
                    avgConvergence = np.around(
                        np.mean(convergence, axis=0, dtype=np.float64), decimals=2
                    ).tolist()
                    a = np.concatenate(
                        [[optimizerName, objfname, avgstopiter], avgConvergence]
                    )
                    writer.writerow(a)
                out.close()
