
from optimizer import run
import os

#### Make results directory if it does not exist
directory = './Results/'
if not os.path.exists(directory):
    os.makedirs(directory)
######## Define Optimizers 
######## Algorithms from EvoloPy package can be used. Other arbitrary algorithms can also be used. 
#           To run the MAB-OS or Random bandit setting, we can use "BANDIT" and "RANDOM" algorithms, respectively.
optimizer=["AGWO","GWO"]

######### Objective functions as defined in the functions.py (for the first 23 functions) or arbitrary functions like CEC-2017 functions. 
objectivefunc=["F21"]

######### Number of Runs for statistical analysis of the convergence curves
NumOfRuns=1

######### Parameters of optimization including number of agents (population size), and number of iterations as defined in EvoloPy package
params = {'PopulationSize' : 50, 'Iterations' : 1000}

########### to save results in csv files in the results directory
export_flags = {'Export_avg':True, 'Export_details':True}

########### RUN !
run(optimizer, objectivefunc, NumOfRuns, params, export_flags)