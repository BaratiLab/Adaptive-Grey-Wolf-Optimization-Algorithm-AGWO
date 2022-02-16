import random
import numpy as np
import math
from solution import solution
import time
import matplotlib.pyplot as plt

def AGWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    Prev_Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = np.zeros(Max_iter)
    s = solution()
    print('AGWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    dF_avg_list = []
    F_avg_list = []
    a_list = []
    damper = 1
    # Main loop
    for l in range(0, Max_iter):
        dF = []
        F_all = []
        for i in range(0, SearchAgents_no):
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])
            if l ==0:
                dF_avg = float('-inf')
            if l != 0:
                Prev_fitness = objf(Prev_Positions[i, :])
                dF.append(fitness - Prev_fitness)
                F_all.append(fitness)
            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        Convergence_curve[l] = Alpha_score

        ########################### ADAPTIVE GWO #####################################################
        ### Epsilon parameter (Equation 4 in paper)
        eps = 1e-5
        ### Damping parameter (Algorithm 1 in paper)
        gamma = 0.95 
        ### Minimum initial steps (Full Exploration Time)
        min_steps = 10
        ### delta (threshold in paper )
        threshold = 1e-3 

        ##### Tune the damper variable with gamma
        if l>min_steps and F_avg >= (mov_avg_F - eps):
            damper = damper * gamma
        
        #### Store function values and changes 
        if l != 0:
            dF_avg = np.mean(np.abs(dF))
            dF_avg_list.append(dF_avg)
            F_avg = np.mean(F_all)
            F_avg_list.append(F_avg)

        #### Full Exploration
        if l <= min_steps:
            a = 2
        #### Control parameter 'a' with AGWO damper
        Delta = False ### Using Extended Delta Version if True
        if l > min_steps: 
            if Delta == False:
                ### AGWO with epsilon (Damper only)
                a = damper * 2
            else:
                ## AGWO Delta (Extended version in the paper)
                a = 0.9*a + 0.1 * damper * 2 * (dF_avg)/(np.sqrt(E_g2))
                a = np.clip(a ,0,2)
        
        if Delta == True:
            if l ==0 :
                E_g2 = 0
            else: 
                E_g2 =  (dF_avg)**2

        #### Moving Average (Sliding window for objective function values)
        mov_avg_dF = np.mean(dF_avg_list[-10:])  # Exclude -inf at 0
        mov_avg_F = np.mean(F_avg_list[-10:])  # Exclude -inf at 0

        ###### Stopping criteria
        if (l>min_steps and a < threshold ) and F_avg >= mov_avg_F -eps:
            break

        a_list.append(a)
        ############################################################################################   

        # Update the Position of search agents including omegas (GWO Original implemented as EvoloPy)
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta
                Prev_Positions[i,j]= Positions[i,j]
                Positions[i, j] = (X1 + X2 + X3) / 3  

        if l % 100 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)]
            )

    print("Damper: ", damper)
    timerEnd = time.time()
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "AGWO"
    s.objfname = objf.__name__
    s.stopiter = l 
    s.dF_list = dF_avg_list
    s.F_list = F_avg_list 
    s.a_list = a_list

    return s
