##### Source code from: EvoloPy package 

import random
import numpy
import math
from solution import solution
import time
import matplotlib.pyplot as plt

def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    Prev_Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    print('GWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])

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

        a = 2 - l * ((2) / Max_iter)
        # a decreases linearly from 2 to 0

        # Update the Position of search agents including omegas
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

    timerEnd = time.time()
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.objfname = objf.__name__
    s.stopiter = l

    return s
