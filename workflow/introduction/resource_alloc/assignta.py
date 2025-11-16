"""
Name: John Rachlin
Date: November 2022
File: assignta.py

Description: Use evolutionary computing framework to assign TAs to Lab Practicums
subject to multiple objectives.

"""

import numpy as np
import pandas as pd
import random as rnd
import evo_v6 as evo
import os
from profiler import profile, Profiler

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))


tas = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'tas.csv'))
sections = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'sections.csv'))


@profile
def overallocation(sol):
    """ penalty for Overallocation of tas to sections """
    diff = sol.sum(axis=1) - tas.max_assigned
    return sum(diff[diff > 0])


@profile
def conflicts(sol):
    """ # TAs with time conflicts """
    num_conflicts = 0
    for ta in range(sol.shape[0]):
        times = sections.daytime[sol[ta] == 1]
        if len(list(times)) > len(set(times)):
            num_conflicts += 1
    return num_conflicts


@profile
def undersupport(sol):
    """ penalty for not assigning enough TAs to sections """
    diff = sections.min_ta - sol.sum(axis=0)
    return sum(diff[diff > 0])


@profile
def unavailable(sol):
    """ # times a ta is assigned to a section they are unwilling to support """
    cantdo = tas.loc[:, '0':] == 'U'
    return int((np.array(cantdo) & sol).sum())


@profile
def unpreferred(sol):
    """ # times a ta is assigned to a section they are willing to support but not preferred """
    willdo = tas.loc[:, '0':] == 'W'
    return int((np.array(willdo) & sol).sum())


@profile
def mutate(sols):
    """ Toggle a random TA/section assignment """
    sol = sols[0]
    shape = sol.shape
    i = rnd.randrange(shape[0])
    j = rnd.randrange(shape[1])
    sol[i, j] = 1 - sol[i, j]
    return sol


@profile
def crossover(sols):
    """ Extract a random area of one solution and overlay it upon another """
    solA = sols[0]
    solB = sols[1]
    rows, cols = solA.shape
    imin = rnd.randrange(rows)
    imax = rnd.randrange(imin, rows)
    jmin = rnd.randrange(cols)
    jmax = rnd.randrange(jmin, cols)
    solA[imin:imax, jmin:jmax] = solB[imin:imax, jmin:jmax]
    return solA


@profile
def overlay(sols):
    """ Assign only where two input solutions both have an assignment """
    return sols[0] & sols[1]


@profile
def fix_unavailable(sols):
    """ Remove any assignments where TA is unwilling """
    cando = tas.loc[:, '0':] != 'U'
    return np.array(cando) & sols[0]


def main():

    E = evo.Environment()

    # register objectives
    E.add_fitness_criteria("overallocation", overallocation)
    E.add_fitness_criteria("conflicts", conflicts)
    E.add_fitness_criteria("undersupport", undersupport)
    E.add_fitness_criteria("unavailable", unavailable)
    E.add_fitness_criteria("unpreferred", unpreferred)

    # register agents
    E.add_agent("mutate", mutate, 1)
    E.add_agent("crossover", crossover, 2)
    E.add_agent("overlay", overlay, 2)
    E.add_agent("fix_unwilling", fix_unavailable, 1)

    # Add 10 initial solutions + empty + full
    # Create initial values
    empty = np.zeros((len(tas), len(sections)), dtype=int)
    full = np.ones((len(tas), len(sections)), dtype=int)

    E.add_solution(empty)
    E.add_solution(full)
    for _ in range(10):
        rand = (np.random.rand(len(tas), len(sections)) > 0.9) * 1
        E.add_solution(rand)

    # Run the optimizer
    E.evolve(10**9, dom=100, status=2000, time_limit=300, reset=True)

    E.summarize(source='rachlin', with_details=False)

    Profiler.report()


if __name__ == '__main__':
    main()
