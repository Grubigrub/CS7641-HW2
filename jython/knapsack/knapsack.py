from __future__ import with_statement
import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array

from array import array

def list_sum_average(*args):
    output = []
    for i in range(len(args[0])):
        temp = 0
        for j in range(len(args)):
            temp += args[j][i]
        output.append(temp / len(args))
    return output


"""
Commandline parameter(s):
   none
"""

OUTPUT_FILE = '../../output/ks_out.csv'

REPEAT = 10
MIN_ITERATION = 100
MAX_ITERATION = 50000
ITERATION_STEP = 50

SA_TEMPERATURE = 1000
SA_COOLING_FACTOR = .95

GA_POPULATION = 200
GA_CROSSOVER = 150
GA_MUTATION = 25

MIMIC_SAMPLES = 200
MIMIC_TO_KEEP = 10

# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 40
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack 
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

rhc_accuracy = [[] for i in range(REPEAT)]
rhc_train_time = [[] for i in range(REPEAT)]

sa_accuracy = [[] for i in range(REPEAT)]
sa_train_time = [[] for i in range(REPEAT)]

ga_accuracy = [[] for i in range(REPEAT)]
ga_train_time = [[] for i in range(REPEAT)]

mimic_accuracy = [[] for i in range(REPEAT)]
mimic_train_time = [[] for i in range(REPEAT)]

last_train_time_rhc = 0
last_train_time_sa = 0
last_train_time_ga = 0
last_train_time_mimic = 0

iterations_count = [MIN_ITERATION + i * ITERATION_STEP for i in range((MAX_ITERATION - MIN_ITERATION) / ITERATION_STEP + 1)]

for repetition in range(REPEAT):
    print("Repetion %d" % (repetition + 1))
    current_iteration_count = MIN_ITERATION

    fill = [0] * NUM_ITEMS
    weights = array('d', fill)
    volumes = array('d', fill)
    for i in range(0, NUM_ITEMS):
        weights[i] = random.nextDouble() * MAX_WEIGHT
        volumes[i] = random.nextDouble() * MAX_VOLUME

    fill = [COPIES_EACH + 1] * NUM_ITEMS
    ranges = array('i', fill)

    # Problem Definition

    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = UniformCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    # Algorithm declaration
    rhc = RandomizedHillClimbing(hcp)
    sa = SimulatedAnnealing(SA_TEMPERATURE, SA_COOLING_FACTOR, hcp)
    ga = StandardGeneticAlgorithm(GA_POPULATION, GA_CROSSOVER, GA_MUTATION, gap)
    mimic = MIMIC(MIMIC_SAMPLES, MIMIC_TO_KEEP, pop)

    # Trainer declaration
    fit_rhc = FixedIterationTrainer(rhc, current_iteration_count)
    fit_sa = FixedIterationTrainer(sa, current_iteration_count)
    fit_ga = FixedIterationTrainer(ga, current_iteration_count)
    fit_mimic = FixedIterationTrainer(mimic, current_iteration_count)

    print("Computing for %d iterations" % current_iteration_count)

    # Fitting
    start_rhc = time.time()
    fit_rhc.train()
    end_rhc = time.time()

    start_sa = time.time()
    fit_sa.train()
    end_sa = time.time()
    
    start_ga = time.time()
    fit_ga.train()
    end_ga = time.time()
    
    start_mimic = time.time()
    fit_mimic.train()
    end_mimic = time.time()

    # Result handling
    last_train_time_rhc = end_rhc - start_rhc
    rhc_train_time[repetition].append(last_train_time_rhc)
    rhc_accuracy[repetition].append(ef.value(rhc.getOptimal()))

    last_train_time_sa = end_sa - start_sa
    sa_train_time[repetition].append(last_train_time_sa)
    sa_accuracy[repetition].append(ef.value(sa.getOptimal()))

    last_train_time_ga = end_ga - start_ga
    ga_train_time[repetition].append(last_train_time_ga)
    ga_accuracy[repetition].append(ef.value(ga.getOptimal()))

    last_train_time_mimic = end_mimic - start_mimic
    mimic_train_time[repetition].append(last_train_time_mimic)
    mimic_accuracy[repetition].append(ef.value(mimic.getOptimal()))

    while current_iteration_count <= MAX_ITERATION - ITERATION_STEP:
        print("Computing for %d iterations" % (current_iteration_count + ITERATION_STEP))
        # Trainer declaration
        fit_rhc = FixedIterationTrainer(rhc, ITERATION_STEP)
        fit_sa = FixedIterationTrainer(sa, ITERATION_STEP)
        fit_ga = FixedIterationTrainer(ga, ITERATION_STEP)
        fit_mimic = FixedIterationTrainer(mimic, ITERATION_STEP)

        # Fitting
        start_rhc = time.time()
        fit_rhc.train()
        end_rhc = time.time()

        start_sa = time.time()
        fit_sa.train()
        end_sa = time.time()
        
        start_ga = time.time()
        fit_ga.train()
        end_ga = time.time()
        
        start_mimic = time.time()
        fit_mimic.train()
        end_mimic = time.time()

        # Result handling
        last_train_time_rhc = last_train_time_rhc + end_rhc - start_rhc
        rhc_train_time[repetition].append(last_train_time_rhc)
        rhc_accuracy[repetition].append(ef.value(rhc.getOptimal()))

        last_train_time_sa = last_train_time_sa + end_sa - start_sa
        sa_train_time[repetition].append(last_train_time_sa)
        sa_accuracy[repetition].append(ef.value(sa.getOptimal()))

        last_train_time_ga = last_train_time_ga + end_ga - start_ga
        ga_train_time[repetition].append(last_train_time_ga)
        ga_accuracy[repetition].append(ef.value(ga.getOptimal()))

        last_train_time_mimic = last_train_time_mimic + end_mimic - start_mimic
        mimic_train_time[repetition].append(last_train_time_mimic)
        mimic_accuracy[repetition].append(ef.value(mimic.getOptimal()))

        current_iteration_count += ITERATION_STEP


final_rhc_train_time = list_sum_average(*rhc_train_time)
final_rhc_accuracy = list_sum_average(*rhc_accuracy)

final_sa_train_time = list_sum_average(*sa_train_time)
final_sa_accuracy = list_sum_average(*sa_accuracy)

final_ga_train_time = list_sum_average(*ga_train_time)
final_ga_accuracy = list_sum_average(*ga_accuracy)

final_mimic_train_time = list_sum_average(*mimic_train_time)
final_mimic_accuracy = list_sum_average(*mimic_accuracy)



with open(OUTPUT_FILE, "w") as outFile:
    for i in range(len(iterations_count)):
        outFile.write(','.join([
            str(iterations_count[i]),
            str(final_rhc_accuracy[i]),
            str(final_rhc_train_time[i]),
            str(final_sa_accuracy[i]),
            str(final_sa_train_time[i]),
            str(final_ga_accuracy[i]),
            str(final_ga_train_time[i]),
            str(final_mimic_accuracy[i]),
            str(final_mimic_train_time[i])]) + '\n')
