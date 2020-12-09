from deap import base, creator
import random
import numpy as np
from deap import tools
import predictor_model as fit_obj
import csv

from sklearn import svm
from scoop import futures
from joblib import dump, load

class PeptideSelectionGA:

    AMINOACIDS = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
 
    def __init__(self, classifier, sequence_len, array_max, array_min, descriptors,fixed_residues):

        self.classifier = classifier
        self.descriptors = descriptors
        self.sequence_len = sequence_len
        self.fixed_residues = fixed_residues

        self.array_max = array_max
        self.array_min = array_min
      
        self.toolbox = self.create_toolbox()
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = []

    def evaluate(self,individual):
        np_ind = np.asarray(individual)

        peptide_sequence = ""
        for index in np_ind:
            peptide_sequence += PeptideSelectionGA.AMINOACIDS[index]
  
        fitness_model = fit_obj.predictor_model(peptide_sequence, self.classifier, self.array_max, self.array_min, self.descriptors)       
        fitness = fitness_model.predictor()
       
        return fitness, #Single-element tuples should be avoided.  use comma.
      
    def aa_index(self):
        return random.randint(0,19)

    def create_toolbox(self):
        creator.create("PeptideSequence", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.PeptideSequence)
        toolbox = base.Toolbox()
        # Structure initializers
        toolbox.register("attr_aa_index", self.aa_index)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_aa_index, self.sequence_len)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxUniform)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=19, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("map", futures.map)
 
        return toolbox
    
    def get_final_scores(self,pop,fits):
        self.final_fitness = list(zip(pop,fits))
        
    def out_put_best(self,pop, num, output_file):
        print("\n-- Only the fittest survive --\n")
        results_dict = {}

        self.best_ind = tools.selBest(pop, len(pop))
        offspring_best = [self.toolbox.clone(ind) for ind in self.best_ind]

        print("-- SEQUENCES OF LENGTH : %d -- \n" % (len(offspring_best[0])))
        output_file.write("-- SEQUENCES OF LENGTH : %d -- \n" % (len(offspring_best[0])))

        for index, individual in zip(range(len(pop)),offspring_best):
            if num > len(results_dict):
                fitness_value = individual.fitness.values
                np_ind = np.asarray(individual)

                optimum_pep_seq = ""

                for index in np_ind:
                    optimum_pep_seq += PeptideSelectionGA.AMINOACIDS[index]

                if not optimum_pep_seq in results_dict:
                    results_dict[optimum_pep_seq] = fitness_value[0]
                    print("Optimum peptide : %s, probability %2f" % (optimum_pep_seq, fitness_value[0]))
                    output_file.write("Optimum peptide : %s, probability %2f\n" % (optimum_pep_seq, fitness_value[0]))        
        

        del creator.PeptideSequence
        del creator.Individual

    def generate(self,pop_size=50,CXPB=0.5,MUTPB=0.2,NGEN=50):
       
        pop = self.toolbox.population(pop_size)

        #if fixed residues are provided
        if len(self.fixed_residues) > 0:
            pop_ind_temp = []
            for index, residue in self.fixed_residues.items():
                residue_index = PeptideSelectionGA.AMINOACIDS.index(residue)
                for pop_lst in pop:
                    pop_lst[index-1]= residue_index
                    pop_ind_temp.append(pop_lst)

            pop = pop_ind_temp


        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("-- generation {} --".format(g + 1))
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g + 1)] = max([ind.fitness.values[0] for ind in pop])
           
           # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring)) #TypeError: 'map' object is not subscriptable

            # Apply crossover and mutation on the offspring [startAt:endBefore:skip]
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2, CXPB)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            cx_mut_ind = [ind for ind in offspring if not ind.fitness.valid]
            

            #if fixed residues are provided
            if len(self.fixed_residues) > 0:
                cx_mut_ind_temp = []
                for index, residue in self.fixed_residues.items():
                    residue_index = PeptideSelectionGA.AMINOACIDS.index(residue)
                    for cx_mut_lst in cx_mut_ind:
                        cx_mut_lst[index-1]= residue_index
                        cx_mut_ind_temp.append(cx_mut_lst)

                cx_mut_ind = cx_mut_ind_temp

        
            fitnesses = map(self.toolbox.evaluate, cx_mut_ind)
            for ind, fit in zip(cx_mut_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
                
        return pop

