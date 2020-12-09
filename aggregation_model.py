from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
import math


class operators():
    def __init__(self, descriptor_array):
        self.descriptor_array = descriptor_array

    def sum(self):
        return np.sum(self.descriptor_array)

    def geomean(self):
        return gmean(self.descriptor_array)

    def PotentialMean(self, pot):
        sum = 0;
        n = self.descriptor_array.size
        nZeros = 0

        if n == 0 :
            return 0

        for value in self.descriptor_array:
            if value == 0:
                nZeros += 1
            else:
                sum = math.pow(value, pot)

        if (n - nZeros) == 0:
            return 0;

        if (pot == -1):
            sum = (n - nZeros) / sum
        elif (pot == 2):
            sum = math.sqrt(sum / (n - nZeros))
        elif(pot == 3):
            root_cube = lambda x: x**(1./3.) if 0 <= x else -(-x)**(1./3.)
            sum = root_cube(sum / (n - nZeros))

        return sum

    #Statistics
    def percentile(self, percentage):
        return np.percentile(self.descriptor_array, percentage)

    def i50(self):
        return self.percentile(75) - self.percentile(25)

    def range(self):
        return np.max(self.descriptor_array) - np.min(self.descriptor_array)

    def min(self):
        return np.min(self.descriptor_array)


    def MoreauBrotoAutocorrelation(self, numOfLag):

        pep_seq_length = self.descriptor_array.size
        moreauBrotoAutocorrArray = []

        for index in range(numOfLag):
            lag = index + 1
            array_value = 0

            for pep_index in range (pep_seq_length - lag):
                array_value = (self.descriptor_array[pep_index]*self.descriptor_array[pep_index + lag])/ (pep_seq_length - lag)
                moreauBrotoAutocorrArray.append(array_value)

            if array_value == 0:
                moreauBrotoAutocorrArray.append(0.0)

        return moreauBrotoAutocorrArray
  
