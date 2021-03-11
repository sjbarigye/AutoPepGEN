import numpy as np

import descriptor_calculator
import tabu_sequences
from joblib import dump, load
from sklearn import svm

class predictor_model():
    """description of class"""

    def __init__(self, peptide_sequence, prediction_model,array_max ,array_min, descriptors):
        self.peptide_sequence = peptide_sequence
        self.descriptors = descriptors
        self.prediction_model = prediction_model
        self.array_max = array_max
        self.array_min = array_min


    def predictor(self):
        #check if sequence is in tabu_sequence
        if self.peptide_sequence in tabu_sequences.tabu_sequences.PEP_SEQ:
            return 0.0

        des_generator = descriptor_calculator.descriptor_calculator(self.peptide_sequence,self.descriptors)
        desc_value_array = des_generator.parse_seq_compute_alldes()

        #standardize value array
        seq_value_array_std = self.standardize_min_man(np.asarray(desc_value_array), self.array_max, self.array_min)

        #each label is mapped to the numerical index
        probability = self.prediction_model.predict_proba(np.reshape(seq_value_array_std,(-1,seq_value_array_std.size)))
       
        return probability[0][1], seq_value_array_std


    def standardize_min_man(self, desc_value_array, array_max, array_min):

        pep_seq_length = desc_value_array.size
        seq_value_array_std = []

        for index in range(pep_seq_length):
            value = desc_value_array[index]

            max_value = array_max[index]
            min_value = array_min[index]

            x_std = (value - min_value) / (max_value - min_value)
            x_std_scaled = x_std * (1 - (-1)) + (-1)
            seq_value_array_std.append(x_std_scaled)

        return np.array(seq_value_array_std)

