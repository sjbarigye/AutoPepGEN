import numpy as np
import constants_peptides
import aggregation_model 


class descriptor_calculator(object):
    """description of class"""

    AMINOACIDS = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

    def __init__(self, peptide_sequence, *descriptors):
        self.peptide_sequence = peptide_sequence
        self.descriptors = descriptors[0]



    def parse_seq_compute_alldes(self):
        #results_dict = {}
        results_array = []

        for descriptor in self.descriptors:
            descriptor_split = str(descriptor).split("_")

            des_constant = descriptor_split[0]
            des_operator = descriptor_split[1]

            seq_value_array = []

            for aminoacid in self.peptide_sequence:
                des_constant_ind = descriptor_calculator.AMINOACIDS.index(aminoacid)
                
                for enum_const in constants_peptides.constants_peptides:
                    if enum_const.name == des_constant:
                        seq_value_array.append(enum_const.value[des_constant_ind])

  
            #print(des_operator)

            results_array.append(self.compute_operator(np.asarray(seq_value_array), des_operator))


        #return results_dict
        return results_array


    def compute_operator(self, seq_value_array, invariant):
        operator = aggregation_model.operators(seq_value_array)

        if invariant == "M1":
            return operator.sum()
        elif invariant == "PM2":
            return operator.PotentialMean(2)
        elif invariant == "GM":
            return operator.geomean()
        elif invariant == "PM3":
            return operator.PotentialMean(3)
        elif invariant == "P75":
            return operator.percentile(75)
        elif invariant == "I50":
            return operator.i50()
        elif invariant == "R":
            return operator.range()
        elif invariant == "MIN":
            return operator.min()
        elif invariant == "[MBAC(3)]M1":
            return aggregation_model.operators(operator.MoreauBrotoAutocorrelation(3)).sum()
        else:
            return np.NaN