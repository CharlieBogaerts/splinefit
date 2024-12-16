import numpy as np
import itertools as it

class MultiIndexSet:
    def __init__(self, upto_values):
        ranges = [range(upto_value) for upto_value in upto_values]
        combinations = it.product(*ranges)
        self.upto = upto_values
        self.matrix = np.array(list(combinations))
        self.length, self.dim = self.matrix.shape

    def to_mi(self, nr):
        return self.matrix[nr,:]

    def to_nr(self, multi_index):
        bool_matrix = np.zeros((self.length, self.dim), dtype = bool)
        for i in range(self.dim):
            bool_matrix[:,i] = self.matrix[:,i] == multi_index[i]
        found_nrs = np.where(np.all(bool_matrix, axis = 1))[0]
        if found_nrs.size == 1:
            return found_nrs[0]
        else:
            raise ValueError('multi index not found or double')
        
    def to_nrs(self, multi_indices):
        n, dim = multi_indices.shape
        found_nrs = np.zeros(n)
        for i in range(n):
            found_nrs[i] = self.to_nr(multi_indices[i,:])
        return found_nrs