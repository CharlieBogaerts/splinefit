import numpy as np
import itertools as it

class MultiIndexSet:
    def __init__(self, upto_values):
        """
        Class to contain a set of multi indexes, so a multi index set.

        :param  upto_values: (1D numpy array) upto what numbers, but not
            including, the columns of MultiIndexSet.matrix should go.
        """

        ranges = [range(upto_value) for upto_value in upto_values]
        combinations = it.product(*ranges)
        self.upto = upto_values   
        self.matrix = np.array(list(combinations))
        self.length, self.dim = self.matrix.shape

    def to_mi(self, i):
        """
        Find the i'th multi index (mi) of the mis matrix.

        :param i: (int) The row of the multi index.
        :returns: (1D numpy array) The multi index.
        """
        return self.matrix[i,:]

    def to_nr(self, multi_index):
        """
        Find the row number in the matrix of given multi index.

        :param multi_index: (1D numpy array) The multi index.
        :returns: (int) The row of the multi index.
        """
        bool_matrix = np.zeros((self.length, self.dim), dtype = bool)
        for i in range(self.dim):
            bool_matrix[:,i] = self.matrix[:,i] == multi_index[i]
        found_nrs = np.where(np.all(bool_matrix, axis = 1))[0]
        if found_nrs.size == 1:
            return found_nrs[0]
        else:
            raise ValueError('multi index not found or double')
        
    def to_nrs(self, multi_indices):
        """
        Find the row number in the matrix of given multi index.

        :param multi_index: (2D numpy array) The multi indices.
        :returns: (1D numpy array) The indices of the rows the multi indices are at.
        """
        n, dim = multi_indices.shape
        found_nrs = np.zeros(n)
        for i in range(n):
            found_nrs[i] = self.to_nr(multi_indices[i,:])
        return found_nrs