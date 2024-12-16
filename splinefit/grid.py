import numpy as np
import splinefit.multiindexset as mis
import itertools as it


class Grid:
    """
    This class contains information about the grid structure that is used to 
    construct the spline. It contains measurements and locations of each hyper 
    cube that make up the N-dimensional grid.
    """
    def __init__(self, border_loc):
        '''
        Constructor using the locations of the border positions of the grid 
        along each dimensions. A border is a hyperplane seperating the 
        different hyper cubes. In a two dimensional grid, the borders would be 
        all horizontal and all vertical lines. The borders also include the 
        outer edges.

        :param border_loc): (2D list) The border locations. The 0th list 
            corresponds to the different dimensions, and th 1'st to the 
            different borders in this dimension. Entries are the locations of 
            each border.
        '''
        self.border_loc = border_loc
        self.nr_of_borders = self._calc_grid_size(border_loc)
        self.MIS = mis.MultiIndexSet(self.nr_of_borders)
        self.dim = len(self.nr_of_borders)

    @staticmethod
    def _calc_grid_size(border_loc):
        dim_max = len(border_loc)
        grid_size = np.zeros(dim_max, dtype = int)
        for dim in range(dim_max):
            grid_size[dim] = len(border_loc[dim])-1
        return grid_size
    
    def classify(self, x_points):
        '''
        Finds in what hypercube a set of x data falls. Functon returns a list 
        of arrays. Each array corresponds to a hypercube. Each array is a list 
        of indices, or labels, which indicate that the i'th point of x_points 
        is in this hypercube. 

        :param x_points: (2D numpy array) The dependent variable points. The
            0th axis should contain the different vertices, and the 1st axis 
            the different entries of each point.
        :returns: (list of 1D numpy arrays) bins containing the sets of labels
        '''
        nr_of_points, dim_max = x_points.shape
        if self.dim != dim_max:
            raise Exception("x data does not match dimension of grid")
        locations = np.zeros((nr_of_points,dim_max), dtype=int)   #multi indices of all points within range
        labels = np.arange(nr_of_points)
        for dim_nr in range(dim_max):
            nr_of_borders = len(self.border_loc[dim_nr])
            locs_in_dim = np.searchsorted(np.array(self.border_loc[dim_nr]), x_points[:,dim_nr])
            large_enough = locs_in_dim != 0
            small_enough = locs_in_dim != nr_of_borders
            within_range = large_enough & small_enough
            locations = locations[within_range]
            x_points = x_points[within_range]
            labels = labels[within_range]
            locations[:,dim_nr] =locs_in_dim[within_range]-1
        locs_by_id = self.MIS.to_nrs(locations)
        bins_with_labels = []
        for i in range(self.MIS.length):
            in_cube = locs_by_id == i
            bins_with_labels.append(labels[in_cube])
        return bins_with_labels


    def get_cube_measurements(self):
        """
        For each hyper cube, get the width along each dimension. 

        :returns: (2D numpy array) On the 0'th axis the different hyper cubes, 
            and on the 1'th axis the cube widths
        """
        diffs = []
        for dim_nr in range(self.dim):
            borders_in_dim = self.border_loc[dim_nr]
            diff = [borders_in_dim[i + 1] - borders_in_dim[i] for i in range(len(borders_in_dim)-1)]
            diffs.append(diff)
        combinations = it.product(*diffs)
        return np.array(list(combinations))
    
    def get_cube_roots(self):
        """
        For each hyper cube, get the lowest border along each dimension. This 
        thus describes the corner of the hyper cube closest to minus infinity 
        in each dimension.

        :returns: (2D numpy array) On the 0'th axis the different hyper cubes, 
        and on the 1'th axis the cube roots
        """
        border_loc_red = [border_in_dim[:-1] for border_in_dim in self.border_loc]
        combinations = it.product(*border_loc_red)
        return np.array(list(combinations))