import numpy as np

class MVSModel:
    def __init__(self, params, Grid, PolyMIS):
        self.PolyMIS = PolyMIS
        self.Grid = Grid
        self.params = params


    def evalSingle(self, vertex_c):
        #return Y
        pass

    def eval(self, x_points):
        bins_with_labels = self.Grid.classify(x_points)

        cube_dimensions = self.Grid.get_cube_dimensions()
        cube_roots = self.Grid.get_cube_roots()

        y_points = np.empty(x_points.shape[0])
        y_points.fill(np.nan)
        for cube_nr in range(self.Grid.MIS.length):
            labels_in_cube = bins_with_labels[cube_nr]
            x_in_cube = x_points[labels_in_cube]
            x_data_norm = (x_in_cube - cube_roots[cube_nr])/cube_dimensions[cube_nr]
            products = np.zeros((x_in_cube.shape[0], self.PolyMIS.length))  
            for col_nr in range(self.PolyMIS.length):
                products[:,col_nr] = np.prod(x_data_norm ** self.PolyMIS.to_mi(col_nr), axis=1)
            params_cube = self.params[cube_nr*self.PolyMIS.length : (cube_nr+1)*self.PolyMIS.length]
            y_in_cube = products@params_cube
            y_points[labels_in_cube] = y_in_cube
        return y_points
