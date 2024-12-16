import numpy as np

class MVSModel:
    def __init__(self, params, Grid, PolyMIS):
        self.PolyMIS = PolyMIS
        self.Grid = Grid
        self.params = params


    def evalSingle(self, x_point):
        """
        Same function as 'MVSModel.eval', but accepts a single point.

        :param vertices_c: (1D numpy array) The dependent variable point.
        :returns: (float) calculated dependent variable value.
        """
        if x_point.ndim != 1:
            raise ValueError("'point_c' should be a 1D numpy array")
        x_point = x_point.reshape(1,-1)
        y_point = self.eval(x_point)[0]
        return y_point

    def eval(self, x_points):
        """
        Evaluates the simplex spline at independent variable points x_points. 
        For vertices outside the spline domain np.NaN is returned

        :param x_points: (2D numpy array) The dependent variable points. The
            0th axis should contain the different vertices, and the 1st axis 
            the different entries of each point.
        :returns: (1D numpy array) calculated dependent variable values.
        """
        if x_points.shape[1] != self.PolyMIS.dim:
            raise ValueError('Given points and spline model differ in dimension.')
        bins_with_labels = self.Grid.classify(x_points)

        cube_dimensions = self.Grid.get_cube_measurements()
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
