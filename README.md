# splinefit
Estimate the relation between a scalar dependent variable Y and an N dimensional indepentent variable X using a spline based on data points. The spline tries to fit the data using a least squares approach. The continuity between the splines can be set seperately along each dimension. In each dimension, it is possible to set
- no continuity (independent polynomials)
- absolute value continuity (0th order), polynomials intersect along their mating borders
- first derivative continuity (1st order), polynomials intersect and have same first derivative along their mating borders
- second derivative continuity (3st order), polynomials intersect and have same first and second derivative along their mating borders
- etc...

An N dimensional cartesian grid has to be defined, which does not have to be equidistant along any dimension. Each hypercube of this grid corresponds to an N dimensional polynomial. The polynomial structure that is used in each hypercube can be defined by setting the maximum power along each dimension. A polynomial is then created that uses all possible mononomials upto these maximum powers. 

See example/BasicsN2 to see an example of fitting a spline to data with 2 dimensional dependent variable X.

## Dependencies
Only uses numpy and itertools

## Available user functions
See docstrings for info.

- model_from_data(X_fit, Y_fit, border_loc, poly_orders, deriv_orders)
- MVSModel.evalSingle(self, x_point)
- MVSModel.eval(self, x_points)
