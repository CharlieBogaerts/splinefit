import numpy as np
import sys
sys.path.append('G:/My Drive/Documents/Python3/MyPackages/splinefit')
import splinefit as mvs
from matplotlib import pyplot as plt


# Making data to fit model to
x1_range = 2.2
x2_range = 2.2

data_res = 30
X1 = np.linspace(-x1_range/2, x1_range/2, data_res)
X2 = np.linspace(-x2_range/2, x2_range/2, data_res)
X1_mesh, X2_mesh= np.meshgrid(X1, X2)
X1_flat = X1_mesh.flatten()
X2_flat = X2_mesh.flatten()
X_fit = np.vstack([X1_flat, X2_flat]).T
Y_fit = np.random.normal(size = X2_flat.size)
R = X1_flat**2 + X2_flat**2
Y_fit = np.sin(10*R)/R

# Set up model with four hyper cubes (squares in this case, since 2D), with 
# x0 borders at -1, .5, ,1 and x1 borders at -1,-.5, 0, .5, 1. (dont need to be
# equidistant)
border_loc = [[-1, .5 ,1],[-1,-.5, 0, .5, 1]] 

# define polynomial with terms along x0 of up to (but not including) order 3 
# so max is x0^3,  and along x1 up to order 3, so max x1^2. The highest order
# term is thus x0^3 * x1^2. polynomial includes all possible lower order terms.
poly_orders = np.array([4, 3])

# Set continuity in x0 direction to 1 (so up to 2), so function values are 
# equal at polynomial intersections, and the same along x1
deriv_orders = np.array([2, 2])

# Make model. Result is best estimate model, and the estimated variance of the
# model estimation
Model, ModelVar = mvs.model_from_data(X_fit, Y_fit, border_loc, poly_orders, deriv_orders)


# Plotting results
data_res = 50
X1 = np.linspace(-x1_range/2, x1_range/2, data_res)
X2 = np.linspace(-x2_range/2, x2_range/2, data_res)
X1_mesh, X2_mesh= np.meshgrid(X1, X2)
X1_flat = X1_mesh.flatten()
X2_flat = X2_mesh.flatten()

X_model = np.vstack([X1_flat, X2_flat]).T

Y_model = Model.eval(X_model)
Y_model_var = ModelVar.eval(X_model)

Y_model_mesh = Y_model.reshape(X1_mesh.shape)
Y_model_std_mesh = np.sqrt(Y_model_var.reshape(X1_mesh.shape))

if True:
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    # First subplot (ax[0])
    ax[0].scatter(X_fit[:, 0], X_fit[:, 1], Y_fit, c='orange')
    ax[0].plot_surface(X1_mesh, X2_mesh, Y_model_mesh)
    ax[0].set_xlabel('X1')
    ax[0].set_ylabel('X2')
    ax[0].set_zlabel('Y')


    # Second subplot (ax[1])
    ax[1].plot_surface(X1_mesh, X2_mesh, Y_model_std_mesh)
    ax[1].set_xlabel('X1')
    ax[1].set_ylabel('X2')
    ax[1].set_zlabel('Y_std')

    # Show the plot
    plt.show()
