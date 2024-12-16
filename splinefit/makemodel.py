
import numpy as np
import splinefit.grid as grd
import splinefit.multiindexset as mis
import splinefit.mvsmodel as mvsm



def model_from_data(x_data, y_data, border_loc, poly_orders, deriv_orders):
        Grid = grd.Grid(border_loc)
        PolyMIS =  mis.MultiIndexSet(poly_orders)
        A, b = make_regression_mats(Grid, PolyMIS, x_data, y_data)
        H = make_continuity_mat(Grid, PolyMIS, deriv_orders)
        params, covars =  ECLQS(A, b, H)
        PolyMISVar, var_params = calc_variance_params(covars, PolyMIS, Grid)
        ParameterModel = mvsm.MVSModel(params, Grid, PolyMIS)
        VarianceModel = mvsm.MVSModel(var_params, Grid, PolyMISVar)
        return ParameterModel, VarianceModel

def make_regression_mats(Grid, PolyMIS, x_points, y_points):
    bins_with_labels = Grid.classify(x_points)
    cube_dimensions = Grid.get_cube_measurements()
    cube_roots = Grid.get_cube_roots()
    Asub_list = []
    bsub_list = []
    for cube_nr in range(Grid.MIS.length):
        labels_in_cube = bins_with_labels[cube_nr]
        x_in_cube = x_points[labels_in_cube]
        y_in_cube = y_points[labels_in_cube]
        x_points_norm = (x_in_cube - cube_roots[cube_nr])/cube_dimensions[cube_nr]
        Asub = np.zeros((x_in_cube.shape[0], Grid.MIS.length*PolyMIS.length))
        for term_nr in range(PolyMIS.length):
            coef_nr = cube_nr*PolyMIS.length + term_nr
            Asub[:,coef_nr] = np.prod(x_points_norm**PolyMIS.matrix[term_nr], axis = 1)
        Asub_list.append(Asub)
        bsub_list.append(y_in_cube)
    A = np.vstack(Asub_list)
    b = np.concatenate(bsub_list)
    return A, b


def make_continuity_mat(Grid, PolyMIS, deriv_orders):
    H_sub_list = []
    for dim_nr in range(Grid.dim):
        c0_tuples = get_first_c0_tuples(Grid, dim_nr)

        poly_upto_reduced = PolyMIS.upto.copy()
        poly_upto_reduced[dim_nr]=1
        H_sub_eqs = np.prod(poly_upto_reduced)
        PolyReducedMIS = mis.MultiIndexSet(poly_upto_reduced)
        
        for border_nr in range(Grid.nr_of_borders[dim_nr]-1):
            c1_tuples = c0_tuples.copy()
            c1_tuples[:,dim_nr] +=1 
            
            terms = PolyMIS.matrix.copy()
            factors_deriv = np.ones(terms.shape[0])
            x_diff = Grid.border_loc[dim_nr][border_nr+1] - Grid.border_loc[dim_nr][border_nr]
            for deriv_nr in range(deriv_orders[dim_nr]):
                terms_fi = get_fill_in_terms(terms, dim_nr)
                factors_fi_0 = get_fill_in_factors(terms, dim_nr, 1)
                factors_fi_1 = get_fill_in_factors(terms, dim_nr, 0)

                H_mate_0 = np.zeros((H_sub_eqs, PolyMIS.length))
                H_mate_1 = np.zeros((H_sub_eqs, PolyMIS.length))

                for coef_nr in range(PolyMIS.length):           #coef_nr in single poly
                    row_nr = PolyReducedMIS.to_nr(terms_fi[coef_nr])
                    
                    H_mate_0[row_nr, coef_nr] = factors_deriv[coef_nr] * factors_fi_0[coef_nr]
                    H_mate_1[row_nr, coef_nr] = - factors_deriv[coef_nr] * factors_fi_1[coef_nr]
                
                for mate_nr in range(c0_tuples.shape[0]): 
                    H_sub = np.zeros((H_sub_eqs, PolyMIS.length*Grid.MIS.length))
                    c0_tuple = c0_tuples[mate_nr]
                    c1_tuple = c1_tuples[mate_nr]

                    first_coef_nr_0 = PolyMIS.length * Grid.MIS.to_nr(c0_tuple)
                    first_coef_nr_1 = PolyMIS.length * Grid.MIS.to_nr(c1_tuple)

                    col_start = first_coef_nr_0
                    col_end = first_coef_nr_0+PolyMIS.length
                    H_sub[:,col_start:col_end] = H_mate_0
                    col_start = first_coef_nr_1
                    col_end = first_coef_nr_1+PolyMIS.length
                    H_sub[:,col_start:col_end] = H_mate_1
                    H_sub_list.append(H_sub)

                terms, factors_deriv = deriv(terms, factors_deriv, x_diff, dim_nr)

            c0_tuples = c1_tuples
    if H_sub_list:
        return np.concatenate(H_sub_list)
    else:
        return np.empty((0,PolyMIS.length*Grid.MIS.length))

def get_first_c0_tuples(Grid, dim_nr):
    grid_size_reduced = Grid.nr_of_borders.copy()
    grid_size_reduced[dim_nr]=1
    MISReduced = mis.MultiIndexSet(grid_size_reduced)
    c1_tuples = MISReduced.matrix
    return c1_tuples


def deriv(T, v, x_diff, dim):
    D = T.copy()
    v*=D[:,dim]/x_diff
    D[:,dim]-=D[:,dim]>0
    return D, v

def get_fill_in_terms(terms, dim):
    terms_fi = terms.copy()
    terms_fi[:,dim] = 0
    return terms_fi

def get_fill_in_factors(terms, dim, value):
    factors_fi = value**terms[:,dim]
    return factors_fi


def ECLQS(A, b, H):
    n_A, m_A = A.shape
    n_H, m_H = H.shape
    M1 = np.block([[A.T@A, H.T],
                      [H, np.zeros((n_H, n_H))]])
    M2 = np.concatenate([A.T@b, np.zeros(n_H)])
    M1inv = np.linalg.pinv(M1)
    C1 = M1inv[:m_A,:m_A]       #b coefficient covariance matrix
    params_aug = M1inv @ M2
    return params_aug[:m_A], C1

def calc_variance_params(covars, PolyMIS, Grid):
    polies =  PolyMIS.matrix
    n =  PolyMIS.length
    new_mis_mat = np.zeros((int(n**2),  PolyMIS.dim), dtype = int)
    counter = 0
    for i in range(n):
        for j in range(n):
            new_mis_mat[counter] = polies[i,:] + polies[j,:]
            counter += 1

    upto_values_2 = polies[-1,:]*2+1
    MISNew = mis.MultiIndexSet(upto_values_2)

    var_params_list = []
    for cube_nr in range(Grid.MIS.length):
        params_start_i = cube_nr * PolyMIS.length
        params_end_i = params_start_i + PolyMIS.length

        covars_cube = covars[params_start_i:params_end_i,params_start_i:params_end_i]
        covars_flat = covars_cube.flatten()
        var_params_cube = np.zeros(MISNew.length)
        for i in range(n**2):
            var_params_cube[MISNew.to_nr(new_mis_mat[i])] += covars_flat[i]
        var_params_list.append(var_params_cube)
    var_params = np.concatenate(var_params_list)
    return MISNew, var_params