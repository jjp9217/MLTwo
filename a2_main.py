################################################################
# Machine Learning
# Programming Assignment - Bayesian Classifier
#
# Some Numpy Notes:
# * np.matmul (@) -- standard matrix product
# * np.dot -- dot/inner product of two arrays. Differs in how dimensions are
#           handled in some cases (see docs)
# * np.multiply (*) -- element-wise multiplication
#
# To compute all dot products for row vectors with themselves in an Nx2 array:
#  1. Multiply with itself (M @ M.T) then select diagonal elements OR
#  2. Use an Einstein sum operation: np.einsum('ij,ij->i',a,a) 
#     to take dot products over rows with themselves directly
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import math
import numpy as np

from debug import *
from bayes import *
from results_visualization import *

################################################################
# Data set functions / test functions
################################################################

def split_rows_on_class_labels( sample_array, debug=False ):
    index_positions = np.argsort( sample_array[:, -1] )
    sorted_array = sample_array[ index_positions ]

    # Collect feature vectors and corresponding labels AFTER sort 
    X = sorted_array[:,:-1]
    y = sorted_array[:,-1]

    # Produce *list* of data arrays w. one array per class (( represented by list index ))
    # * np.diff( data[:,2] ) -- computes differences between rows in the label column
    # * np.flatnonzero -- returns vector of INDICES for non-zero elements in vector of differences
    #     NOTE: +1 corrects index offsets (N item list -> N-1 differences)
    label_diffs = np.diff( sorted_array[:,-1] )
    split_indices = np.flatnonzero( label_diffs ) + 1

    sorted_no_labels = sorted_array[:,:-1]
    split_data = np.vsplit( sorted_no_labels, split_indices )

    # Debugging
    if debug:
        dcheck( 'split shape', [ np.shape(a) for a in split_data ] )
        dnpcheck( 'Top elements from list of arrays in split', [ a[:3,:] for a in split_data ] )

    # Return split data, along with data matrix and corresponding labels in y
    return ( split_data, X, y )


def gauss_test():
    # Test scores for given Gaussian parameters (see Wk 4 Lecture 1 slides)
    test_mean_vector = np.array([5,1])
    test_cov_matrix = np.array([[2,0],[0,1]])
    
    test_point = np.array([[1,2]])
    test_points = np.array([[5,1],[1,2]])

    # Check distance computations
    test_value = sq_mhlnbs_dist( test_point, test_mean_vector, np.linalg.inv( test_cov_matrix ) ) 
    two_values = sq_mhlnbs_dist( test_points, test_mean_vector, np.linalg.inv( test_cov_matrix ) )
    dcheck('mhd test value (expecting 9.0)', test_value)
    dncheck('mhd mean + test (expecting [ 0.0, 9.0])', two_values )

    # Check gaussian computations
    g_test_value = gaussian( mean_density(test_cov_matrix), test_value )
    g_two_values = gaussian( mean_density(test_cov_matrix), two_values )
    dcheck('gaussian test value (expecting 0.00125... )', g_test_value)
    dncheck('gaussian mean + test (expecting [ 0.11254..., 0.00125... ])', g_two_values)


################################################################
# Main Program
################################################################

def main():
    ##################################
    # Class dicts, formatting
    # & load data
    ##################################
    A1_CLASS = { 0: 'Blue', 1: 'Orange' }
    BNRS_CLASS = { 0: 'Bolts', 1: 'Nuts', 2: 'Rings', 3: 'Scrap' }
 
    A1_CLASS_FORMAT = { 0: ('o','blue'), 1: ('o','orange') }
    BNRS_CLASS_FORMAT = { 0: ('+', 'blue'), 1: ('*', 'red'), 2: ('o', 'yellow'), 3: ('x', 'black') }

    # For 'metal parts' data, decrease class labels values by 1
    data = np.loadtxt('data/data_bnrs.csv', delimiter=',')
    data = data - np.array([0,0,1])

    # Load a1 data
    a1_data = np.load('data/data_a1.npy')

    # Split samples into list of samples for each class
    ( split_data, X, y ) = split_rows_on_class_labels( data )
    ( a1_split_data, a1_X, a1_y ) = split_rows_on_class_labels( a1_data )

    ##################################
    # Load parameters
    ##################################
    (a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices ) = \
            bayesian_parameters( A1_CLASS, a1_split_data, title='A1 Parameters')
    (class_priors, class_mean_vectors, class_cov_matrices ) = \
        bayesian_parameters( BNRS_CLASS, split_data, title='Metal Parts Parameters')

    # A1 classifiers
    a1_map = map_classifier( a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices )
    a1_bc_uniform = bayes_classifier( uniform_cost_matrix( len(A1_CLASS) ), a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices )
    
    # Bolts, nuts, rings, scrap classifiers
    bc_map = map_classifier( class_priors, class_mean_vectors, class_cov_matrices )
    bc_uniform = bayes_classifier( uniform_cost_matrix( len(BNRS_CLASS) ), class_priors, class_mean_vectors, class_cov_matrices )
    bc_non_uniform = bayes_classifier( bnrs_unequal_costs( len(BNRS_CLASS) ), class_priors, class_mean_vectors, class_cov_matrices )

    ##################################
    # Compute evaluation metrics for
    # training samples, draw plots
    ##################################
    # Sanity check
    conf_matrix(BNRS_CLASS, data, data, title='SANITY CHECK: Metal Parts Data vs. Itself')

    # Results on A1 training data
    conf_matrix(A1_CLASS, a1_map( a1_X ), a1_data, title='A1 MAP')
    conf_matrix(A1_CLASS, a1_bc_uniform( a1_X ), a1_data, title='A1 Uniform Cost Bayesian')

    # Results for Bolts, nuts, rings, and scrap training data
    conf_matrix(BNRS_CLASS, bc_map(X), data, title='Metal Parts MAP', cost_matrix=bnrs_unequal_costs( len(BNRS_CLASS)))
    conf_matrix(BNRS_CLASS, bc_uniform(X), data, title='Metal Parts Uniform Cost Bayesian', cost_matrix=bnrs_unequal_costs( len(BNRS_CLASS)))
    conf_matrix(BNRS_CLASS, bc_non_uniform(X), data, title='Metal Parts - Unequal Costs', cost_matrix=bnrs_unequal_costs( len(BNRS_CLASS)))

    ##################################
    # Visualizations
    ##################################
    print("\n>>> Drawing decision boundaries (2D) and class score contours (3D)")

    # >>> AIM TO GET 'ticks' UP TO 1000 WHILE STILL HAVING THE PROGRAM RUN QUICKLY
    ticks=100
    draw_results(a1_data, a1_map, "A1 MAP", "A1_MAP.pdf", A1_CLASS_FORMAT, axis_tick_count=ticks)
    draw_results(data, bc_map, "Metal Parts MAP", "Metal_MAP.pdf", BNRS_CLASS_FORMAT, axis_tick_count=ticks)
    draw_results(data, bc_non_uniform, "Metal Parts Unequal Costs", "Metal_NonUniform.pdf", BNRS_CLASS_FORMAT, axis_tick_count=ticks)

    draw_contours(a1_data, a1_map, "A1 MAP", "A1_MAP_Contours.pdf", A1_CLASS_FORMAT, axis_tick_count=ticks) 
    draw_contours(data, bc_map, "Metal Parts MAP", "Metal_MAP_Contours.pdf", BNRS_CLASS_FORMAT, axis_tick_count=ticks)
    draw_contours(data, bc_non_uniform, "Metal Parts Unequal Costs", "Metal_NonUniform_Contours.pdf", BNRS_CLASS_FORMAT, axis_tick_count=ticks)

    # Create interactive plot to visualize scrap class scores in feature space.
    # Feel free to try others, comment out as needed.
    draw_contours(data, bc_non_uniform, "Metal Parts Unequal Costs - Scrap Class Scores", "", BNRS_CLASS_FORMAT, axis_tick_count=ticks, show=True, classes=[3])
    
    print("done.")

# Main program to run
if __name__ == "__main__":
    gauss_test()
    main()
