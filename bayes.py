################################################################
# Machine Learning
# Programming Assignment - Bayesian Classifier
#
# bayes.py - functions for Bayesian classifier
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import math
import numpy as np

from debug import *
from results_visualization import *

################################################################
# Cost matrices
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def uniform_cost_matrix( num_classes ):
    # The uniform cost is just an inverted identity matrix.
    cost_matrix = np.zeros((num_classes,num_classes))
    return np.fill_diagonal(cost_matrix,1)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def bnrs_unequal_costs( num_classes ):
    # Rows: output class, Columns: Target (ground truth) class

    # HARDCODED FROM SLIDES, from Hastie's 2.2. I don't get why it has a num_classes argument.
    cost_matrix = np.array([
        [-0.20, 0.07, 0.07, 0.07],
        [0.07, -0.15, 0.07, 0.07],
        [0.07, 0.07, -0.05, 0.07],
        [0.03, 0.03, 0.03, 0.03]
    ])

    return cost_matrix

################################################################
# Bayesian parameters 
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def priors( split_data ):
    # split data is a list of two np arrays
    # first array is N X 2
    # second array is N X 2 as well?

    # we need to det where are the features, and where are the associated output classes
    split_one: np.matrix = split_data[0]
    split_two: np.matrix = split_data[1]

    # to calculate priors, it's the easiest thing of all.
    # just count the number of times each class appears in the training data, divide by total number of samples

    # so which is the actual class? this is just unlabeled data

    # so apparently split_one is actually the points classified as 0

    sizes = []
    total_size = 0
    for item in split_data:
        sizes.append(item.shape[0])
        total_size = total_size + item.shape[0]

    est_priors = []
    for class_size in sizes:
        est_priors.append(class_size / total_size)

    # size_split_one = split_one.shape[0]
    # size_split_two = split_two.shape[0]
    # total_size = size_split_two + size_split_one
    # prob_class_0 = size_split_one / total_size
    # prob_class_1 = size_split_two / total_size
    # est_priors = [prob_class_0, prob_class_1] # manual version for size two list


    # est_priors = [ 1/len(split_data) ] * len(split_data) # stock answer


    # what does this function expect as a return value?
    # let's give a list L=[A,B] where the chance of being in class A is L[0]
    return est_priors



def bayesian_parameters( CLASS_DICT, split_data, title='' ):
    # Compute class priors, means, and covariances matrices WITH their inverses (as pairs)
    class_priors = priors(split_data)
    class_mean_vectors = list( map( mean_vector, split_data ) )
    class_cov_matrices = list( map( covariances, split_data ) )

    # Show parameters if title passed
    if title != '':
        print('>>> ' + title)
        show_for_classes(CLASS_DICT, "[ Priors ]", class_priors )

        show_for_classes(CLASS_DICT, "[ Mean Vectors ]", class_mean_vectors)
        show_for_classes(CLASS_DICT, '[ Covariances and Inverse Covariances]', class_cov_matrices )
        print('')

    return (class_priors, class_mean_vectors, class_cov_matrices)


################################################################
# Gaussians (for class-conditional density estimates) 
################################################################

def mean_vector( data_matrix ):
    # Axis 0 is along columns (default)
    return np.mean( data_matrix, axis=0)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def covariances( data_matrix ):
    # HEADS-UP: The product of the matrix by its inverse may not be identical to the identity matrix
    #           due to finite precision. Can use np.allclose() to test for 'closeness'
    #           to ideal identity matrix (e.g., np.eye(2) for 2D identity matrix)
    d = data_matrix.shape[1] # number of rows

    covariance = np.cov(data_matrix.T)
    inverted_covariance = np.linalg.inv(covariance)


    # Returns a pair: ( covariance_matrix, inverse_covariance_matrix )
    return covariance, inverted_covariance
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def mean_density( cov_matrix ):
    # Mean density is actually the highlighted red part from the multivariate gaussian dist from slides

    # cov_matrix is still 2x2, std dev in 0,0 and 1,1, cov in 0,1 and 1,0

    sigma = np.linalg.det(cov_matrix)
    dimensionality = cov_matrix.shape[0] #will always be fine to get row count, covariance matrix is square

    left_division = 1 / ( (2 * np.pi) ** (dimensionality / 2))
    right_division = 1 / (sigma ** 2)
    product = left_division * right_division
    return product
 

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def sq_mhlnbs_dist( data_matrix, mean_vector, cov_inverse ):
    # Square of distance from the mean in *standard deviations* 
    # (e.g., a sqared mahalanobis distance of 9 implies a point is sqrt(9) = 3 standard
    # deviations from the mean.

    # Numpy 'broadcasting' insures that the mean vector is subtracted row-wise
    diff = data_matrix - mean_vector

    return np.min(diff,axis=1) 

def gaussian( mean_density, distances ):
    # NOTE: distances is a column vector of squared mahalanobis distances

    # Use numpy matrix op to apply exp to all elements of a vector
    scale_factor = np.exp( -0.5 * distances )

    # Returns Gaussian values as the value at the mean scaled by the distance
    return mean_density * scale_factor


################################################################
# Bayesian classification
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#     ** Where indicated
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def map_classifier( priors, mean_vectors, covariance_pairs ):
    # Unpack data once outside definition (to avoid re-computation)
    covariances =  np.array( [ cov_pair[0] for cov_pair in covariance_pairs ] )
    peak_scores = priors * np.array( [ mean_density(c) for c in covariances ] )

    inv_covariances =  np.array( [ cov_pair[1] for cov_pair in covariance_pairs ] )
    num_classes = len(priors)

    def classifier( data_matrix ):
        num_samples = data_matrix.shape[0]

        # Create arrays to hold distances and class scores
        distances = np.zeros( ( num_samples, num_classes ) )
        class_scores = np.zeros( ( num_samples, num_classes + 1 ) ) 

        #>>>>>>>>> EDIT THIS SECTION
        
        class_scores[:,-1] = np.zeros( num_samples)
        
        #>>>>>>>>>> END SECTION TO EDIT
        
        return class_scores

    return classifier


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#     ** Where indicated
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def bayes_classifier( cost_matrix, priors, mean_vectors, covariance_pairs ):
    # Unpack data once outside definition (to avoid re-computation)
    covariances =  np.array( [ cov_pair[0] for cov_pair in covariance_pairs ] )
    peak_scores = priors * np.array( [ mean_density(c) for c in covariances ] )

    inv_covariances =  np.array( [ cov_pair[1] for cov_pair in covariance_pairs ] )
    num_classes = len(priors)

    def classifier( data_matrix ):
        num_samples = data_matrix.shape[0]

        # Create arrays to hold distances and class scores
        distances = np.zeros( ( num_samples, num_classes ) )
        class_posteriors = np.zeros( ( num_samples, num_classes ) ) 
        class_costs_output = np.zeros( ( num_samples, num_classes + 1) ) 

        #>>>>>>>>> EDIT THIS SECTION
        
        class_costs_output[:,-1] = np.ones( num_samples )
        
        #>>>>>>>>>> END SECTION TO EDIT

        return class_costs_output

    return classifier



