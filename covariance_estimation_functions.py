
import numpy as np
from itertools import product
import sys
import scipy


def sort_flex(cop_values, n, upper, lower):
    '''
    Sort data points into windows and calculate frequencies.
    :param cop_values: np-array of pseudo-observations u,v of copula
    :param n: number of samples
    :param upper: np-array with upper bound of the rectangle
    :param lower: np-array with lower bound of the rectangle
    :return: np-array with counted frequencies in each box of the copula domain
    '''

    # Get number of dimensions
    dimensions = np.size(cop_values, axis=1)

    # Window size
    windows = 3

    # Copy cop_values to a
    a = np.copy(cop_values)

    # Create helper array with windows repeated for each dimension
    dimhelper = np.repeat(windows, dimensions)

    # Create empty matrix with windows repeated for each dimension
    p = np.zeros(shape=dimhelper)

    # Convert u,v to target coordinates
    for i in range(0, n):
        for j in range(0, dimensions):
            if a[i, j] < lower[j]:
                a[i, j] = 0
            elif a[i, j] > upper[j]:
                a[i, j] = 2
            else:
                a[i, j] = 1

    # Fill matrix p
    for index, x in np.ndenumerate(p):
        for i in range(0, n):
            if np.array_equal(a[i, :], index):
                p[index] = p[index] + 1

    p = p / n

    return p


def a_e(i, k, z, n1, cop_values, upper, lower):
    '''
    Function estimates the facor in front of Cov-Term a. [e.g. Var (ax+y)=a**2*Var(x)+Var(y)+2aCov(x,y)] :param i:
    tuple indicating the Covariance of which copulas should be calculated :param k: int denoting if a partial
    derivative (i.e. if k=1 --> part. deriv. of first dim) of the first box should be used :param z: np-array
    specifing the position [1,...,1] of the relevant box in phat_u/phat_v. :param n1: number of copula samples :param
    cop_values: np-array with pseudo observations :param upper: np-array with upper bound of the rectangle :param
    lower: np-array with lower bound of the rectangle :return: Prefactor a
    '''

    # Calculate delta as 1 divided by the square root of n1
    delta = 1 / (np.sqrt(n1))
    # Convert i to a numpy array
    i = np.array(i)
    # Convert z to a numpy array and set u to upper if i == 0 and lower otherwise
    # u = np.array(z)
    u = np.where(i == 0, upper, lower)
    # Subtract sys.float_info.epsilon from u if it is equal to 1 or add it if u is equal to 0
    u = np.where(u == 1, u - sys.float_info.epsilon, u)
    u = np.where(u == 0, u + sys.float_info.epsilon, u)

    # If k is equal to 0, return 1
    if k == 0:
        return 1

    else:
        # Set checkifworking to 0
        checkifworking = 0
        # Create a copy of u called u_help_up
        u_help_up = np.copy(u)
        # Add delta to the k-1 element of u_help_up
        u_help_up[k - 1] = u[k - 1] + delta

        # If the k-1 element of u_help_up is greater than 1, set checkifworking to 1 and delta_help to 1 minus
        # the k-1 element of u
        if u_help_up[k - 1] > 1:
            checkifworking = 1
            delta_help = 1 - u[k - 1]
        # Set all elements of u_help_up that are greater than 1 to 1
        u_help_up = np.where(u_help_up > 1, 1, u_help_up)

        # Create a copy of u called u_help_low
        u_help_low = np.copy(u)
        # Subtract delta from the k-1 element of u_help_low
        u_help_low[k - 1] = u[k - 1] - delta

        # If the k-1 element of u_help_low is less than 0, set checkifworking to 1 and delta_help to
        # the k-1 element of u
        if u_help_low[k - 1] < 0:
            checkifworking = 1
            delta_help = u[k - 1]
        # Set all elements of u_help_low that are smaller than 0 to 0
        u_help_low = np.where(u_help_low < 0, 0, u_help_low)

        # Create a copy of cop_values called cop_values_copy
        cop_values_copy = cop_values.copy()
        # Iterate over elements in cop_values_copy and add 1 if the value is below u_help_up
        for i in range(u.shape[0]):
            cop_values_copy[:, i] = np.where(cop_values_copy[:, i] <= u_help_up[i], 1, 0)
        up = np.sum(np.where(np.sum(cop_values_copy, axis=1) == np.size(cop_values, axis=1), 1, 0))
        # Convert the integer to a frequency
        value_up = up / n1

        # Create a new copy of cop_values called cop_values_copy
        cop_values_copy = cop_values.copy()
        # Iterate over elements in cop_values_copy and add 1 if the value is below u_help_low
        for i in range(u.shape[0]):
            cop_values_copy[:, i] = np.where(cop_values_copy[:, i] <= u_help_low[i], 1, 0)
        low = np.sum(np.where(np.sum(cop_values_copy, axis=1) == np.size(cop_values, axis=1), 1, 0))
        # Convert the integer to a frequency
        value_low = low / n1

        # Calculate gradient and use corresponding intervall
        if checkifworking == 1:
            a = (value_up - value_low) / (delta + delta_help)
        else:
            a = (value_up - value_low) / (2 * delta)

    return float(a)


def covCop_est(i, k_1, k_2, z, d, phat_u, phat_v, number_bins_per_dimension, upper_u, lower_u, upper_v, lower_v,
               cop_values):
    '''
    Function calculates the specific cov-Term of two Gaussian Processes within copula 1 and copula 2 as well as the
    sign of the term
    :param i: tuple indicating the Covariance of which copulas should be calculated
    :param k_1: int denoting if a partial derivative (i.e. if k_1=1 --> part. deriv. of first dim) of the first box
    should be used
    :param k_2: int denoting if a partial derivative (i.e. if k_2=1 --> part. deriv. of first dim) of the second box
    should be used
    :param z: np-array specifing the position [1,...,1] of the relevant box in phat_u/phat_v.
    :param d: #dimesions
    :param phat_u: np-array with number of samples within each box of  copula 1
    :param phat_v: np-array with number of samples within each box of  copula 2
    :param number_bins_per_dimension: int with fixed value of 3. Needed for computational reasons
    :param upper_u: np-array with upper bound of the first rectangle
    :param lower_u: np-array with lower bound of the first rectangle
    :param upper_v: np-array with upper bound of the second rectangle
    :param lower_v: np-array with lowerbound of the second rectangle
    :param cop_values: np-array with pseudo observations
    :return: covaraince of two Gaussian Processes within copula 1 and copula 2
    '''

    # Create helping vectors
    i = np.array(i)
    u = list(z)
    u_1 = np.where(i[0] == 0, (i[0] + u + 1), (1 * u))
    u_2 = np.where(i[1] == 0, (i[1] + u + 1), (1 * u))

    if k_1 == 0:
        # Convert u_1 to a list of integers
        u_1 = list(map(int, u_1))

        # Use slicing notation to select the first d dimensions of phat_u
        f = np.sum(phat_u[tuple(slice(x) for x in u_1)][tuple(slice(None) for _ in range(d))])

        # Assign u_1 to u_1help
        u_1help = u_1

    else:
        # u_help als in C eingesetzer Vektor um f zu erhalten erzeugen
        u_1help = np.ones(d) * number_bins_per_dimension
        u_1help[k_1 - 1] = u_1[k_1 - 1]
        # Convert u_1help to a list of integers
        u_1help = list(map(int, u_1help))

        # Use slicing notation to select the first d dimensions of phat_u
        f = np.sum(phat_u[tuple(slice(x) for x in u_1help)][tuple(slice(None) for _ in range(d))])

    if k_2 == 0:
        # Convert u_2 to a list of integers
        u_2 = list(map(int, u_2))

        # Use slicing notation to select the first d dimensions of phat_v
        g = np.sum(phat_v[tuple(slice(x) for x in u_2)][tuple(slice(None) for _ in range(d))])

        # Assign u_2 to u_2help
        u_2help = u_2
    else:
        u_2help = np.ones(d) * number_bins_per_dimension
        u_2help[k_2 - 1] = u_2[k_2 - 1]

        # Convert u_2help to a list of integers
        u_2help = list(map(int, u_2help))

        # Use slicing notation to select the first d dimensions of phat_v
        g = np.sum(phat_v[tuple(slice(x) for x in u_2help)][tuple(slice(None) for _ in range(d))])

    # Convert u_1help and u_2help to arrays of floats
    u_1help = np.array(u_1help, dtype='float64')
    u_2help = np.array(u_2help, dtype='float64')

    # Iterate over the dimensions of u_1help and u_2help
    for i in range(d):
        u_1help[i] = np.where(u_1help[i] == 1, lower_u[i], np.where(u_1help[i] == 2, upper_u[i], 1))
        u_2help[i] = np.where(u_2help[i] == 1, lower_v[i], np.where(u_2help[i] == 2, upper_v[i], 1))

    # Get u_3 as minimum uf u_1_help and u_2help
    u_3 = np.minimum(u_1help, u_2help)

    # Initialize the mask to all True values
    mask = np.ones(cop_values.shape[0], dtype=bool)

    # Iterate over the dimensions of u_3
    for i in range(d):
        mask &= (cop_values[:, i] <= u_3[i])

    # Compute the fraction of True values in the mask
    h = np.sum(mask) / np.size(cop_values, axis=0)

    # Get Sign
    vzhelp1 = 1 if k_1 > 0 else 0
    vzhelp2 = 1 if k_2 > 0 else 0
    vz = ((-1) ** (np.sum(i) + vzhelp1 + vzhelp2))

    return (h - f * g) * vz


def calculate_Covariance_est(z, phat_u, phat_v, number_bins_per_dimension, cop_values, n1, upper_u, lower_u, upper_v,
                             lower_v):
    '''
    Function calculates the covariance of two given copula windows.
    :param z: np-array specifing the position [1,...,1] of the relevant boy in phat_u/phat_v.
    :param phat_u: np-array with number of samples within each box of  copula 1
    :param phat_v: np-array with number of samples within each box of  copula 2
    :param number_bins_per_dimension: int with fixed value of 3. Needed for computational reasons
    :param cop_values: np-array with pseudo observations
    :param n1: number of copula samples
    :param upper_u: np-array with upper bound of the first rectangle
    :param lower_u: np-array with lower bound of the first rectangle
    :param upper_v: np-array with upper bound of the second rectangle
    :param lower_v: np-array with lower bound of the second rectangle
    :return: covarance estiamte
    '''

    # Get number of dimensions d
    d = np.size(z)
    _ = 0

    # Initialize covariance
    covariance = 0.0

    # Calculate variance-terms
    y = list(product([0, 1], repeat=d))
    for i in product(y, repeat=2):
        for k_1 in range(0, d + 1):
            for k_2 in range(0, d + 1):
                covariance = covariance + a_e(i[0], k_1, z, n1, cop_values, upper_u,
                                              lower_u) * a_e(i[1], k_2, z,
                                                             n1,
                                                             cop_values,
                                                             upper_v,
                                                             lower_v) * covCop_est(
                    i, k_1,
                    k_2, z, d,
                    phat_u, phat_v,
                    number_bins_per_dimension, upper_u, lower_u, upper_v, lower_v, cop_values)

    return covariance


def get_covariance(upper_bound_rectangle_1, lower_bound_rectangle_1, upper_bound_rectangle_2, lower_bound_rectangle_2,
                   pseudo_obs):
    ''' Function serves as a wrapper-function for (co-)varaince estiamtion. First copula samples in the relevant window
     are counted, then the estiamtion is performed.
    :param upper_bound_rectangle_1: np-array with upper bound of the first rectangle
    :param lower_bound_rectangle_1: np-array with lower bound of the first rectangle
    :param upper_bound_rectangle_2: np-array with upper bound of the second rectangle
    :param lower_bound_rectangle_2: np-array with lower bound of the second rectangle
    :param pseudo_obs: np-array with copula pseudo observations
    :return: (co-)variance estiamtion
    '''
    # Sanity checks
    s = {upper_bound_rectangle_1.shape, lower_bound_rectangle_1.shape, upper_bound_rectangle_2.shape,
         lower_bound_rectangle_2.shape}
    if len(s) != 1:
        raise ValueError('Rectangle bounds do not match in dimension!')
    if any(upper_bound_rectangle_1 <= lower_bound_rectangle_1):
        raise ValueError('Rectangle_1 bounds do not fit!')
    if any(upper_bound_rectangle_2 <= lower_bound_rectangle_2):
        raise ValueError('Rectangle_2 bounds do not fit!')
    if (np.size(pseudo_obs, axis=1) != int(upper_bound_rectangle_1.shape[0])):
        raise ValueError('Dimensions of rectangles and data does not fit!')

    # Get number of samples and dimensions
    size_of_copula = np.size(pseudo_obs, axis=0)
    dim = np.size(pseudo_obs, axis=1)

    # Create z. Used for computational reasons
    z = np.ones(dim)

    # Use function sort_flex to count samples in relevant window of the copula
    phat_u = sort_flex(pseudo_obs, size_of_copula, upper_bound_rectangle_1, lower_bound_rectangle_1)
    phat_v = sort_flex(pseudo_obs, size_of_copula, upper_bound_rectangle_2, lower_bound_rectangle_2)

    # Estimate (co-)variance of empirical copula rectangle
    erg_var_pred_est = calculate_Covariance_est(z, phat_u, phat_v, 3, pseudo_obs, size_of_copula,
                                                upper_bound_rectangle_1, lower_bound_rectangle_1,
                                                upper_bound_rectangle_2, lower_bound_rectangle_2)
    return erg_var_pred_est


def make_pseudo_obs(dimensions, samplesize, dependence, family):
    ''' Function creates pseudo-observations in given dimensions, size and from given copula model.
    :param dimensions: # Dimensions
    :param samplesize: Desired samplesize
    :param dependence: Covariance matrix or other dependence measure
    :param family: Type of copula
    '''
    if family == 'normal':
        # Check if covariance matrix is given
        if type(dependence) != np.ndarray:
            raise TypeError("Covariance must be a numpy ndarray")
        #  Check if matrix is symmetric
        if dependence.ndim != 2 or dependence.shape[0] != dependence.shape[1]:
            raise ValueError("Covariance must be a 2D square matrix")

        # Create multivariate normal samples
        A = np.random.multivariate_normal(np.zeros(dimensions), dependence, samplesize)
        B = A.copy()
        # Make Pseudo-Obs
        for col in range(A.shape[1]):
            B[:, col] = scipy.stats.rankdata(A[:, col]) / (samplesize + 1)

    elif family == 'clayton':
        raise ValueError('Not implemented yet!')

    return B
