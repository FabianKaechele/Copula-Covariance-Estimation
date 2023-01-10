# Empirical copula covariance estimation
This repository contains the Python-Code for the paper:  ****************  


## Features
* Estimate the variance and covariance for the empirical copula process of a given sample with pseudo-observations
* Specify any two rectangles in the domain of the copula
* Code runs on any dimensionality (however slow for high dimensions)

## Technologies
* Python version: 3.8
* Numpy version: 1.21.5
* Scipy verison: 1.7.3

## Usage
```python
    # Initialize values and arrays for copula
    size_of_copula = 100
    dimensions = 3
    covariance =  np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])

    # Set bounds for rectangle of interest
    upper_bound_rectangle_1 = np.array([1 / 3, 1 / 3, 1 / 3])
    lower_bound_rectangle_1 = np.array([0, 0, 0])
    upper_bound_rectangle_2 = np.array([1 / 3, 1 / 3, 1 / 3])
    lower_bound_rectangle_2 = np.array([0, 0, 0])

    # Create Pseudo-Obs
    pseudo_obs = make_pseudo_obs(dimensions=dimensions, samplesize=size_of_copula, dependence=covariance,
                                 family='normal')
    # Call covariance estimation
    estimated_covariance = get_covariance(upper_bound_rectangle_1, lower_bound_rectangle_1, upper_bound_rectangle_2,
                                          lower_bound_rectangle_2, pseudo_obs)

    print(estimated_covariance)
```
