# import numpy as np
""" A utility file containing functions for optimizing a fitness function. """
import math
from typing import Callable, List, Tuple
import numpy as np
from commons.heat_exchanger import HeatExchangerModel

# create the sphere function that takes an array of parameters and returns an array of fitness values
def sphere(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The sphere function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = np.square(param1) + np.square(param2)
    else:
        fitness = np.sum(np.square(params), axis=1)
    # check if the fitness should be minimized
    return fitness if minimize else -fitness # type: ignore

def minmax_function(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray: # type: ignore
    """ The cosine mixture function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 3*(1-param1)**2*np.exp(-param1**2 - (param2+1)**2) - 10*(param1/5 - param1**3 - param2**5)*np.exp(-param1**2 - param2**2) -1/3*np.exp(-(param1+1)**2 - param2**2)
    else:
        fitness = 3*(1-params[:,0])**2*np.exp(-params[:,0]**2 - (params[:,1]+1)**2) - 10*(params[:,0]/5 - params[:,0]**3 - params[:,1]**5)*np.exp(-params[:,0]**2 - params[:,1]**2) -1/3*np.exp(-(params[:,0]+1)**2 - params[:,1]**2)
    # check if the fitness should be minimized
    return -fitness if minimize else fitness

# # cosine mixture function that takes an array of parameters and returns an array of fitness values
def cosine_mixture(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The cosine mixture function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 0.1 * (np.cos(5 * math.pi * param1) + np.cos(5 * math.pi * param2)) - (np.square(param1) + np.square(param2))
    else:
        fitness = 0.1 * np.sum(np.cos(5 * math.pi * params), axis=1) - np.sum(np.square(params), axis=1)
    # check if the fitness should be minimized
    return -fitness if minimize else fitness

def ghabit_function_2D(X):
    x1 = X[0,:]
    x0 = X[1,:]
    y = (1 - x1/2 + x1**5 + x0**3) * np.exp(-x1**2 - x0**2)
    return -y

# ghabit function that takes an array of parameters and returns an array of fitness values
def ghabit(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The ghabit function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = (1 - param1 / 2 + param1 ** 5 + param2 ** 3) * np.exp(-param1 ** 2 - param2 ** 2)
    else:
        fitness = (1 - params[:, 0] / 2 + params[:, 0] ** 5 + params[:, 1] ** 3) * np.exp(-params[:, 0] ** 2 - params[:, 1] ** 2)
    # check if the fitness should be minimized
    return -fitness if minimize else fitness

def cos_function_2D(X):
    x1 = X[0,:]
    x0 = X[1,:]
    y = (np.cos(x1-2) + np.cos(x0-2)) + (np.cos(2*x1-4) + np.cos(2*x0-4)) + (np.cos(4*x1-8) + np.cos(4*x0-8))
    return -y

def minmax_function(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The cosine mixture function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 3*(1-param1)**2*np.exp(-param1**2 - (param2+1)**2) - 10*(param1/5 - param1**3 - param2**5)*np.exp(-param1**2 - param2**2) -1/3*np.exp(-(param1+1)**2 - param2**2)
    else:
        fitness = 3*(1-params[:,0])**2*np.exp(-params[:,0]**2 - (params[:,1]+1)**2) - 10*(params[:,0]/5 - params[:,0]**3 - params[:,1]**5)*np.exp(-params[:,0]**2 - params[:,1]**2) -1/3*np.exp(-(params[:,0]+1)**2 - params[:,1]**2)
    # check if the fitness should be minimized
    return -fitness if minimize else fitness

# use the formula in cos_function_2D to create a fitness function
def cos_function(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The cos function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = np.cos(param2 - 2) + np.cos(param1 - 2) + (np.cos(2 * param2 - 4) + np.cos(2 * param1 - 4)) + (np.cos(4 * param2 - 8) + np.cos(4 * param1 - 8))
    else:
        fitness = np.sum(np.cos(params - 2), axis=1) + (np.sum(np.cos(2 * params - 4), axis=1)) + (np.sum(np.cos(4 * params - 8), axis=1))
    # check if the fitness should be minimized
    return np.multiply(fitness, -1) if minimize else fitness


def get_obj_func(name):
    """ Returns the objective function with the given name.
    :param name: The name of the objective function.
    :return: The objective function and their bounds.
    """
    return {
        'sphere': (sphere, [np.array([-3, -3]),np.array([3, 3])], 0, "maximize"),
        'minmax': (minmax_function, [np.array([-3, -3]),np.array([3, 3])], 8.107, "maximize"),
        'cosine_mixture': (cosine_mixture, [np.array([-1, -1]), np.array([1, 1])], 0.2, "maximize"),
        'ghabit': (ghabit, [np.array([-3, -3]), np.array([3, 3])], 1.058, "maximize"),
        'cos_function': (cos_function, [np.array([-2, -2]),np.array([4, 4])], 6, "maximize"),
        'heat_exchanger': (HeatExchangerModel.objective_function, [np.array([0.015, 0.1, 0.05]),np.array([0.051, 1.5, 0.5])], -49322.8, "maximize")
    }.get(name)
    
    
