import numpy as np 


def array_size(array):
    """
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Invalid input array!")

    mb_size = array.nbytes / (1024 ** 2)
    return mb_sizes

import unitary_generator

unitary_generator.hello()