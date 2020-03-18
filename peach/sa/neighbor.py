################################################################################
# Peach - Computational Intelligence for Python
# Jose Alexandre Nalon
#
# This file: sa/neighbor.py
# Simulated Annealing
################################################################################

# Doc string, reStructuredText formatted:
__doc__ = """
This module implements a general class to compute neighbors for continuous and
binary simulated annealing algorithms. The continuous neighbor functions return
an array with a neighbor of a given estimate; the binary neighbor functions
return a ``bitarray`` object.
"""

################################################################################
from numpy import array, reshape, vectorize
from numpy.random import uniform, standard_normal
from numpy import zeros, ones, arange
from numpy import tile
import numpy.random

from random import randrange
import types


################################################################################
# Classes
################################################################################
class ContinuousNeighbor(object):
    '''
    Base class for continuous neighbor functions

    This class should be derived to implement a function which computes the
    neighbor of a given estimate. Every neighbor function should implement at
    least two methods, defined below:

      __init__(self, *cnf, **kw)
        Initializes the object. There are no mandatory arguments, but any
        parameters can be used here to configure the operator. For example, a
        class can define a variance for randomly chose the neighbor -- this
        should be defined here::

          __init__(self, variance=1.0)

        A default value should always be offered, if possible.

      __call__(self, x):
        The ``__call__`` interface should be programmed to actually compute the
        value of the neighbor. This method should receive an estimate in ``x``
        and use whatever parameters from the instantiation to compute the new
        estimate. It should return the new estimate.

    Please, note that the SA implementations relies on this behaviour: it will
    pass an estimate to your ``__call__`` method and expects to received the
    result back.

    This class can be used also to transform a simple function in a neighbor
    function. In this case, the outside function must compute in an appropriate
    way the new estimate.
    '''
    def __init__(self, f):
        '''
        Creates a neighbor function from a function.

        :Parameters:
          f
            The function to be transformed. This function must receive an array
            of any size and shape as an estimate, and return an estimate of the
            same size and shape as a result. A function that operates only over
            a single number can be used -- in this case, the function operation
            will propagate over all components of the estimate.
        '''
        if isinstance(f, types.FunctionType):
            self.__f = vectorize(f)
        else:
            self.__f = f

    def __call__(self, x):
        '''
        Computes the neighbor of the given estimate.

        :Parameters:
          x
            The estimate to which the neighbor must be computed.
        '''
        return self.__f(x)


################################################################################
class GaussianNeighbor(ContinuousNeighbor):
    '''
    A new estimate based on a gaussian distribution

    This class creates a function that computes the neighbor of an estimate by
    adding a gaussian distributed randomly choosen vector with the same shape
    and size of the estimate.
    '''
    def __init__(self, variance=0.05):
        '''
        Initializes the neighbor operator

        :Parameters:
          variance
            This is the variance of the gaussian distribution used to randomize
            the estimate. This can be given as a single value or as an array. In
            the first case, the same value will be used for all the components
            of the estimate; in the second case, ``variance`` should be an array
            with the same number of components of the estimate, and each
            component in this array is the variance of the corresponding
            component in the estimate array.
        '''
        self.variance = variance
        '''Variance of the gaussian distribution.'''

    def __call__(self, x):
        '''
        Computes the neighbor of the given estimate.

        :Parameters:
          x
            The estimate to which the neighbor must be computed.
        '''
        s = x.shape
        x = array(x).ravel()
        xn = x + self.variance*standard_normal(x.shape)
        return reshape(xn, s)


################################################################################
class UniformNeighbor(ContinuousNeighbor):
    '''
    A new estimate based on a uniform distribution

    This class creates a function that computes the neighbor of an estimate by
    adding a uniform distributed randomly choosen vector with the same shape
    and size of the estimate.
    '''
    def __init__(self, xl=-1.0, xh=1.0):
        '''
        Initializes the neighbor operator

        :Parameters:
          xl
            The lower limit of the distribution;

          xh
            The upper limit of the distribution. Both values can be given as a
            single value or as an array. In the first case, the same value will
            be used for all the components of the estimate; in the second case,
            they should be an array with the same number of components of the
            estimate, and each component in this array is the variance of the
            corresponding component in the estimate array.
        '''
        self.xl = xl
        '''Lower limit of the uniform distribution.'''
        self.xh = xh
        '''Upper limit of the uniform distribution.'''

    def __call__(self, x):
        '''
        Computes the neighbor of the given estimate.

        :Parameters:
          x
            The estimate to which the neighbor must be computed.
        '''
        s = x.shape
        x = array(x).ravel()
        n = len(x)
        xn = x + uniform(self.xl, self.xh, n)
        return reshape(xn, s)


################################################################################
class BinaryNeighbor(object):
    '''
    Base class for binary neighbor functions

    This class should be derived to implement a function which computes the
    neighbor of a given estimate. Every neighbor functions should implement at
    least two methods, defined below:

      __init__(self, *cnf, **kw)
        Initializes the object. There are no mandatory arguments, but any
        parameters can be used here to configure the operator. For example, a
        class can define a bit change rate -- this should be defined here::

          __init__(self, rate=0.01)

        A default value should always be offered, if possible.

      __call__(self, x):
        The ``__call__`` interface should be programmed to actually compute the
        value of the neighbor. This method should receive an estimate in ``x``
        and use whatever parameters from the instantiation to compute the new
        estimate. It should return the new estimate.

    Please, note that the SA implementations relies on this behaviour: it will
    pass an estimate to your ``__call__`` method and expects to received the
    result back. Notice, however, that the SA implementation does not expect
    that the result is sane, ie, that it is in conformity with the
    representation used in the algorithm. A sanity check is done inside the
    binary SA class. Please, consult the documentation on ``BinarySA`` for
    further details.

    This class can be used also to transform a simple function in a neighbor
    function. In this case, the outside function must compute in an appropriate
    way the new estimate.
    '''
    def __init__(self, f):
        '''
        Creates a neighbor function from a function.

        :Parameters:
          f
            The function to be transformed. This function must receive a
            bitarray of any length as an estimate, and return a new bitarray of
            the same length as a result.
        '''
        self.__f = f

    def __call__(self, x):
        '''
        Computes the neighbor of the given estimate.

        :Parameters:
          x
            The estimate to which the neighbor must be computed.
        '''
        return self.__f(x)


################################################################################
class InvertBitsNeighbor(BinaryNeighbor):
    '''
    A simple neighborhood based on the change of a few bits.

    This neighbor will be computed by randomly choosing a bit in the bitarray
    representing the estimate and change a number of bits in the bitarray and
    inverting their value.
    '''
    def __init__(self, nb=2):
        '''
        Initializes the operator.

        :Parameters:
          nb
            The number of bits to be randomly choosen to be inverted in the
            calculation of the neighbor. Be very careful while choosing this
            parameter. While very large optimizations can benefit from a big
            value here, it is not recommended that more than one bit per
            variable is inverted at each step -- otherwise, the neighbor might
            fall very far from the present estimate, which can make the
            algorithm not work accordingly. This defaults to 2, that is, at each
            step, only one bit will be inverted at most.
        '''
        self.__nb = nb


    def __call__(self, x):
        '''
        Computes the neighbor of the given estimate.

        :Parameters:
          x
            The estimate to which the neighbor must be computed.
        '''
        xn = x[:]
        for i in range(self.__nb):
            index = randrange(len(xn))
            xn[index] = 1 - xn[index]
        return xn


##########################################################################
class MaskedBitsNeighbor(BinaryNeighbor):
    '''
    A neighborhood based on the change of a few bits in positions
    defined by a mask.

    This neighbor will be computed by randomly choosing a bit in the bitarray
    representing the estimate and change a number of bits in the bitarray and
    inverting their value. The bits that may be changed are selected by a
    mask supplied by user.
    '''
    def __init__(self, mask=ones((8,), dtype='bool'), nb=2):
        '''
        Initializes the operator.

        :Parameters:
          mask
            Binary array which specifies which bits may be changed.
            The length of the argument to the __call__ method
            must be  divisible by the length of the mask.
          nb
            The number of bits to be randomly choosen to be inverted
            in the calculation of the neighbor. Be very careful while
            choosing this parameter. While very large optimizations
            can benefit from a big value here, it is not recommended
            that more than one bit per variable is inverted at each
            step -- otherwise, the neighbor might
            fall very far from the present estimate, which can make the
            algorithm not work accordingly. This defaults to 2, that is,
            at each step, only one bit will be inverted at most.
          nsign
            The number of sign changes made when calculating a
            neighbor. Sign change for integers consists of flipping
            all the bits
        '''
        self.__nb = nb
        self.__bitmask = array(mask, dtype='bool')
        self.__allowed_bits = None

    def _flip_bits(self, x):
        """
        Inverts self.__nb bits of the input bitarray
        """
        xn = x[:]
        # cache the mask
        if self.__allowed_bits is None:
            assert(len(xn) % len(self.__bitmask) == 0)
            data_length = len(xn)
            num_elems = data_length // len(self.__bitmask)

            self.__allowed_bits = arange(
                data_length)[tile(self.__bitmask, num_elems)]

        # invert some bits
        indices = numpy.random.choice(self.__allowed_bits, self.__nb)
        for index in indices:
            xn[index] = 1 - xn[index]

        return xn

    def __call__(self, x):
        '''
        Computes the neighbor of the given estimate.

        :Parameters:
          x
            The estimate to which the neighbor must be computed.
            It is assumed that the estimate's length does not change
            after the first call; otherwise, unspecified behaviour
            will happen.
        '''
        return self._flip_bits(x)


# alias as this class is suitable for changing unsigned integers
UintBitsNeighbor = MaskedBitsNeighbor


##########################################################################
class IntBitsNeighbor(MaskedBitsNeighbor):
    '''
    A neighborhood based on the change of a few bits in positions
    defined by a mask.

    This neighbor will be computed by randomly choosing a bit in the bitarray
    representing the estimate and change a number of bits in the bitarray and
    inverting their value. The bits that may be changed are selected by a
    mask supplied by user.
    '''
    def __init__(self, mask=ones((8,), dtype='bool'), nb=2, nsign=2):
        '''
        Initializes the operator.

        :Parameters:
          mask
            Binary array which specifies which bits may be changed.
            The length of the argument to the __call__ method
            must be  divisible by the length of the mask.
          nb
            The number of bits to be randomly choosen to be inverted
            in the calculation of the neighbor. Be very careful while
            choosing this parameter. While very large optimizations
            can benefit from a big value here, it is not recommended
            that more than one bit per variable is inverted at each
            step -- otherwise, the neighbor might
            fall very far from the present estimate, which can make the
            algorithm not work accordingly. This defaults to 2, that is,
            at each step, only one bit will be inverted at most.
          nsign
            The number of sign changes made when calculating a
            neighbor. Sign change for integers consists of flipping
            all the bits
        '''
        super().__init__(mask=mask, nb=nb)

        self.__nsign = nsign
        self.__typelen = len(self.__bitmask)
        self.__elem_offsets = None

    def _invert_sign(self, x):
        """
        Inverts the sign of self.__nsign integer data elements in the
        input bitarray
        """
        xn = x[:]
        # cache element offsets
        if self.__elem_offsets is None:
            self.__elem_offsets = arange(0, len(xn), len(self.__bitmask))

        elem_offsets = numpy.random.choice(
            self.__elem_offsets, self.__nsign)

        # Invert sign of an integer
        for offset in elem_offsets:
            xn[offset:offset+self.__typelen-1] = ~xn[
                offset:offset+self.__typelen-1]
        return xn

    def __call__(self, x):
        '''
        Computes the neighbor of the given estimate.

        :Parameters:
          x
            The estimate to which the neighbor must be computed.
            It is assumed that the estimate's length does not change
            after the first call; otherwise, unspecified behaviour
            will happen.
        '''
        x = self._flip_bits(x)
        return self._invert_sign(x)


##########################################################################
if __name__ == "__main__":
    pass
