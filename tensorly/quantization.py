import tensorly as tl
from .base import unfold

import torch
import math

def get_tensor_stats(tensor, qscheme, mode = 0):  
    """Returns max and min along last dimension for mode-`mode` unfolding
    of `tensor` with modes starting at `0`, if qscheme is per channel.
    If qscheme is per tensor,  returns max and min computed across all elements.
    
    Parameters
    ----------
    tensor : ndarray
    qscheme : quantization scheme, default is ``torch.per_tensor_affine``
        Has to be one of: ``torch.per_tensor_affine``, ``torch.per_tensor_symmetric``, ``torch.per_channel_affine``, ``torch.per_channel_symmetric``
    mode : int, default is 0
          Indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``.

    Returns
    -------
    tuple
        max, min of unfolded_tensor along last dimension.
    """
    if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
        unfolded_tensor = unfold(tensor, mode = mode)

        tmax = unfolded_tensor.max(dim = -1)[0]
        tmin = unfolded_tensor.min(dim = -1)[0]
        #tmean = unfolded_tensor.mean(dim = -1)
        
    elif qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
        tmax = tensor.max()
        tmin = tensor.min()
        
    else:
        raise TypeError("Can't collect statistics. Unknown quantization scheme: {}".format(qscheme))
        return
   
    
    return tmax, tmin


def get_scale_zeropoint(tensor,\
                        dtype = torch.qint8,\
                        qscheme = torch.per_tensor_affine,\
                        dim = None):
    """Returns scale and zero_point to apply in quantization formula.
    
    Parameters
    ----------
    tensor : Tensor
        Float tensor to quantize.
    dtype : ``torch.dtype``, default is ``torch.qint8``
        The desired data type of returned tensor.
        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``.
    qscheme : quantization scheme, default is ``torch.per_tensor_affine``
        Has to be one of: ``torch.per_tensor_affine``, ``torch.per_tensor_symmetric``, ``torch.per_channel_affine``, ``torch.per_channel_symmetric``
    dim : int or None, default is None
        If dim is not None, along the dimension `dim` the values in the `tensor` are scaled and offset by a different value (effectively the scale and offset become vectors).
        If dim is None, all values in the `tensor` are scaled and offset by the same value.
    
    Returns
    -------
    scale
        Scale to apply in quantization formula.
    zero_point
        Offset in integer value that maps to float zero.
    """
    
    #scale_denom = qmax - qmin, where qmax = 2**(nbits-1) - 1, qmin = 2**(nbits - 1)
    if dtype == torch.qint8:
        q = 2**7
        scale_denom = 2*q - 1 
    elif dtype == torch.qint32:
        q = 2**31
        scale_denom = 2*q - 1
    else:
        raise TypeError("Can't perform quantization. Unknown quantization type: {}".format(dtype))
        return
    
    
    tmax, tmin = get_tensor_stats(tensor, qscheme, mode = dim)
    
    if qscheme in [torch.per_channel_symmetric, torch.per_tensor_symmetric]:
        scale = 2 * torch.where(tmin.abs() > tmax, tmin.abs(), tmax) / scale_denom
        
        if dtype == torch.qint8:
            zero_point = torch.zeros(scale.shape).int()
        else:
            zero_point = (torch.zeros(scale.shape) + 128).int()
        
    elif qscheme in [torch.per_channel_affine, torch.per_tensor_affine]:
        scale = (tmax - tmin) / scale_denom 
        zero_point = (-q - (tmin / scale).int()).int()
        
        zero_point = torch.clamp(zero_point, -q, q - 1)
    else:
        raise TypeError("Can't perform quantization. Unknown quantization scheme: {}".format(qscheme))
        return

    return scale, zero_point 
    
    
    
def quantize_qint(tensor,\
                  dtype = torch.qint8,\
                  qscheme = torch.per_tensor_affine,\
                  dim = None,\
                  return_scale_zeropoint = False):
    """Converts a float `tensor` to quantized tensor with `scale` and `zero_point`
    computed via function `get_scale_zeropoint`.
    
    Parameters
    ----------
    tensor : Tensor
        Float tensor to quantize.
    dtype : ``torch.dtype``, default is ``torch.qint8``
        The desired data type of returned tensor.
        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``.
    qscheme : quantization scheme, default is ``torch.per_tensor_affine``
        Has to be one of: ``torch.per_tensor_affine``, ``torch.per_tensor_symmetric``, ``torch.per_channel_affine``, ``torch.per_channel_symmetric``
    dim : int or None, default is None
        If dim is not None, along the dimension `dim` the values in the `tensor` are scaled and offset by a different value (effectively the scale and offset become vectors).
        If dim is None, all values in the `tensor` are scaled and offset by the same value.
    return_scale_zeropoint : bool, default False
        Activate return of scale and zero_point.
            
    Returns
    -------
    Quantized Tensor
        float version of quantized `tensor`.
        
    scale
        Scale to apply in quantization formula.
    zero_point
        Offset in integer value that maps to float zero.          
    """
    
    
    scale, zero_point = get_scale_zeropoint(tensor,\
                                            dtype = dtype,\
                                            qscheme = qscheme,\
                                            dim = dim)
    
    if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
        qtensor = torch.quantize_per_channel(tensor,\
                                             scales=scale, zero_points = zero_point,\
                                             dtype = dtype, axis = dim)
    
    elif qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
        qtensor = torch.quantize_per_tensor(tensor,\
                                            scale=scale, zero_point = zero_point,\
                                            dtype = dtype)
    else:
        raise TypeError("Can't perform quantization. Unknown quantization scheme: {}".format(qscheme))
        return
    
    
    if return_scale_zeropoint:
        return qtensor.dequantize(), scale, zero_point
    
    else:
        return qtensor.dequantize()

