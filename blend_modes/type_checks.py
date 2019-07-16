"""This module includes functions to check if variable types match expected formats
"""
import numpy as np


def assert_image_format(image, fcn_name: str, arg_name: str, force_alpha: bool = True):
    """Assert if image arguments have the expected format

    This function asserts if image arguments have the expected format and raises easily understandable errors
    if the format does not match the expected format.

    The function specifically checks:
        - If the image is a numpy array
        - If the numpy array is of 'float' type
        - If the array is 3-dimensional (height x width x R/G/B/alpha layers)
        - If the array has the required amount of layers

    Args:
        image: The image to be checked
        fcn_name(str): Name of the calling function, for display in error messages
        arg_name(str): Name of the relevant function argument, for display in error messages
        force_alpha(bool): Indicate whether the image is expected to include an alpha layer on top of R/G/B layers

    Raises:
        TypeError: If the image does not match the format

    """
    if not isinstance(image, np.ndarray):
        err_msg = 'The blend_modes function "{fcn_name}" received a value of type "{var_type}" for its argument ' \
                  '"{arg_name}". The function however expects a value of type "np.ndarray" for this argument. Please ' \
                  'supply a variable of type np.ndarray to the "{arg_name}" argument.' \
            .format(fcn_name=fcn_name, arg_name=arg_name, var_type=str(type(image).__name__))
        raise TypeError(err_msg)

    if not image.dtype.kind == 'f':
        err_msg = 'The blend_modes function "{fcn_name}" received a numpy array of dtype (data type) kind ' \
                  '"{var_kind}" for its argument "{arg_name}". The function however expects a numpy array of the ' \
                  'data type kind "f" (floating-point) for this argument. Please supply a numpy array with the data ' \
                  'type kind "f" (floating-point) to the "{arg_name}" argument.' \
            .format(fcn_name=fcn_name, arg_name=arg_name, var_kind=str(image.dtype.kind))
        raise TypeError(err_msg)

    if not len(image.shape) == 3:
        err_msg = 'The blend_modes function "{fcn_name}" received a {n_dim}-dimensional numpy array for its argument ' \
                  '"{arg_name}". The function however expects a 3-dimensional array for this argument in the shape ' \
                  '(height x width x R/G/B/A layers). Please supply a 3-dimensional numpy array with that shape to ' \
                  'the "{arg_name}" argument.' \
            .format(fcn_name=fcn_name, arg_name=arg_name, n_dim=str(len(image.shape)))
        raise TypeError(err_msg)

    if force_alpha and not image.shape[2] == 4:
        err_msg = 'The blend_modes function "{fcn_name}" received a numpy array with {n_layers} layers for its ' \
                  'argument "{arg_name}". The function however expects a 4-layer array representing red, green, ' \
                  'blue, and alpha channel for this argument. Please supply a numpy array that includes all 4 layers ' \
                  'to the "{arg_name}" argument.' \
            .format(fcn_name=fcn_name, arg_name=arg_name, n_layers=str(image.shape[2]))
        raise TypeError(err_msg)


def assert_opacity(opacity, fcn_name: str, arg_name: str = 'opacity'):
    """Assert if opacity has the expected format

    This function checks if opacity has a float format and is in the range 0.0 <= opacity <= 1.0.

    Args:
        opacity: The opacity value to be checked
        fcn_name(str): Name of the calling function, for display in error messages
        arg_name(str): Name of the 'opacity' argument in the calling function, for display in error messages.
            Defaults to 'opacity'.

    Raises:
        TypeError: If the opacity is not a float value
        ValueError: If the opacity is not in the range 0.0 <= opacity <= 1.0

    """
    # Allow ints for opacity
    if not isinstance(opacity, float) and not isinstance(opacity, int):
        err_msg = 'The blend_modes function "{fcn_name}" received a variable of type "{var_type}" for its argument ' \
                  '"{arg_name}". The function however expects the value passed to "{arg_name}" to be of type ' \
                  '"float". Please pass a variable of type "float" to the "{arg_name}" argument of function ' \
                  '"{fcn_name}".' \
            .format(fcn_name=fcn_name, arg_name=arg_name, var_type=str(type(opacity).__name__))
        raise TypeError(err_msg)

    if not 0.0 <= opacity <= 1.0:
        err_msg = 'The blend_modes function "{fcn_name}" received the value "{val}" for its argument "{arg_name}". ' \
                  'The function however expects that the value for "{arg_name}" is inside the range 0.0 <= x <= 1.0. ' \
                  'Please pass a variable in that range to the "{arg_name}" argument of function "{fcn_name}".' \
            .format(fcn_name=fcn_name, arg_name=arg_name, val=str(opacity))
        raise ValueError(err_msg)
