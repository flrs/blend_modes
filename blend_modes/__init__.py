"""This Python package implements blend modes for images.
"""
import numpy as np


def _assert_image_format(image, fcn_name: str, arg_name: str, force_alpha: bool = True):
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
                  'supply a variable of type np.ndarray to the "{arg_name}" argument.'\
            .format(fcn_name=fcn_name, arg_name=arg_name, var_type=str(type(image).__name__))
        raise TypeError(err_msg)

    if not image.dtype.kind == 'f':
        err_msg = 'The blend_modes function "{fcn_name}" received a numpy array of dtype (data type) kind ' \
                  '"{var_kind}" for its argument "{arg_name}". The function however expects a numpy array of the ' \
                  'data type kind "f" (floating-point) for this argument. Please supply a numpy array with the data ' \
                  'type kind "f" (floating-point) to the "{arg_name}" argument.'\
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
                  'to the "{arg_name}" argument.'\
            .format(fcn_name=fcn_name, arg_name=arg_name, n_layers=str(image.shape[2]))
        raise TypeError(err_msg)


def _assert_opacity(opacity, fcn_name: str, arg_name: str = 'opacity'):
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
    if not isinstance(opacity, float):
        err_msg = 'The blend_modes function "{fcn_name}" received a variable of type "{var_type}" for its argument ' \
                  '"{arg_name}". The function however expects the value passed to "{arg_name}" to be of type ' \
                  '"float". Please pass a variable of type "float" to the "{arg_name}" argument of function ' \
                  '"{fcn_name}".'\
            .format(fcn_name=fcn_name, arg_name=arg_name, var_type=str(type(opacity).__name__))
        raise TypeError(err_msg)

    if not 0.0 <= opacity <= 1.0:
        err_msg = 'The blend_modes function "{fcn_name}" received the value "{val}" for its argument "{arg_name)". ' \
                  'The function however expects that the value for "{arg_name}" is inside the range 0.0 <= x <= 1.0. ' \
                  'Please pass a variable in that range to the "{arg_name}" argument of function "{fcn_name}".' \
            .format(fcn_name=fcn_name, arg_name=arg_name, val=str(opacity))
        raise ValueError(err_msg)


def normal(img_in, img_layer, opacity):
    """Apply "normal" blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/wiki/Alpha_compositing#Description>`__.

    Example::

        import cv2, numpy
        from blend_modes import normal
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = normal(img_in,img_layer,0.8)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()
    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      opacity(float): Desired opacity of layer for blending
    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image
    """

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    # Add alpha-channels, if they are not provided
    if img_in_norm.shape[2] == 3:
        img_in_norm = np.dstack((img_in_norm, np.zeros(img_in_norm.shape[:2] + (3,))))
    if img_layer_norm.shape[2] == 3:
        img_layer_norm = np.dstack((img_layer_norm, np.zeros(img_layer_norm.shape[:2] + (3,))))

    # Extract alpha-channels and apply opacity
    img_in_alp = np.expand_dims(img_in_norm[:, :, 3], 2)  # alpha of b, prepared for broadcasting
    img_layer_alp = np.expand_dims(img_layer_norm[:, :, 3], 2) * opacity  # alpha of a, prepared for broadcasting

    # Blend images
    c_out = (img_layer_norm[:, :, :3] * img_layer_alp + img_in_norm[:, :, :3] * img_in_alp * (1 - img_layer_alp)) \
            / (img_layer_alp + img_in_alp * (1 - img_layer_alp))

    # Blend alpha
    cout_alp = img_layer_alp + img_in_alp * (1 - img_layer_alp)

    # Combine image and alpha
    c_out = np.dstack((c_out, cout_alp))

    np.nan_to_num(c_out, copy=False)

    return c_out * 255.0


def soft_light(img_in, img_layer, opacity):
    """Apply soft light blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Soft_Light>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import soft_light
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = soft_light(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    _assert_image_format(img_in, 'soft_light', 'img_in')
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    # The following code does this:
    #   multiply = img_in_norm[:, :, :3]*img_layer[:, :, :3]
    #   screen = 1.0 - (1.0-img_in_norm[:, :, :3])*(1.0-img_layer[:, :, :3])
    #   comp = (1.0 - img_in_norm[:, :, :3]) * multiply + img_in_norm[:, :, :3] * screen
    #   ratio_rs = np.reshape(np.repeat(ratio,3),comp.shape)
    #   img_out = comp*ratio_rs + img_in_norm[:, :, :3] * (1.0-ratio_rs)

    comp = (1.0 - img_in_norm[:, :, :3]) * img_in_norm[:, :, :3] * img_layer_norm[:, :, :3] \
           + img_in_norm[:, :, :3] * (1.0 - (1.0 - img_in_norm[:, :, :3]) * (1.0 - img_layer_norm[:, :, :3]))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def lighten_only(img_in, img_layer, opacity):
    """Apply lighten only blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Lighten_Only>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import lighten_only
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = lighten_only(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.maximum(img_in_norm[:, :, :3], img_layer_norm[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def screen(img_in, img_layer, opacity):
    """Apply screen blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Screen>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import screen
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = screen(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = 1.0 - (1.0 - img_in_norm[:, :, :3]) * (1.0 - img_layer_norm[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def dodge(img_in, img_layer, opacity):
    """Apply dodge blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Dodge_and_burn>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import dodge
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = dodge(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.minimum(img_in_norm[:, :, :3] / (1.0 - img_layer_norm[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def addition(img_in, img_layer, opacity):
    """Apply addition blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Addition>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import addition
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = addition(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = img_in_norm[:, :, :3] + img_layer_norm[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = np.clip(comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def darken_only(img_in, img_layer, opacity):
    """Apply darken only blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Darken_Only>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import darken_only
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = darken_only(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.minimum(img_in_norm[:, :, :3], img_layer_norm[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def multiply(img_in, img_layer, opacity):
    """Apply multiply blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Multiply>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import multiply
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = multiply(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.clip(img_layer_norm[:, :, :3] * img_in_norm[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def hard_light(img_in, img_layer, opacity):
    """Apply hard light blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Hard_Light>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import hard_light
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = hard_light(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.greater(img_layer_norm[:, :, :3], 0.5) \
           * np.minimum(1.0 - ((1.0 - img_in_norm[:, :, :3])
           * (1.0 - (img_layer_norm[:, :, :3] - 0.5) * 2.0)), 1.0) \
           + np.logical_not(np.greater(img_layer_norm[:, :, :3], 0.5)) \
           * np.minimum(img_in_norm[:, :, :3] * (img_layer_norm[:, :, :3] * 2.0), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def difference(img_in, img_layer, opacity):
    """Apply difference blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Difference>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import difference
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = difference(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = img_in_norm[:, :, :3] - img_layer_norm[:, :, :3]
    comp[comp < 0.0] *= -1.0

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def subtract(img_in, img_layer, opacity):
    """Apply subtract blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Subtract>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import subtract
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = subtract(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = img_in[:, :, :3] - img_layer_norm[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = np.clip(comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def grain_extract(img_in, img_layer, opacity):
    """Apply grain extract blending mode of a layer on an image.
    
    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Extract>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import grain_extract
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_extract(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.clip(img_in_norm[:, :, :3] - img_layer_norm[:, :, :3] + 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def grain_merge(img_in, img_layer, opacity):
    """Apply grain merge blending mode of a layer on an image.
    
    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Merge>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import grain_merge
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_merge(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.clip(img_in_norm[:, :, :3] + img_layer_norm[:, :, :3] - 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def divide(img_in, img_layer, opacity):
    """Apply divide blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Divide>`__.
    
    Example::
    
        import cv2, numpy
        from blend_modes import divide
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = divide(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.minimum((256.0 / 255.0 * img_in_norm[:, :, :3]) / (1.0 / 255.0 + img_layer_norm[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def overlay(img_in, img_layer, opacity):
    """Apply overlay blending mode of a layer on an image.
    
    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=868545948#Overlay>`__.
    
    .. note:: The implementation of this method was changed in version 2.0.0. Previously, it would be identical to the
              soft light blending mode. Now, it resembles the implementation on Wikipedia. You can still use the soft light
              blending mode if you are looking for backwards compatibility.
    
    Example::
    
        import cv2, numpy
        from blend_modes import overlay
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = overlay(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    Args:
      img_in(3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0): Image to be blended upon
      img_layer(3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0): Layer to be blended with image
      opacity(float): Desired opacity of layer for blending

    Returns:
      3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0: Blended image

    """

    # Sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in_norm = img_in/255.0
    img_layer_norm = img_layer/255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.less(img_in_norm[:, :, :3], 0.5) * (2 * img_in_norm[:, :, :3] * img_layer_norm[:, :, :3]) \
           + np.greater_equal(img_in_norm[:, :, :3], 0.5) \
           * (1 - (2 * (1 - img_in_norm[:, :, :3]) * (1 - img_layer_norm[:, :, :3])))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def _compose_alpha(img_in, img_layer, opacity):
    """Calculate alpha composition ratio between two images.
    """

    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.NAN] = 0.0
    return ratio
