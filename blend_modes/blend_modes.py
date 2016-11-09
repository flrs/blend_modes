import numpy as np


def soft_light(img_in, img_layer, opacity):
    """
    Apply soft light blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Soft_Light>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = soft_light(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    # The following code does this:
    #   multiply = img_in[:, :, :3]*img_layer[:, :, :3]
    #   screen = 1.0 - (1.0-img_in[:, :, :3])*(1.0-img_layer[:, :, :3])
    #   comp = (1.0 - img_in[:, :, :3]) * multiply + img_in[:, :, :3] * screen
    #   ratio_rs = np.reshape(np.repeat(ratio,3),comp.shape)
    #   img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)

    comp = (1.0 - img_in[:, :, :3]) * img_in[:, :, :3] * img_layer[:, :, :3] \
           + img_in[:, :, :3] * (1.0 - (1.0-img_in[:, :, :3])*(1.0-img_layer[:, :, :3]))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def lighten_only(img_in, img_layer, opacity):
    """
    Apply lighten only blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Lighten_Only>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = lighten_only(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.maximum(img_in[:, :, :3], img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def screen(img_in, img_layer, opacity):
    """
    Apply screen blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Screen>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = screen(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = 1.0 - (1.0 - img_in[:, :, :3]) * (1.0 - img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def dodge(img_in, img_layer, opacity):
    """
    Apply dodge blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Dodge_and_burn>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = dodge(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum(img_in[:, :, :3]/(1.0 - img_layer[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def addition(img_in, img_layer, opacity):
    """
    Apply addition blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Addition>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = addition(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] + img_layer[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = np.clip(comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs), 0.0, 1.0)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def darken_only(img_in, img_layer, opacity):
    """
    Apply darken only blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Darken_Only>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = darken_only(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum(img_in[:, :, :3], img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def multiply(img_in, img_layer, opacity):
    """
    Apply multiply blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Multiply>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = multiply(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_layer[:, :, :3] * img_in[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def hard_light(img_in, img_layer, opacity):
    """
    Apply hard light blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Hard_Light>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = hard_light(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.greater(img_layer[:, :, :3], 0.5)*np.minimum(1.0-((1.0 - img_in[:, :, :3])
                                                             * (1.0 - (img_layer[:, :, :3] - 0.5) * 2.0)), 1.0) \
           + np.logical_not(np.greater(img_layer[:, :, :3], 0.5))*np.minimum(img_in[:, :, :3]
                                                                            * (img_layer[:, :, :3] * 2.0), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def difference(img_in, img_layer, opacity):
    """
    Apply difference blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Difference>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = difference(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] - img_layer[:, :, :3]
    comp[comp < 0.0] *= -1.0

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def subtract(img_in, img_layer, opacity):
    """
    Apply subtract blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Subtract>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = subtract(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] - img_layer[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = np.clip(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def grain_extract(img_in, img_layer, opacity):
    """
    Apply grain extract blending mode of a layer on an image.

    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Extract>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_extract(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_in[:, :, :3] - img_layer[:, :, :3] + 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def grain_merge(img_in, img_layer, opacity):
    """
    Apply grain merge blending mode of a layer on an image.

    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Merge>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_merge(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_in[:, :, :3] + img_layer[:, :, :3] - 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def divide(img_in, img_layer, opacity):
    """
    Apply divide blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Divide>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = divide(img_in,img_layer,0.5)
        cv2.imshow('window', img_out)
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum((256.0 / 255.0 * img_in[:, :, :3]) / (1.0 / 255.0 + img_layer[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out*255.0


def _compose_alpha(img_in, img_layer, opacity):
    """
    Calculate alpha composition ratio between two images.
    """

    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3])*opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3])*comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha/new_alpha
    ratio[ratio == np.NAN] = 0.0
    return ratio
