[![PyPI version](https://badge.fury.io/py/blend-modes.svg)](https://badge.fury.io/py/blend-modes)
[![Conda-forge version](https://img.shields.io/conda/vn/conda-forge/blend_modes.svg)](https://anaconda.org/conda-forge/blend_modes)
[![Documentation Status](https://readthedocs.org/projects/blend-modes/badge/?version=latest)](https://blend-modes.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://dev.azure.com/florianroscheck/blend_modes/_apis/build/status/flrs.blend_modes?branchName=master)](https://dev.azure.com/florianroscheck/blend_modes/_build/latest?definitionId=1&branchName=master)

Blend Modes
===========
This Python package implements blend modes for images.

**⚠️ This project is no longer maintained. Are you interested in maintaining it? Please file let us know [here](https://github.com/flrs/blend_modes/issues/28). ⚠️**

Description
-----------
The Blend Modes package enables blending different images, or image layers, by means of blend modes. These modes are commonly found in graphics programs like [Adobe Photoshop](http://www.adobe.com/Photoshop) or [GIMP](https://www.gimp.org/).

Blending through blend modes allows to mix images in a variety of ways. This package currently supports the following blend modes (name of the respective functions in the package in `italics`):

* Soft Light (`blend_modes.soft_light`)
* Lighten Only (`blend_modes.lighten_only`)
* Dodge (`blend_modes.dodge`)
* Addition (`blend_modes.addition`)
* Darken Only (`blend_modes.darken_only`)
* Multiply (`blend_modes.multiply`)
* Hard Light (`blend_modes.hard_light`)
* Difference (`blend_modes.difference`)
* Subtract (`blend_modes.subtract`)
* Grain Extract (known from GIMP, `blend_modes.grain_extract`)
* Grain Merge (known from GIMP, `blend_modes.grain_merge`)
* Divide (`blend_modes.divide`)
* Overlay (`blend_modes.overlay`)
* Normal (`blend_modes.normal`)

The intensity of blending can be controlled by means of an *opacity* parameter that is passed into the functions. See [Usage](#usage) for more information.

The Blend Modes package is optimized for speed. It takes advantage of vectorization through Numpy. Further speedup can be achieved when implementing the package in Cython. However, Cython implementation is not part of this package.

Installation
------------
The Blend Modes package can be installed through pip:

```pip install blend_modes```

or conda:

```conda install -c conda-forge blend_modes```

Usage
-----
The blend mode functions take image data expressed as arrays as an input. These image data are usually obtained through functions from image processing packages. Two popular image processing packages in Python are [PIL](https://pypi.python.org/pypi/PIL) or its fork [Pillow](https://pypi.python.org/pypi/Pillow/) and [OpenCV](http://opencv.org/). The examples in this chapter show how to blend images using these packages.
 
### Input and Output Formats

A typical blend mode operation is called like this:

```python
blended_img = soft_light(bg_img, fg_img, opacity)
```

The blend mode functions expect [Numpy](https://pypi.python.org/pypi/numpy) float arrays in the format [*pixels in dimension 1*,*pixels in dimension 2*,4] as an input. Both images needs to have the same size, so the *pixels in dimension 1* must be the same for `bg_img` and `fg_img`. Same applies to the *pixels in dimension 2*. Thus, a valid shape of the arrays would be `bg_img.shape == (640,320,4)` and `fg_img.shape == (640,320,4)`.

The order of the channels in the third dimension should be *R, G, B, A*, where *A* is the alpha channel. All values should be *floats* in the range *0.0 <= value <= 255.0*.

The blend mode functions return arrays in the same format as the input format.

### Examples

The following examples show how to use the Blend Modes package in typical applications.

The examples are structured in three parts:

 1. Load background and foreground image. The foreground image is to be blended onto the background image.
 
 2. Use the Blend Modes package to blend the two images via the "soft light" blend mode.
    The package supports multiple blend modes. See the [Description](#description) for a full list.
    
 3. Display the blended image.

#### PIL/Pillow Example
The following example shows how to use the Blend Modes package with the [PIL](https://pypi.python.org/pypi/PIL) or [Pillow](https://pypi.python.org/pypi/Pillow/) packages.
 
```python
from PIL import Image
import numpy
from blend_modes import soft_light

# Import background image
background_img_raw = Image.open('background.png')  # RGBA image
background_img = numpy.array(background_img_raw)  # Inputs to blend_modes need to be numpy arrays.
background_img_float = background_img.astype(float)  # Inputs to blend_modes need to be floats.

# Import foreground image
foreground_img_raw = Image.open('foreground.png')  # RGBA image
foreground_img = numpy.array(foreground_img_raw)  # Inputs to blend_modes need to be numpy arrays.
foreground_img_float = foreground_img.astype(float)  # Inputs to blend_modes need to be floats.

# Blend images
opacity = 0.7  # The opacity of the foreground that is blended onto the background is 70 %.
blended_img_float = soft_light(background_img_float, foreground_img_float, opacity)

# Convert blended image back into PIL image
blended_img = numpy.uint8(blended_img_float)  # Image needs to be converted back to uint8 type for PIL handling.
blended_img_raw = Image.fromarray(blended_img)  # Note that alpha channels are displayed in black by PIL by default.
                                                # This behavior is difficult to change (although possible).
                                                # If you have alpha channels in your images, then you should give
                                                # OpenCV a try.

# Display blended image
blended_img_raw.show()
```

#### OpenCV Example
The following example shows how to use the Blend Modes package with [OpenCV](http://opencv.org/).
 
```python
import cv2  # import OpenCV
import numpy
from blend_modes import soft_light

# Import background image
background_img_float = cv2.imread('background.png',-1).astype(float)

# Import foreground image
foreground_img_float = cv2.imread('foreground.png',-1).astype(float)

# Blend images
opacity = 0.7  # The opacity of the foreground that is blended onto the background is 70 %.
blended_img_float = soft_light(background_img_float, foreground_img_float, opacity)

# Display blended image
blended_img_uint8 = blended_img_float.astype(numpy.uint8)  # Convert image to OpenCV native display format
cv2.imshow('window', blended_img_uint8)
cv2.waitKey()  # Press a key to close window with the image.
```

Documentation
-------------
Full documentation for the latest version of this package is available on [Read the Docs](https://blend-modes.readthedocs.io/en/latest/).

Dependencies
------------
The Blend Modes package needs [Numpy](https://pypi.python.org/pypi/numpy) to function correctly. For loading images the following packages have been successfully used:
 
 * [PIL](https://pypi.python.org/pypi/PIL)
 * [Pillow](https://pypi.python.org/pypi/Pillow/)
 * [OpenCV](http://opencv.org/)

See Also
--------
Blend modes are further described on [Wikipedia](https://en.wikipedia.org/wiki/Blend_modes). An actual implementation can be found in the [GIMP source code](https://gitlab.gnome.org/GNOME/gimp/blob/master/app/operations/layer-modes-legacy/gimpoperationdividelegacy.c).

Contribution
------------
I am happy about any contribution or feedback. Please let me know about your comments via the Issues tab on [GitHub](https://github.com/flrs/blend_modes/issues).

License
-------
The Blend Modes package is distributed under the [MIT License (MIT)](https://github.com/flrs/blend_modes/blob/master/LICENSE.txt). Please also take note of the licenses of the dependencies.
