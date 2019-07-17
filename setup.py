from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, './docs/README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='blend_modes',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version='2.1.0',

    description='Image processing blend modes',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/flrs/blend_modes',

    # Author details
    author='Florian Roscheck',
    author_email='florian.ros.check+blendmodes@gmail.com',

    # Choose your license
    license='MIT',
    packages=find_packages(),

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='image processing blend modes',

    install_requires=['numpy'],

    tests_require=['pytest', 'opencv-python']

)
