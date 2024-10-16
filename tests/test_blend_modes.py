"""This module contains code for testing the blend_modes package
"""
import cv2
import pytest
import os

from blend_modes import *
from blend_modes.type_checks import assert_opacity, assert_image_format

_TEST_LIMIT = 10  # test fails if max. image color difference is > test_limit
_TEST_TOLERANCE = 0.001  # max. ratio of RGBA pixels that may not match test criteria

# Change current directory to directory of this test file so relative paths work
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def _test_criteria(out, comp):
    return (np.sum(np.absolute(out - comp) > _TEST_LIMIT)) / np.prod(comp.shape) < _TEST_TOLERANCE


@pytest.fixture
def img_in():
    return cv2.imread('./orig.png', -1).astype(float)


@pytest.fixture
def img_layer():
    return cv2.imread('./layer.png', -1).astype(float)


@pytest.fixture
def img_layer_50p():
    return cv2.imread('./layer_50p.png', -1).astype(float)


def test_addition(img_in, img_layer):
    out = soft_light(img_in, img_layer, 0.5)
    comp = cv2.imread('./soft_light.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_darken_only(img_in, img_layer):
    out = darken_only(img_in, img_layer, 0.5)
    comp = cv2.imread('./darken_only.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_multiply(img_in, img_layer):
    out = multiply(img_in, img_layer, 0.5)
    comp = cv2.imread('./multiply.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_difference(img_in, img_layer):
    out = difference(img_in, img_layer, 0.5)
    comp = cv2.imread('./difference.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_divide(img_in, img_layer):
    out = divide(img_in, img_layer, 0.5)
    comp = cv2.imread('./divide.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_dodge(img_in, img_layer):
    out = dodge(img_in, img_layer, 0.5)
    comp = cv2.imread('./dodge.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_grain_extract(img_in, img_layer):
    out = grain_extract(img_in, img_layer, 0.5)
    comp = cv2.imread('./grain_extract.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_grain_merge(img_in, img_layer):
    out = grain_merge(img_in, img_layer, 0.5)
    comp = cv2.imread('./grain_merge.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_hard_light(img_in, img_layer):
    out = hard_light(img_in, img_layer, 0.5)
    comp = cv2.imread('./hard_light.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_lighten_only(img_in, img_layer):
    out = lighten_only(img_in, img_layer, 0.5)
    comp = cv2.imread('./lighten_only.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_soft_light_50p(img_in, img_layer_50p):
    out = soft_light(img_in, img_layer_50p, 0.8)
    comp = cv2.imread('./soft_light_50p.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_overlay(img_in, img_layer):
    out = overlay(img_in, img_layer, 0.5)
    comp = cv2.imread('./overlay.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_normal_50p(img_in, img_layer):
    out = normal(img_in, img_layer, 0.5)
    comp = cv2.imread('./normal_50p.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_normal_100p(img_in, img_layer):
    out = normal(img_in, img_layer, 1.0)
    comp = cv2.imread('./normal_100p.png', -1).astype(float)
    assert _test_criteria(out, comp)


def test_assert_image_format_dims_force_alpha():
    with pytest.raises(TypeError):
        assert_image_format(np.ndarray(dtype=float, shape=[640, 640, 3]), fcn_name='', arg_name='',
                             force_alpha=True)


def test_assert_image_format_dims_not_force_alpha():
    assert_image_format(np.ndarray(dtype=float, shape=[640, 640, 3]), fcn_name='', arg_name='', force_alpha=False)


def test_assert_image_format_dims():
    with pytest.raises(TypeError):
        assert_image_format(np.ndarray(dtype=float, shape=[640, 640, 2]), fcn_name='', arg_name='')


def test_assert_image_format_shape():
    with pytest.raises(TypeError):
        assert_image_format(np.ndarray(dtype=float, shape=[640, 640]), fcn_name='', arg_name='')


def test_assert_image_format_kind():
    with pytest.raises(TypeError):
        assert_image_format(np.ndarray(dtype=int, shape=[640, 640, 4]), fcn_name='', arg_name='')


def test_assert_image_format_type():
    with pytest.raises(TypeError):
        assert_image_format(2.0, fcn_name='', arg_name='')


def test_assert_opacity_wrong_variable_type():
    opacity = '0.5'
    with pytest.raises(TypeError):
        assert assert_opacity(opacity, '')


def test_assert_opacity_right_variable_type():
    assert_opacity(0.5, '')


@pytest.mark.parametrize('opacity', [-5.0, 1.01])
def test_assert_opacity_wrong_variable_range(opacity):
    with pytest.raises(ValueError):
        assert assert_opacity(opacity, '')
