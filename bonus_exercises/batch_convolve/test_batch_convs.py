from .batch_convolve import *
from functools import partial

"""
You can run this file using pytest
"""

def test_convolution():
    input = np.ones((10, 2, 5, 5))
    kernel = np.ones((2, 1, 3, 3))
    convolved = convolve(input, kernel, stride=(1, 1), padding=(0, 0))
    assert convolved.shape == (10, 1, 3, 3)

def test_get_paddings_fn():
    to_execute = partial(get_padding, (5, 5), (3, 3), (3, 3))
    ans1 =  to_execute((1, 1)) 
    assert (5 - 3 + 2*ans1[0])//1 + 1 == 3
    assert (5 - 3 + 2*ans1[1])//1 + 1 == 3

    ans2 = to_execute((2, 2))
    assert (5 - 3 + 2*ans2[0])//2 + 1 == 3
    assert (5 - 3 + 2*ans2[1])//2 + 1 == 3

def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def test_pad_and_convolve():
    input = np.ones((3, 1, 5, 5))
    kernel = np.ones((1, 2, 3, 3))
    convolved = pad_and_convolve(input, kernel, stride=(1, 1), out_dims=(3, 3))
    assert convolved.shape == (3, 2, 3, 3)
    assert (conv2d(input[0][0], kernel[0][0]) == convolved[0][0]).all()

def test_non_symmetric_strides():
    input = np.ones((10, 1, 5, 5))
    kernel = np.ones((1, 2, 3, 3))
    convolved = pad_and_convolve(input, kernel, stride=(1, 2), out_dims=(3, 3))
    assert convolved.shape == (10, 2, 3, 3)

def test_conv_on_values():
    # test simple convolution and assert elementwise result with what is expected
    input = np.ones((1, 1, 5, 5))
    kernel = np.ones((1, 1, 3, 3))
    convolved = convolve(input, kernel, stride=(1, 1), padding=(0, 0))
    assert (convolved == 9).all()

    # test convolution with stride and assert elementwise result with what is expected
    input = np.ones((1, 1, 5, 5))
    kernel = np.ones((1, 1, 3, 3))
    convolved = convolve(input, kernel, stride=(2, 2), padding=(2, 2))
    assert (convolved == np.array([[[ [1, 3, 3, 1], 
                                      [3, 9, 9, 3],
                                      [3, 9, 9, 3],
                                      [1, 3, 3, 1] ]]])
                                      ).all()

