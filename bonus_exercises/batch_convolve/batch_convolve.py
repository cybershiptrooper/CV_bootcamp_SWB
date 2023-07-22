import numpy as np

def get_padding(img_shape, kernel_shape, out_shape, stride):
    '''
    Returns the padding needed to get the desired output shape.
    :param img_shape: tuple(int, int, int, int)
    :param kernel_shape: tuple(int, int, int, int)
    :param out_shape: tuple(int, int, int, int)
    :param stride: tuple(int, int)
    '''
    return None

def get_out_shape(img_shape, kernel_shape, padding, stride):
    '''
    Returns the output shape given the input shape, kernel shape, padding, and stride.
    :param img_shape: tuple(int, int, int, int)
    :param kernel_shape: tuple(int, int, int, int)
    :param padding: tuple(int, int)
    :param stride: tuple(int, int)
    '''
    return None

def convolve(image, kernel, stride=(1, 1), padding=(0, 0)):
    '''
    Convolve the given image and kernel.
    :param image: numpy array NxCxHxW
    :param kernel: numpy array C_inxC_outxHxW
    :param stride: tuple(int, int)
    :param padding: tuple(int, int)

    :return: numpy array NxC_outxHxW
    '''
    return None

def pad_and_convolve(image, kernel, stride = (1, 1), out_dims=(-1, -1)):
    '''
    Pad the image and convolve it with the kernel. If out_dims is not specified, the output shape is the same as the input shape.
    :param image: numpy array NxCxHxW
    :param kernel: numpy array C_inxC_outxHxW
    :param stride: tuple(int, int)
    :param out_dims: tuple(int, int)
    '''
    return None