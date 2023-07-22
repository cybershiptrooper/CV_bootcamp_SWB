import numpy as np

def get_padding(img_shape, kernel_shape, out_shape, stride):
    ans = []
    for i in [-2, -1]:
        w = img_shape[i]
        f = kernel_shape[i]
        pad = (out_shape[i] - 1) * stride[i] + f - w 
        ans.append(pad//2)
    return ans

def get_out_shape(img_shape, kernel_shape, padding, stride):
    ans = []
    for i in [-2, -1]:
        w = img_shape[i]
        f = kernel_shape[i]
        pad = padding[i]
        ans.append((w - f + 2*pad)//stride[i] + 1)
    return ans

def convolve(image, kernel, stride=(1, 1), padding=(0, 0)):
    '''
    Convolve the given image and kernel.
    :param image: numpy array NxCxHxW
    :param kernel: numpy array C_inxC_outxHxW
    :param stride: tuple(int, int)
    :param padding: tuple(int, int)

    :return: numpy array NxC_outxHxW
    '''
    padded = np.pad(image, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]) ))
    strided = np.lib.stride_tricks.sliding_window_view(
            padded, 
            window_shape=[kernel.shape[-2], kernel.shape[-1]],
            axis=(-2, -1)
    )[:, :, ::stride[0], ::stride[1], ...]
    return np.einsum('ncijkl,cokl->noij', strided, kernel)

def pad_and_convolve(image, kernel, stride = (1, 1), out_dims=(-1, -1)):
    if(out_dims[-2] == -1):
        out_dims[-2] = image.shape[-2]
    if(out_dims[-1] == -1):
        out_dims[-1] = image.shape[-1]
    padding = get_padding(image.shape, kernel.shape, out_dims, stride)
    assert(get_out_shape(image.shape, kernel.shape, padding, stride) == list(out_dims))
    return convolve(image, kernel, stride, padding)