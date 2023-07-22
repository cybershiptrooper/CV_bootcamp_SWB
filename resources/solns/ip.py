import numpy as np
import matplotlib.pyplot as plt

# plot corrected image with a slider to change gamma
def gamma_correct(img, gamma):
    corrected = img**gamma
    # plot the histogram and image side by side
    hist, bins = np.histogram(corrected, bins=256)
    cfd = np.cumsum(hist)
    cfd = cfd * float(hist.max())  / cfd[-1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(corrected, cmap='gray')
    ax[1].plot(cfd, color = 'r')
    ax[1].hist(corrected.flatten()*256,256,[0,256], color = 'b')
    plt.show()

def convolve(img, kernel):
    strided = np.lib.stride_tricks.sliding_window_view(img, kernel.shape)
    return np.einsum('ijkl,kl->ij', strided, kernel)

def nonlinear_convolve(img, kernel_func, window):
    strided = np.lib.stride_tricks.sliding_window_view(img, window)
    np.apply_over_axes(kernel_func, strided, [-1, -2]).reshape(2, 2)