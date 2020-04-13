# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:37:25 2020

@author: Hrishikesh Paul (hrpaul)
"""

import imageio
import numpy as np


def separate_kernel(kernel):
    """
    Function to separate kernels into its x and y axes.
    :param kernel: Numpy array of kernel
    :return: 2 1-d separated kernels
    """
    x, y, z = np.linalg.svd(kernel)
    k1 = x[:,0] * np.sqrt(y[0])
    k2 = z[0] * np.sqrt(y[0])

    return k1, k2


def seperable_convolution(kernel, picture):
    """
    Function to perform convolution
    :param kernel: Numpy array of a separable kernel
    :param picture: Numpy array of the image
    :return: Numpy array of convoluted image
    """

    # Creating a placeholder array for image
    conv = np.ones(picture.shape) * 255

    # Separating the kernel into in x and y derivative
    kernel_1, kernel_2 = separate_kernel(kernel)

    # Looping through the the RGB values and columns and convolving
    for i in range(0, len(picture[0][0])):
        for j in range(0, len(picture[0])):
            conv[:, j, i] = np.convolve(picture[:, j, i], kernel_1, 'same')

    # Looping through the RGB values and rows and convolving
    for i in range(0, len(picture[0][0])):
        for j in range(0, len(picture)):
            conv[j, :, i] = np.convolve(conv[j, :, i], kernel_2, 'same')

    return conv


def convolution(kernel, picture):
    """
    Function to perform convolution for non-separable kernels using FFT
    :param kernel: Numpy array of kernel
    :param picture: Numpy array of image
    :return: Numpy array for convoluted imag
    """
    # image = imageio.imread('example.jpg', as_gray=True)
    image_fft = np.fft.fft2(picture)

    padded_kernel = np.zeros(picture.shape)
    padded_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
    kernel_fft = np.fft.fft2(padded_kernel)

    final_fft = np.multiply(kernel_fft, image_fft)
    inverse_fft = np.fft.ifft2(final_fft)
    imageio.imsave('fft-then-ifft.png', inverse_fft.astype(np.uint8))

    return inverse_fft
