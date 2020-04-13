# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:24:10 2020

@author: Sumanyu Garg (sumgarg) and Mahesh Latnekar (mrlatnek)
"""

import imageio
from PIL import Image, ImageFilter
import numpy as np
import cv2 as cv

# scaling function
def scaling(arr):
    """
    Function to scale image for convolution purpose
    image : takes image as input
    """
    #arr = np.array(image)
    scale = (arr - arr.min())/(arr.max() - arr.min())
    return scale

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
    #####need to account for boundary effects
    # Creating a placeholder array for image
    conv = np.ones(picture.shape) * 255

    # Separating the kernel into in x and y derivative
    kernel_1, kernel_2 = separate_kernel(kernel)
    
    # Looping through the the RGB values and columns and convolving
    for i in range(picture.shape[1]):
            conv[:, i] = np.convolve(picture[:, i], kernel_1, 'same')

    # Looping through the RGB values and rows and convolving
    for i in range(picture.shape[0]):
            conv[i, :] = np.convolve(conv[i, :], kernel_2, 'same')

    return conv



def edgeMapHough(picture):
    """
    Function to find edge maps using sobel operator and seperable
    convolution for Hough Transform
    """
    sobel_hor = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]) * 1/8
    sobel_ver = np.array([[1,2,1], [0,0,0], [-1,-2,-1]]) * 1/8
    image_hor = seperable_convolution(sobel_hor, picture)
    image_ver = seperable_convolution(sobel_ver, picture)
    edge_map = np.sqrt(np.square(image_hor) + np.square(image_ver))
    edge_map[edge_map >= 0.25]=255
    edge_map[edge_map < 0.25]=0
    edge_map[0,:]=0
    edge_map[-1,:]=0
    edge_map[:,0]=0
    edge_map[:,-1]=0
    edge_map = np.uint8(edge_map)
    #imageio.imwrite('test-images/t1_image_edge.png', edge_map)        
    return edge_map

def edgeMapDistance(picture,ind):
    """
    Function to find edge maps using sobel operator and seperable
    convolution for the purpose of distance transformation as described in 
    part 6
    """
    sobel_hor = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]) * 1/8
    sobel_ver = np.array([[1,2,1], [0,0,0], [-1,-2,-1]]) * 1/8
    image_hor = seperable_convolution(sobel_hor, picture)
    image_ver = seperable_convolution(sobel_ver, picture)
    edge_map1 = np.sqrt(np.square(image_hor) + np.square(image_ver))
    edge_map1[edge_map1 >= 0.25]=255
    edge_map1[edge_map1 < 0.25]=0
    edge_map1[0,:]=0
    edge_map1[-1,:]=0
    edge_map1[:,0]=0
    edge_map1[:,-1]=0
    edge_map1 = np.uint8(edge_map1)
    #imageio.imwrite('test-images/t2_temp_edge.png', edge_map1)
    if ind==1:
        #imageio.imwrite('test-images/t1_image_edge.png', edge_map1)
    
        edge_map = np.sqrt(np.square(image_hor) + np.square(image_ver))
        edge_map[edge_map >= 0.25]=1
        edge_map[edge_map < 0.25]=0
        edge_map[0,:]=0
        edge_map[-1,:]=0
        edge_map[:,0]=0
        edge_map[:,-1]=0
    else:
        edge_map = np.sqrt(np.square(image_hor) + np.square(image_ver))
        edge_map[edge_map >= 0.25]=1
        edge_map[edge_map < 0.25]=0
        edge_map[0,:]=0
        edge_map[-1,:]=0
        edge_map[:,0]=0
        edge_map[:,-1]=0
        
    return np.uint8(edge_map)

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
    
    return inverse_fft


def template_match(image,template):
    """
    Function to match template in the image
    :param template: Numpy array of the template (scaled from 0 to 1)
    :param image: Numpy array of the image (scaled from 0 to 1)
    :return: template match image
    """
    # Padding scale image array
    pad_im_array = np.zeros(shape=(image.shape[0] + template.shape[0] - 1, image.shape[1] + template.shape[1] -1))
    pad_im_array[:image.shape[0], :image.shape[1]] = image
    # Hamming distance of template and patch calculation
    im_match = np.zeros((image.shape[0],image.shape[1]))
    for i in range(len(image)):
        for j in range(len(image[0])):
            patch = pad_im_array[i:len(template)+i,j:len(template[0])+j]
            im_match[i,j] = np.sum(np.multiply(patch,template)) 
    # scaling pixel from 0 to 255     
    # converting into image format        
    im_match_scale = 255*(im_match - im_match.min())/(im_match.max() - im_match.min())

    im_match1 = im_match_scale.astype(np.uint8)
    return im_match1


""" 
The below code for Distance Transform which takes O(n^2) is taken from
http://www.logarithmic.net/pfh/blog/01185880752
"""
def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x
   

def edgeDistance(picture):
    f = np.where(picture, 0.0, np.inf)
    for i in range(f.shape[0]):
        _upscan(f[i,:])
        _upscan(f[i,::-1])
    for i in range(f.shape[1]):
        _upscan(f[:,i])
        _upscan(f[::-1,i])
    np.sqrt(f,f)
    imageio.imwrite('test-images/im1_distance_map.png', np.uint8(f))

    return f    
"""
This an alternative code for distance transform which takes O(n^4) order of computation
"""
def edgeDist(picture):

    edges = np.argwhere(picture==1)
    distance = np.full(picture.shape, np.inf)
    for i in range(picture.shape[0]):
        for j in range(picture.shape[1]):
            for k in range(edges.shape[0]):
                distance[i,j] = min(distance[i,j],np.sqrt((edges[k,0]-i)**2 + (edges[k,1]-j)**2))

    return distance
