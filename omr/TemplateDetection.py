# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:39:10 2020

@author: Yash Kumar (yashkuma)
"""
import numpy as np

def rescale_template(template, size):
    """
    Function to rescale the template image
    :param template: Numpy array of the image
    :param size: Tuple of new_width and new_height. size[0] = new_width, size[1] = new_height
    :return: Numpy array of rescaled image
    """
    width = len(template)  # old width
    height = len(template[0])  # new width

    return np.array([[template[int(width * w / size[0])][int(height * r / size[1])]
             for r in range(size[1])] for w in range(size[0])])

    

def template_match(image,template):
    """
    Function to match template in the image
    :param template: Numpy array of the template (scaled from 0 to 1)
    :param image: Numpy array of the image (scaled from 0 to 1)
    :return: template match image
    """
    # Padding scale image array
    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    pad_im_array = np.zeros(shape=(image.shape[0] + template.shape[0] - 1, image.shape[1] + template.shape[1] -1))
    pad_im_array[:image.shape[0], :image.shape[1]] = image
    # Hamming distance of template and patch calculation
    im_match = np.zeros((image.shape[0],image.shape[1]))
    for i in range(len(image)):
        for j in range(len(image[0])):
            patch = pad_im_array[i:len(template)+i,j:len(template[0])+j]
            im_match[i,j] = np.sum(np.multiply(patch,template)) + np.sum(np.multiply((1-patch),(1-template)))
    # scaling pixel from 0 to 255     
    # converting into image format        
    im_match_scale = 255*(im_match - im_match.min())/(im_match.max() - im_match.min())

    im_match1 = im_match_scale.astype(np.uint8)
    return im_match1


def temp_position(image,size):
    """
    Function to find starting coordinates template in the matched image
    :param image: Numpy array of the matched image 
    :return: template match image
    """    
    row_count=0
    x_coord,y_coord=[],[]
    while row_count < image.shape[0]:
        flag=False
        col_count=0
        while col_count < image.shape[1]:
            if image[row_count][col_count]==255:
                x_coord.append(row_count)
                y_coord.append(col_count)
                flag=True
                col_count+=size[1]
            else:
                col_count+=1
        if flag:
            row_count+=size[0]
        else:
            row_count+=1
    return x_coord, y_coord
