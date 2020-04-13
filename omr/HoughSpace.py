# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:06:09 2020

@author: Sumanyu Garg (sumgarg)
"""
# In[0]:
# Importing Libraries that will be used
import cv2
import imageio

import numpy as np
from scipy import stats

from omr.CustomEdgeDetection import edgeMapHough # CustomEdgeDetection includes custom code for edge detection

# In[1]:
def get_img_matrix(file, flag):
    '''This function returns the image matrix from file
    file = image location
    flag was helper variable used during testing to specify which music file (however it is not used in the main function)'''
    img = imageio.imread(file,as_gray = True)
    img_matrix = np.array(img)
    return img_matrix
    

def EdgeDetection(img_matrix , threshold_low = 100, threshold_high = 200, self = False):
    '''This function detects edges on the image
    file = image location
    low,high = helper variables for canny edge detection
    self = True denotes , we are using our own Edge Detection Function and False means we are using open CV edge detection'''
    # img_matrix = cv2.imread(file)
    if(self==False):
        return  cv2.Canny(img_matrix,threshold_low, threshold_high)
    
    if(self==True):
        # img_matrix = np.array(imageio.imread(file, as_gray = True))
        return edgeMapHough(img_matrix)
    
    
# In[2]:
        
def calculateHoughSpace(EdgeMatrix):
    '''This function transforms the edges to hough space depending on 5 group of lines
    EdgeMatrix = edge matrix of the image'''
    
    
    height = EdgeMatrix.shape[0]
    width = EdgeMatrix.shape[1]
    HoughSpace = np.zeros(shape = (height ,height), dtype = int)
    
    white = 255
    
    for i in range(0, height,1):
        for j in range(0, width, 1):
            if(EdgeMatrix[i,j]==white): # Map to Hough Space if the pixel is white
            
            
                for x in range(0,i):
                    if(EdgeMatrix[x,j]==white): 

                        # Case2: Current pixel is the second line
                        HoughSpace[i-x,x]+=1

                        # Case3: Current pixel is the third line 
                        if((i-x)%(3-1)==0):
                            HoughSpace[np.int((i-x)/2),x]+=1;

                        # Case4: Current pixel is the fourth line
                        if((i-x)%(4-1)==0):
                            HoughSpace[np.int((i-x)/3),x]+=1;

                        # Case5: Current pixel is the fifth line
                        if((i-x)%(5-1)==0):
                            HoughSpace[np.int((i-x)/4),x]+=1;
                            
    return HoughSpace

# In[3]:

def calculate_indices(HoughSpace):
    '''This function returns the indices from HoughSpace in decreasing order 
    sorted on the basis of voting count, ycoordinate of pixel, x coordinate of pixel'''
    indices = []
    for i in range(HoughSpace.shape[0]):
        for j in range(HoughSpace.shape[1]):
            indices.append((HoughSpace[i,j], i, j))
            
    indices = sorted(indices, key = lambda x:(x[0],x[2],x[1]))
    indices.reverse()
    return indices


def get_important_indices(indices, nrows):
    '''This fucntion returns indices which are important indices depending on which indices have the spacing parameter equal to a particular value
    which is is chosen to be mode of some number of coordinates in Hough Space'''
    
    important_indices = []
    max_range = np.int(nrows/50)
    max_indices = np.int(nrows/2)
    
    candidate_indices = [indices[i] for i in range(max_range)]
    spacing_parameter = stats.mode([i[1] for i in candidate_indices])[0]
    
    for i in range(max_indices):
        if(indices[i][1]==spacing_parameter):
            important_indices.append(indices[i])
        
    important_indices = sorted(important_indices, key = lambda x:(x[0],x[2]))[::-1]
    
    return important_indices
    

    
def should_be_added(starting_points, x):
    '''Helper function to know if particular coordinate should be added or not'''
    flag = True
    for element in starting_points:
        if(abs(x-element) <50):
            flag = False
    return flag



def get_final_indices(important_indices):
    ''' This function kind of applies compression and selects only final staff positions and spacing parameter'''
    compression = []
    starting_points = set()
    for i in range(len(important_indices)):
        if(i==0):
            compression.append(important_indices[i])
            starting_points.add(important_indices[i][2])
        else:
            ans = should_be_added(starting_points, important_indices[i][2])
            if(ans==True):
                compression.append(important_indices[i])
                starting_points.add(important_indices[i][2])
                
    return compression
    



def get_staff_and_spacing_parameter(EdgeMatrix):
    ''' This function takes input as Edge Matrix and returns the spacing parameter and a list of starting staff positions.'''
    HoughSpace = calculateHoughSpace(EdgeMatrix)
    indices = calculate_indices(HoughSpace)
    important_indices = get_important_indices(indices,EdgeMatrix.shape[0])
    final_indices = get_final_indices(important_indices)
    
    starting_positions = []
    spacing_parameter = final_indices[0][1]
    for x in final_indices:
        starting_positions.append(x[2])
        
    starting_positions = sorted(starting_positions)
    
    return spacing_parameter, starting_positions

