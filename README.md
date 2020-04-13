# MUSIC SHEET RECOGNIZER

For running python file: ./omr/omr.py < music image >

## 1. Convolution 
### Steps
- Convolution with a non-separable kernel:  
   * Convert the image into its respective fourier transformation.
   * Pad the kernel to match the size of the image and then convert it into its respective fourier transformation 
   * Multiply the two transofrmations together.
   * Take the inverse fourier transform of the resultant array to get the convoluted image.
   * The function below demonstrates non-separable convolution:
   ```python
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
   ```
- Convolution with a separable kernel:
   * Separate the kernel into it's x component and y component using the function given below:
   ```python
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
   ```
   * Convolve the columns of the image with the x componenet of the kernel, then convolve the rows with the y component of the kernel.
   * The function below demonstrates separable convolution:
   ```python
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
   ```


## 2. Template Detection
### Steps
- Resizing the template given input template image and size parameters
- Size parameters consists of tuple of height and width. Space parameter has been calculated using edge maps and hough transform to get distance between two consecutive staff lines. Edge detection and Hough transform has been explained in 3rd and 4th point. 
- For template 1 , height is considered to be equal to distance between two consecutive staff lines
- For template 2, height is considered to be equal to 3 times distance between two consecutive staff lines
- For template 3, height is considered to be equal to 2.5 times distance between two consecutive staff lines
- Width is calculated using rescale template function given below:
```python
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
```
- After resizing the template, template and input music image has been scaled from 0 to 1, and calculated the hamming distance for each pixel(starting from top left) using template and input patch from music image of same size. For storing hamming distance of each pixel value padding of music images has been done. 
- The returned array of hemming distance for each pixel is scaled back to 0 - 255 pixel value to get the resultant image. The template match function is given below :
```python
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
            im_match[i,j] = np.sum(np.multiply(patch,template)) + np.sum(np.multiply((1-patch),(1-template)))
    # scaling pixel from 0 to 255     
    # converting into image format        
    im_match_scale = 255*(im_match - im_match.min())/(im_match.max() - im_match.min())

    im_match1 = im_match_scale.astype(np.uint8)
    return im_match1
```
- High scoring pixel values has been set to 255 and rest as 0 
- Cutoff has been decided based on manual accuracy check for music1.png, music2.png and music3.png
- Last 5 columns of match image has been deleted because of noise during resizing and padding. 
- For drawing box and text around the matched template IMAGE.draw function has been used from Pillow library
- Top left x and y coordinate has been calculated of matched template position in a music image given image and size of resized template. The function for finding template position has been given below:
```python
def temp_position(image,size):
    """
    Function to find starting coordinates template in the matched image
    :param image: Numpy array of the matched image
    :param size: size of rescaled template array 
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
```
- Text of note type and other rests has been drawn 10 pixel away from top left position of matched templates in a music image
- Confidence score for each template has been calculated using the formula give below:
Score = (# of pixels in matched template music image(where value = 255))/(Number of positions detected for template in music image) * (height x width of rescaled template)

### Assumptions
- Assumed the given music image is of high quality with no distortions. That's why it works well for music1.png but give poor results for other music images especially for template2 and template3 matching. 
- For less complex calculations, assumed confidence score for a given template will be same in a music image
- Last 5 columns of matched template image has been removed assuming there will be no templates in that area

### Challenges
- Very less accuracy of template 2 and template 3 in music2.png and music3.png because of noises in the image. Works well for high quality images. 
- Trade off of cutoff score has been done because of too many noisy detection for music1.png and music3.png
- High computational cost for template matching using pixel by pixel hamming distace calculation

## 3. Edge Detection and Distance Transform
### Steps
- For edge detection, I use Horizontal and Vertical Sobel kernels, and convolve them with the image and template using seperable convolution.
- I calculated the pixelwise euclidean- distance between the output of vertical convolution and horizontal convolution.
- An edge map is created by using a threshold above which all values are edges i.e 1 and rest are 0.
- Below is function for creating the edgemaps to be used for transforming the image to hough space.
```python
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
```

- Below is function for creating the edgemaps to be used for calculating the distance transform.  

```python
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
```

- For calculating the distance transform of the image, I use the edge map of the image, find the edge pixels in O(n^2) time and then calculate the distance of every pixel to the edge pixels in O(n^2xk) time where k is the number of edge pixels. See edgeDist.

```python
def edgeDist(picture):

    edges = np.argwhere(picture==1)
    distance = np.full(picture.shape, np.inf)
    for i in range(picture.shape[0]):
        for j in range(picture.shape[1]):
            for k in range(edges.shape[0]):
                distance[i,j] = min(distance[i,j],np.sqrt((edges[k,0]-i)**2 + (edges[k,1]-j)**2))

    return distance
```
-The time complexity for this approach is very high. Alternatively, I borrowed a program for O(n^2) implementation which works within seconds.
- Source: http://www.logarithmic.net/pfh/blog/01185880752

```python
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
```
### Assumptions
- I have used 3x3 sobel kernels. 
- For the distance transform, I have assumed that the edge detector detects edges with good accuracy. 
- I have also assumed that there are no multiple edges and haven't used non-max suppression as the edgemaps didn't show noise when visualized.
### Challenges
- Setting the threshold of the edge detector required multiple trials with a range of thresholds to get an accurate edge map.
- My implementation of edge transform is more efficient than O(n^4) but still takes a lot of time. So had to search online for a better implementation.
### Outcomes
- For the right threshold, the edge detection and consequently the distance transform work perfectly well. The edge detector is further used for transforming the image to hough space for getting the scaling parameters.

## 4. Transforming to Hough Space
### Steps
- As an initial step, I have calculated the edgeMatrix from the image which hives white pixel values wherever the edges are and black pixel values to non edge pixels. For Edge Detection I have used our both our own implemented edgeMapHough function and the openCV canny edge detection function.
self = True denotes we will be using our implemented version for edge detection and false denotes we will be using Canny edge detection.
```python
def EdgeDetection(file , threshold_low = 100, threshold_high = 200, self = False):
    '''This function detects edges on the image
    file = image location
    low,high = helper variables for canny edge detection
    self = True denotes , we are using our own Edge Detection Function and False means we are using open CV edge detection'''
    img_matrix = cv2.imread(file)
    if(self==False):
        return  cv2.Canny(img_matrix,threshold_low, threshold_high)
    
    if(self==True):
        img_matrix = np.array(imageio.imread(file, as_gray = True))
        return edgeMapHough(img_matrix)
```

- After getting edge detection matrix, I have transformed each white pixel in the edgematrix to hough space based on the following logic.
Whenever I come across a white pixel, I give votes to (spacing_parameter, start_position) coordinate in HoughSpace based on whether this pixel belongs to the second line or the third line or the fourth line or the fifth line of the satffand consecutively calculating spacing parameter.
```python

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
    
```


- Once I have transformed white pixels into the Hough space, I select the most deserving candidates based on the following steps.<br>
 (1) I store coordinates from the hough space in a increasing order sorted on the basis of votes, then the start_position of the staff and then the spacing parameter.

```python
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
```
(2) I then calculate which indices are important indices returned by above function based on first finding the mode of the spacing       parameter and whichever coordinates have that spacing parameter, selecing those coordinates.

```python
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
```
(3) I then apply compression to remove nearby coordinates and only selecting coordinates which are far away. 

```python
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
```
(4) I have combined all the above functions using one function to increase the user interface for the code. The following function just takes edge matrix as an input and returns best spacing parameter and a list of possible start positions for the staff lines.

```python
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

```
### Assumptions
- The implementation of above HoughTransform depends on the edge detection output. I have assumed that the staff lines are detected with good accuracy. I have not assumed that the detection is not noiy but less the noise better the algorithm works.
- I have also assumed that the staff lines are exactly parallel to each other even for noisy images.
### Challenges
- There could have been a better algorithm for transforming into HoughSpace which doesn't take onto account every white pixel but a group of white pixels in a line. This could make the detection more robust and prone to noise.
- Compression technique that is used to get only far away coordinates can be improved by some kind of union find algorithm where find operation returns true only for nearby points.(in this scenario, we would have to define 'nearby' mathematically). Once union find is run over all the candidtae pixels, we could take average of the coordinates. This would agin make the algorithm more robust to noise.
### Outcomes
- Works perfectly well for music1,music2, music3. In music 4 , my code detects the spacing parameter correctly but not the start positions of the staff. For rach file, my code gives starting position of staff somewhere in between the staff lines , however the spacing parameter is off by great margin.


## 5. Note Detection 
### Steps
- I first find out whether the starting position of the clef belongs to a treble clef or for a bass clef.
```python
ef get_clef_positions(starting_positions):
    '''Ths function classifies if the staff is treble cleff or bass clef'''
    treble_clef, bass_clef = [], []
    for i in range(len(starting_positions)):
        if(i%2==0):
            treble_clef.append(starting_positions[i])
        else:
            bass_clef.append(starting_positions[i])
            
    return treble_clef, bass_clef
            
```
- I then find out whether a note belongs to treble clef or to bass clef. I do this for each note position.
```python
def belongs_which_clef(treble_clef, bass_clef, spacing_parameter,starting_positions, detected):
    ''' This function returns which notes belong to which clef
    0 denotes treble clef 
    1 denotes bass clef'''
    white = 255
    
    clef_dict = dict()
    for i in range(detected.shape[0]):
        for j in range(detected.shape[1]):
            if(detected[i,j]==white):
                #print(i,j)
                for clef in treble_clef:
                    if(clef - 3*spacing_parameter < i and i< clef + 8*spacing_parameter):
                        clef_dict[(i,j)] = 0
                        
                for clef in bass_clef:
                    if (i,j) not in clef_dict.keys():
                        clef_dict[(i,j)] = 1
    
    return clef_dict
```
- I then find out whether a given note is A,B,C,D,E,F or G based on the dictionary returned by above function. I have taken into account a parameter max_error = spacing_parameter/2 which is used to specify the range on y axis where to look for the note.
```python
def check_condition(i, position_dict_1, position_dict_2):
    '''This is a helper function to check if a note lies in a given range or not'''
    for x in range(len(position_dict_1)):
        condition = position_dict_1[x] < i and i < position_dict_2[x]
        if (condition==True):
            return True
        
    return False


def which_note_is_it(i, clef, max_error, note, first_encounter, spacing_parameter):
    '''This function specifies if a note is particular or not'''
    candidate_position_1 = [int(x + first_encounter - max_error) for x in clef]
    candidate_position_2 = [int(x + first_encounter + max_error) for x in clef]
    #print(candidate_position_1, candidate_position_2)
    condition_1 = check_condition(i, candidate_position_1, candidate_position_2)
    
    candidate_position_1 = [int(x + first_encounter + spacing_parameter*3.5 - max_error) for x in clef]
    candidate_position_2 = [int(x + first_encounter + spacing_parameter*3.5 + max_error) for x in clef]
    condition_2 = check_condition(i, candidate_position_1, candidate_position_2)
    
    candidate_position_1 = [int(x + first_encounter - spacing_parameter*3.5 - max_error) for x in clef]
    candidate_position_2 = [int(x + first_encounter - spacing_parameter*3.5 + max_error) for x in clef]
    condition_3 = check_condition(i, candidate_position_1, candidate_position_2)
    
    if(condition_1 or condition_2 or condition_3):
        return True
    
    return False
            
def create_notes_dict(clef_dict, treble_clef, bass_clef, spacing_parameter):
    ''' This function specifies which note each note is'''
    max_error = spacing_parameter/2
    notes = []

    for index,clef in clef_dict.items():
        if(clef==0):
            i = index[0]
            j = index[1]
            
            # Detecting which notes are F
            if(which_note_is_it(i, treble_clef, max_error, 'F', 0, spacing_parameter) == True):
                notes.append((i,j,'F'))
            
            # Detecting which notes are D
            elif(which_note_is_it(i, treble_clef, max_error, 'D', spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'D'))
            
            # Detecting which notes are B
            elif(which_note_is_it(i, treble_clef, max_error, 'B', 2*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'B'))
              
            # Detecting which notes are G
            elif(which_note_is_it(i, treble_clef, max_error, 'G', 3*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'G'))
            
            # Detecting which notes are E
            elif(which_note_is_it(i, treble_clef, max_error, 'E', 4*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'E'))
                
            # Detecting which notes are A
            elif(which_note_is_it(i, treble_clef, max_error, 'A', 2.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'A'))
                
            #Detecting which notes are C
            elif(which_note_is_it(i, treble_clef, max_error, 'C', 1.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'C'))
                
            # Whatever is left
            else:
                l = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                notes.append((i,j,np.random.choice(l)))
                
        if(clef==1):
            i = index[0]
            j = index[1]
            
            # Detecting which notes are A
            if(which_note_is_it(i, bass_clef, max_error, 'A', 0, spacing_parameter) == True):
                notes.append((i,j,'A'))
                
            # Detecting which notes are F
            elif(which_note_is_it(i, bass_clef, max_error, 'F', spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'F'))
                
            # Detecting which notes are D
            elif(which_note_is_it(i, bass_clef, max_error, 'D', 2*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'D'))
                
            # Detecting which notes are B
            elif(which_note_is_it(i, bass_clef, max_error, 'B', 3*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'B'))
                
            # Detecting which notes are G
            elif(which_note_is_it(i, bass_clef, max_error, 'G', 4*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'G'))
                
            # Detecting which notes are C
            elif(which_note_is_it(i, bass_clef, max_error, 'C', 2.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'C'))
                
            # Detecting which notes are E
            elif(which_note_is_it(i, bass_clef, max_error, 'E', 1.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'E'))
            # Whatever is left
            else:
                l = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                notes.append((i,j,np.random.choice(l)))
            
    return notes  
```
- I have then combine all the function above to increase the user interface. The below function just accepts three parameters. The spacing, a list of start_position of the staff and the matrix denoting the note positions as 255 and 0 everywhere else.
```python
def specify_notes(spacing_parameter, starting_positions, detected):
    ''' This function combines all above functions and returns final output as a dictionary consisting of coordinates of each note as key and value as what the note is'''
    treble_clef, bass_clef = get_clef_positions(starting_positions)
    clef_dict = belongs_which_clef(treble_clef, bass_clef, spacing_parameter, starting_positions, detected)    
    notes = create_notes_dict(clef_dict, treble_clef, bass_clef, spacing_parameter)
    return notes
```



## 6. Overall
- The overall note detection works well given any input music image. For eigth rests and quarter rest, it doesn't work well because of noisy images and many similar type patterns in the music image
- For overall accurate results, all the steps explained above should be very accurate because in any Computer Vision problem, if any step is not accurate, it affects accuracy in bigger picture of the problem
- As said above, if edge detection is not exact, it will have an affect on the spacing parameter which will have an effect in template detection and note detection further
- Edge matrix calculation has been done using the steps explained in the assignment part 6 as well as using 'opencv' library in Python.
- The function called in main omr.py file for calculating edge matrix is done below :
```python
EdgeMatrix = EdgeDetection(os.path.join(DATA_DIR, music_file), self = False)
```
- If parameter self = True, then it calculates edge matrix using the steps explained in the assignment part 6. The result using that is not at all accurate for music2.png so we have incorporated both arguments in our assignment submission.
 - In main.py, while drawing boxes around notes I have shifted the position of the box up spacing parameter/2. This is done because while calculating the which kind of notes which notes are, I had shifted the position by spacing parameter/2 downwords because the positons of notes detected were based on the first time the white pixel in that region occured. To get correct position of the note I shifted the note y coordinate by spacing parameter/2 and while drawing boxes, I reshifted the postions by that amount.
### Accuracy of template detection on test music images
- music1.png : template1 (89%)
               template2 (100%)
               template3 (100%)
             
- music2.png : template1 (50%)
- music3.png : template1 (35%)

- For next steps, we can work upon the strengthing the image quality of given music image and then using template matching method for more accurate results. Non-maximal suppression and canny edge detection can also improve the result of part 6 of the assignment which has not been incorporated right now. Each step of the problem is very crucial and need to improve the accuracy of each parts, only then we can achieve high accuracy on any given music images. 

## 7. Results

- Results for test-images/music1.png:

![detected.png](python-sample/detected.png)
