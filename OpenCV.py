#!/usr/bin/env python
# coding: utf-8

# # OPENCV(Open Source Computer Vision)
# 
# 
# 

# ## DAY 1

# * It is an image processing library
# 

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# **imread( *image_name*, *any of the options if required*)** : acts as a link for the image

# In[ ]:


img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)


# **Alpha Channel** : Degree of opaqueness
# 
# In an image we deal with 4 major colors BGR and a. Therefore we make use of grayscale(IMREAD_GRAYSCALE) as it rules out 'a' and is much easier to process.
# 
# Options include:
# * IMREAD_GRAYSCALE = 0
# * IMREAD_COLOR = 1
# * IMREAD_UNCHANGED = -1

# ### To show the image using **opencv** -
# 
# **imshow( *window_name*, *image_link* )** : to show the image in a seperate window
# 
# **waitKey** : waitkey() function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed. It takes time in milliseconds as a parameter and waits for the given time to destroy the window, if 0 is passed in the argument it waits till any key is pressed. 

# In[ ]:


cv2.imshow('image',img)  #'image' here is the title of the window
cv2.waitKey(0)           #waits for any key to be pressed 
cv2.destroyAllWindows()  #once any key is pressed, the image windows get destroyed


# ### To show the image using **matplotlib** -
# 
# **imshow( *\<image_link\>*, cmap = '' *if needed*, interpolation = '' *if needed* )** : to show the image in a seperate window

# In[ ]:


plt.imshow(img)
#plt.show()


# **NOTE** : 
# OpenCV uses BGR format while Matplotlib uses RGB format hence when used might produce different results.

# **imwrite** : to save an image in your directory.
# 
# **cv2.imwrite( \<*name_with_which_we_want_to_save_the_image\>, \<image_link\>* )**

# ### Plotting a graph using matplotlib (EXTRA)

# In[ ]:


# create data 
x = [1,2,3,4,5] 
y = x

# y = = np.array([1,2,3,4,5])
# y = = x*x
  
# plot the graph 
plt.plot(x, y, linewidth=1, alpha=0.3)  # alpha=1 means max opacity; values reduce the opacity
plt.show() 


# ## DAY 2

# ### To show a live video from a desktop camera using opencv

# In[ ]:


# 0 in arguement if only 1 camera present, 1,2 or more if other camera also present
cap = cv2.VideoCapture(0) 
 
while(True):
    # ret stands for return and get the value True while frame acts reads the video link
    ret, frame = cap.read()  
    # function applied to frame to change BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    
    # for normal feed
    cv2.imshow('frame',frame)
    
    # for grayscale feed
    #cv2.imshow('frame',gray)  
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# to close the camera just like closing a file
cap.release()   
cv2.destroyAllWindows()


# ### To save the recorded video file 

# **FOURCC Codec** is short for "four character code encoder and decoder (or possibly compressor-decompressor)" - an identifier for a video compression format, color or pixel format and audio tags are used within a media file.
# 
# * So, a video codec is a hardware or software that compresses and decompresses digital video, to make file sizes smaller and storage and distribution of the videos easier. Additionally, codecs are used to optimize files for playback. At the most basic level, a video codec applies an algorithm that compresses video files into a “container format.” When the video files are transported (particularly across the internet) the codec decompresses them so they’re suitable for viewing.
# 
# * AVI files is the most widespread, or the first widely used media file format.
# 
# * Some of the well known FOURCCs include "DIVX", "XVID", "H264", "DX50". But these are just a few of the hundreds in use.
# 
# **fourcc = cv2.VideoWriter_fourcc( \* '\<any codec\>')**
# 
# **out = cv2.VideoWriter( '\<name_of_the_file_to_be_saved'\> , fourcc, \<fps\>, \<video_size\>)**

# In[ ]:


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
while True:
    # to capture/read the video frame by frame
    ret, frame = cap.read() 
    # to save the video be recorded in the avi file
    out.write(frame)   
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
# to close the saved file
out.release() 
cv2.destroyAllWindows()


# ### To view a pre-recorded video file

# In[ ]:


import cv2 
import numpy as np 
  
# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('output2.mp4') 
  
# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 
  
# Read until video is completed 
while(cap.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = cap.read() 
    if ret == True: 
    # Display the resulting frame 
        cv2.imshow('Frame', frame) 
          
    # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
  
# Break the loop 
    else: 
        break

cap.release() 
cv2.destroyAllWindows() 


# ## DAY 3

# ### Drawing lines/shapes on image

# In[ ]:


import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)


# **cv2.line( \<image_link\>, \<starting point coordinate\>, \<ending point coordinate\>, \<BGR codes\>, \<width_of_line\>)**

# In[ ]:


cv2.line(img,(0,0),(150,150),(255,255,255),15)


# **cv2.rectangle( \<image_link\>, \<top-left coordinate\>, \<bottom-right coordinate\>, \<BGR codes\>, \<width_of_line\>)**

# In[ ]:


cv2.rectangle(img,(15,25),(200,150),(0,0,255),15)


# **cv2.circle( \<image_link\>, \<center coordinate\>, \<radius\>, \<BGR codes\>, \<width_of_line\>**
# 
# if width of a shape is -1, it fills the shape

# In[ ]:


cv2.circle(img,(100,63), 55, (0,255,0), -1)


# **cv2.polylines( \<image_link\>, \<numpy_array\>, \<True to connect the starting and ending point else False\>, \<BGR codes\>, \<width_of_line\>**

# In[ ]:


# initialize the points of the polygon
pts = np.array([[100,50],[20,30],[40,60],[80,90]])

cv2.polylines(img, [pts], True, (0,255,255), 1)


# In[ ]:


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### Writing text on the image

# In[ ]:


import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)


# **font = cv2.\<font_type\>**
# 
# Font styles available in opencv are
# 
# FONT_HERSHEY_SIMPLEX = 0
# 
# FONT_HERSHEY_PLAIN = 1
# 
# FONT_HERSHEY_DUPLEX = 2
# 
# FONT_HERSHEY_COMPLEX = 3
# 
# FONT_HERSHEY_TRIPLEX = 4
# 
# FONT_HERSHEY_COMPLEX_SMALL = 5
# 
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6
# 
# FONT_HERSHEY_SCRIPT_COMPLEX = 7
# 

# In[ ]:


font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX = 7


# **cv2.putText( \<image_link\>, <\font_to_be_shown\>, <\starting_point\>, font, \<font_size\>, <BGR_code>\, \<width>\)**

# In[ ]:


cv2.putText(img, 'OpenCv', (50,130), font, 1, (200,255,10), 2)


# In[ ]:


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## DAY 4

# ### Setting the color of a particular pixel in an image

# In[ ]:


import cv2
import numpy as np

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)


# In[ ]:


# referencing specific pixels
px = img[55,55]


# In[ ]:


# setting the color of the pixel
px = [255,255,255]


# In[ ]:


print(px)


# ### ROI : Region of an image

# **img\[ \<lower_y-coord\> : \<upper_y-coord\> , \<left_x-coord\> : \<right_x-coord\> ]**
# 
# To access a particular region of an image

# In[ ]:


img[10:50, 100:150] = [255,255,255]


# In[ ]:


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### Assigning a ROI to some other region of the image

# In[ ]:


roi = img[37:111,107:194]
img[0:74,0:87] = roi


# In[ ]:


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## DAY 5

# ### Image arithmetics

# **Superimposing 2 images** : superimposing 2 images on one another requires the 2 images to be of the same size. Some data from both the images might get lost.

# In[ ]:


import cv2
import numpy as np


# In[ ]:


img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')


# In[ ]:


add = img1 + img2

cv2.imshow('add',add)
cv2.waitKey(0)
cv2.destroyAllWindows()


# **cv2.add( \<img1\>, \<img2\> )** : adds the pixels of the 2 images leading to a brighter image.

# In[ ]:


add = cv2.add(img1,img2)

cv2.imshow('add',add)
cv2.waitKey(0)
cv2.destroyAllWindows()


# **addWeighted method** : the parameters are the first image, the weight, the second image, that weight, and then finally gamma, which is a measurement of light. 
# 
# **cv2.addWeighted( \<img1\>, \<weight\>, \<img2\>, \<weight2\>, \<gamma_value\>)**
# 
# The weights add up to 1 and both the images retain their values/data.

# In[ ]:


weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

cv2.imshow('weighted',weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## DAY 6

# ## Thresholding

# The idea of thresholding is to further-simplify visual data for analysis. First, you may convert to gray-scale, but then you have to consider that grayscale still has at least 255 values. What thresholding can do, at the most basic level, is convert everything to white or black, based on a threshold value. Let's say we want the threshold to be 125 (out of 255), then everything that was 125 and under would be converted to 0, or black, and everything above 125 would be converted to 255, or white. If you convert to grayscale as you normally will, you will get white and black. If you do not convert to grayscale, you will get thresholded pictures, but there will be color.
# 
# **retval, threshold = cv2.threshold( \<img_link\>, \<threshold_value\>, \<max_value\>, cv2.THRESH_BINARY)**
# 
# **cv2.THRESH_BINARY** converts the pixels in either of the colors, white or black.

# In[ ]:


import cv2
import numpy as np


# In[ ]:


img = cv2.imread('bookpage.jpg')


# In[ ]:


retval, threshold1 = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

# original image
cv2.imshow('original',img)

# image without grayscale
cv2.imshow('threshold1',threshold1) 

# image after grayscale
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval, threshold2 = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold2',threshold2)

cv2.waitKey(0)
cv2.destroyAllWindows()


# **Adaptive thresholding** is a form of image thresholding technique in which rather than specifying the threshold value manually or using any restrictions, the threshold value is adjusted and selected automatically according to the image pixels and layout for converting the image pixels to grayscale or a binary image.
# 
# * **ADAPTIVE_THRESH_MEAN_C** − threshold value is the mean of neighborhood area.
# 
# 
# * **ADAPTIVE_THRESH_GAUSSIAN_C** − threshold value is the weighted sum of neighborhood values where weights are a Gaussian window.
# 
# 
# * **thresholdType** − A variable of integer type representing the type of threshold to be used.
# 
# 
# * **blockSize** − A variable of the integer type representing size of the pixelneighborhood used to calculate the threshold value.
# 
# 
# * **C** − A variable of double type representing the constant used in the both methods (subtracted from the mean or weighted mean).
# 
# 
# **cv2.adaptiveThreshold( \<img_link/source\>, \<max_value\>, \<adaptive_method\>, \<threshold_type\>, \<block_size\>, \<c\>)**

# In[ ]:


th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('Adaptive threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## DAY 7

# ### Color Filtering

# HSL and HSV are the two most common cylindrical-coordinate representations of points in an RGB color model.
# 
# HSL stands for hue, saturation, and lightness, and is often also called HLS. HSV stands for hue, saturation, and value, and is also often called HSB (B for brightness). A third model, common in computer vision applications, is HSI, for hue, saturation, and intensity. However, while typically consistent, these definitions are not standardized, and any of these abbreviations might be used for any of these three or several other related cylindrical models.
# 
# In each cylinder, the angle around the central vertical axis corresponds to "hue", the distance from the axis corresponds to "saturation", and the distance along the axis corresponds to "lightness", "value" or "brightness".

# In[ ]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)


# In[ ]:


while(True):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# **mask** : The way this works is what we see will be anything that is between our ranges here, basically 30-255, 150-255, and 50-180. 

# ## DAY 8

# ### Blurring and Smoothing

# To eliminate noise from our filters, like simple thresholds or even a specific color filter 

# In[ ]:


import cv2
import numpy as np


# In[ ]:


cap = cv2.VideoCapture(0)


# When we are dealing with images at some points the images will be crisper and sharper which we need to smoothen or blur to get a clean image, or sometimes the image will be with a really bad edge which also we need to smooth it down to make the image usable. In OpenCV, we got more than one method to smooth or blur an image which are as follows -

# **Method 1: With 2D Convolution**  
# In this method of smoothing, we have complete flexibility over the filtering process because we will be using our custom-made **kernel** [a simple 2d matrix of NumPy array which helps us to process the image by convolving with the image pixel by pixel]. A kernel basically will give a specific weight for each pixel in an image and sum up the weighted neighbor pixels to form a pixel, with this method we will be able to compress the pixels in an image and thus we can reduce the clarity of an image, By this method, we can able to smoothen or blur an image easily. 
# 
# **.filter2D(sourceImage, ddepth, kernel)**

# In[ ]:


while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    kernel = np.ones((15,15),np.float32)/225
    smoothed = cv2.filter2D(res,-1,kernel)
    cv2.imshow('Original',frame)
    cv2.imshow('Averaging',smoothed)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# Method 2:  With pre-built functions
# OpenCV comes with many prebuilt blurring and smoothing functions let us see them in brief,
# 
# 1. Averaging:
# Syntax: cv2.blur(image, shapeOfTheKernel)
# 
# Image– The image you need to smoothen
# shapeOfTheKernel– The shape of the matrix-like 3 by 3 / 5 by 5
# The averaging method is very similar to the 2d convolution method as it is following the same rules to smoothen or blur an image and uses the same type of kernel which will basically set the center pixel’s value to the average of the kernel weighted surrounding pixels. And by this, we can greatly reduce the noise of the image by reducing the clarity of an image by replacing the group of pixels with similar values which is basically similar color. We can greatly reduce the noise of the image and smoothen the image. The kernel we are using for this method is the desired shape of a matrix with all the values as “1”  and the whole matrix is divided by the number of values in the respective shape of the matrix [which is basically averaging the kernel weighted values in the pixel range].
# 
# 
# 2. Gaussian Blur:
# Syntax: cv2. GaussianBlur(image, shapeOfTheKernel, sigmaX )
# 
# Image– the image you need to blur
# shapeOfTheKernel– The shape of the matrix-like 3 by 3 / 5 by 5
# sigmaX– The Gaussian kernel standard deviation which is the default set to 0
# In a gaussian blur, instead of using a box filter consisting of similar values inside the kernel which is a simple mean we are going to use a weighted mean. In this type of kernel, the values near the center pixel will have a higher weight. With this type of blurs, we will probably get a less blurred image but a natural blurred image which will look more natural because it handles the edge values very well.

# In[ ]:


while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    blur = cv2.GaussianBlur(res,(15,15),0)
    cv2.imshow('Original',frame)
    cv2.imshow('Gaussian Blurring',blur)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# 3. Median blur:
# Syntax: cv. medianBlur(image, kernel size)
# 
# Image– The image we need to apply the smoothening
# KernelSize– the size of the kernel as it always takes a square matrix the value must be a positive integer more than 2.
# Note: There are no specific kernel values for this method.
# 
# In this method of smoothing, we will simply take the median of all the pixels inside the kernel window and replace the center value with this value. The one positive of this method over the gaussian and box blur is in these two cases the replaced center value may contain a pixel value that is not even present in the image which will make the image’s color different and weird to look, but in case of a median blur though it takes the median of the values that are already present in the image it will look a lot more natural.

# In[ ]:


while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    median = cv2.medianBlur(res,15)
    cv2.imshow('Original',frame)
    cv2.imshow('Median Blur',median)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# 4. Bilateral blur:
# Syntax: cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
# 
# Image– The image we need to apply the smoothening
# Diameter– similar to the size of the kernel
# SigmaColor– The number of colors to be considered in the given range of pixels [the higher value represents the increase in the number of colors in the given area of pixels]—Should not keep very high
# SigmaSpace – the space between the biased pixel and the neighbor pixel higher value means the pixels far out from the pixel will manipulate in the pixel value
# The smoothening methods we saw earlier are fast but we might end up losing the edges of the image which is not so good. But by using this method, this function concerns more about the edges and smoothens the image by preserving the images. This is achieved by performing two gaussian distributions. This might be very slow while comparing to the other methods we discussed so far. 

# In[ ]:


while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    bilateral = cv2.bilateralFilter(res,15,75,75)
    cv2.imshow('Original',frame)
    cv2.imshow('bilateral Blur',bilateral)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# ## DAY 9

# **Morphological Transformations** : These are some simple operations that we can perform based on the image's shape.
# 
# These tend to come in pairs. The first pair we're going to talk about is **Erosion and Dilation**. 
# 
# **Erosion** is where we will "erode" the edges. 
# 
# Working of erosion: 
# 
# A kernel(a matrix of odd size(3,5,7) is convolved with the image.
# A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel are 1, otherwise, it is eroded (made to zero).
# Thus all the pixels near the boundary will be discarded depending upon the size of the kernel.
# So the thickness or size of the foreground object decreases or simply the white region decreases in the image.
# 
# 
# The other version of this is **Dilation**, which basically does the opposite:
# 
# A kernel(a matrix of odd size(3,5,7) is convolved with the image
# A pixel element in the original image is ‘1’ if at least one pixel under the kernel is ‘1’.
# It increases the white region in the image or the size of the foreground object increases 

# In[ ]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)


# In[ ]:


while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Erosion',erosion)
    cv2.imshow('Dilation',dilation)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# The next pair is **"opening" and "closing."** The goal with opening is to remove "false positives" so to speak i.e. the noise present in the background. 
# 
# The idea of "closing" is to remove false negatives i.e. the noise present in the object in analysis. Basically this is where you have your detected shape, like our hat, and yet you still have some black pixels within the object. Closing will attempt to clear that up.
# 
# the false positive (FP) are the pixels considered by the segmentation in the object, but which in reality are not part of it,
# 
# finally, the false negative (FN) are the pixels of the object that the segmentation has classified outside.
# 
# ![image.png](attachment:image.png)

# In[ ]:


ap = cv2.VideoCapture(1)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Opening',opening)
    cv2.imshow('Closing',closing)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# ## DAY 10

# ## Edge Detection and Gradients
# 
# Image gradients can be used to measure directional intensity, and edge detection does exactly what it sounds like: it finds edges
# 
# **Principle behind Edge Detection** 
# 
# Edge detection involves mathematical methods to find points in an image where the brightness of pixel intensities changes distinctly.
# 
# 
# The first thing we are going to do is find the gradient of the grayscale image, allowing us to find edge-like regions in the x and y direction. The gradient is a multi-variable generalization of the derivative. While a derivative can be defined on functions of a single variable, for functions of several variables, the gradient takes its place.
# 
# The gradient is a vector-valued function, as opposed to a derivative, which is scalar-valued. Like the derivative, the gradient represents the slope of the tangent of the graph of the function. More precisely, the gradient points in the direction of the greatest rate of increase of the function, and its magnitude is the slope of the graph in that direction.
# 
# 
# 
# **Calculation of the derivative of an image**
# 
# A digital image is represented by a matrix that stores the RGB/BGR/HSV(whichever color space the image belongs to) value of each pixel in rows and columns. 
# 
# The derivative of a matrix is calculated by an operator called the Laplacian. In order to calculate a Laplacian, you will need to calculate first two derivatives, called derivatives of Sobel, each of which takes into account the gradient variations in a certain direction: one horizontal, the other vertical. 

# In[ ]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)


# Method for sharpening images is to use the **cv2.Laplacian() function**, which calculates the Laplacian of an image and returns the result as a sharpened image. It is also used for edge detection. 
# 
# cv2.CV_64F is, that's the data type. ksize is the kernel size.
# 
# **Sobel edge detection method**
# 
# **Horizontal Sobel derivative (Sobelx)**: It is obtained through the convolution of the image with a matrix called kernel which has always odd size. The kernel with size 3 is the simplest case.
# 
# **Vertical Sobel derivative (Sobely)**: It is obtained through the convolution of the image with a matrix called kernel which has always odd size. The kernel with size 3 is the simplest case.
# 
# **Convolution** is calculated by the following method: Image represents the original image matrix and filter is the kernel matrix.

# In[ ]:


while True:

    # Take each frame
    _, frame = cap.read()

    laplacian = cv2.Laplacian(frame,cv2.CV_64F)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

    cv2.imshow('Original',frame)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# **Canny() Function** in OpenCV is used to detect the edges in an image.
# 
# **Syntax: cv2.Canny(image, T_lower, T_upper, aperture_size, L2Gradient)**
# 
# T_lower: Lower threshold value in Hysteresis Thresholding
# 
# T_upper: Upper threshold value in Hysteresis Thresholding
# 
# aperture_size: Aperture size of the Sobel filter.
# 
# L2Gradient: Boolean parameter used for more precision in calculating Edge Gradient.

# In[ ]:


while True:

    _, frame = cap.read()

    cv2.imshow('Original',frame)
    edges = cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edges)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# **Edge Detection Applications**
# 
# * Reduce unnecessary information in an image while preserving the structure  of image.
# 
# * Extract important features of image like curves, corners and lines.
# 
# * Recognizes objects, boundaries and segmentation.
# 
# * Plays a major role in computer vision and recognition

# ## DAY 11

# ### Template Matching

# A fairly basic version of object recognition. The idea here is to find identical regions of an image that match a template we provide, giving a certain threshold.

# In[ ]:


import cv2
import numpy as np

img_rgb = cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('opencv-template-for-matching.jpg',0)
# width and height of the image
w, h = template.shape[::-1]


# In the function **cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)** the first parameter is the mainimage, the second parameter is the template to be matched and the third parameter is the method used for matching.

# In[ ]:


res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)


# In[ ]:


for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)         
cv2.destroyAllWindows()


# Limitations of Template Matching: 
# 
# * Pattern occurrences have to preserve the orientation of the reference pattern image(template)
# 
# * As a result, it does not work for rotated or scaled versions of the template as a change in shape/size/shear, etc. of object w.r.t. the template will give a false match.
# 
# * The method is inefficient when calculating the pattern correlation image for medium to large images as the process is time-consuming.

# ## DAY 12

# ###  Foreground extraction
# 
# The idea here is to find the foreground, and remove the background. This is much like what a green screen does, only here we wont actually need the green screen.
# 
# 
# 
# The **numpy.ones()** function returns a new array of given shape and type, with ones.
# 
# **Syntax: numpy.ones(shape, dtype = None, order = 'C')**
# 
# Parameters : 
# 
# * shape : integer or sequence of integers
# * order  : C_contiguous or F_contiguous
#          C-contiguous order in memory(last index varies the fastest)
#          C order means that operating row-rise on the array will be slightly quicker
#          FORTRAN-contiguous order in memory (first index varies the fastest).
#          F order means that column-wise operations will be faster. 
# * dtype : [optional, float(byDefault)] Data type of returned array. 

# The **numpy.zeros()** function returns a new array of given shape and type, with zeros. 

# In[ ]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# **rect = (start_x, start_y, width, height)**
# 
# This is the rectangle that encases our main object.

# In[ ]:


img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# rect used here only encases the face for this particular image
rect = (161,79,150,150)  


# **cv2.grabCut** takes quite a few parameters. First the input image, then the mask, then the rectangle for our main object, the background model, foreground model, the amount of iterations to run, and what mode you are using.

# In[ ]:


cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.show()


# ## DAY 13

# ### Corner Detection
# 
# The purpose of detecting corners is to track things like motion, do 3D modeling, and recognize objects, shapes, and characters.
# 
# We detect corners with the **goodFeaturesToTrack** function. The parameters here are the image, max corners to detect, quality, and minimum distance between corners.

# In[ ]:


import numpy as np
import cv2

img = cv2.imread('opencv-corner-detection-sample.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.intp(corners)


# In[ ]:


for corner in corners:
    # returns contiguous flattened array(1D array with all the input-array elements and with the same type as it
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)
    
cv2.imshow('Corner',img)
cv2.waitKey(0)         
cv2.destroyAllWindows()


# ## DAY 14

# ## Feature Matching
# 
# Feature matching is going to be a slightly more impressive version of template matching, where a perfect, or very close to perfect, match is required.
# 
# We start with the image that we're hoping to find, and then we can search for this image within another image. The beauty here is that the image does not need to be the same lighting, angle, rotation...etc. The features just need to match up.

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('opencv-feature-matching-template.jpg',0)
img2 = cv2.imread('opencv-feature-matching-image.jpg',0)


# In[ ]:





# In[ ]:


# This is the detector we're going to use for the features.
orb = cv2.ORB_create()
# the key points and their descriptors with the orb detector.
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# **BFMatcher()** function is used in feature matching and used to match features in one image with other image. BFMatcher refers to a Brute-force matcher that is nothing, but a distance computation used to match the descriptor of one feature from the first set with each of the other features in the second set. The nearest is then returned. For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches of descriptor sets. So in order to implement the function, our aim is to find the closest descriptor from the set of features of one image to the set of features of another image.

# In[ ]:


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# we create matches of the descriptors, then we sort them based on their distances.
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)


# In[ ]:


# we've drawn the first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()


# ## DAY 15

# ### Background Reduction
# 
# To reduce the background of images, by detecting motion.

# The idea here is to extract the moving forground from the static background. You can also use this to compare two similar images, and immediately extract the differences between them.
# 
# **Background Subtraction** is one of the major Image Processing tasks. It is used in various Image Processing applications like Image Segmentation, Object Detection, etc. OpenCV provides us 3 types of Background Subtraction algorithms:-
# 
# * BackgroundSubtractorMOG
# 
# * BackgroundSubtractorMOG2 : In the previous subtractor worked fairly well but in real-world situations, there is also a presence of shadows. In BackgroundSubtractorMOG2, we can also detect shadows and in the output of the following code, it’s clearly seen.
# 
# * BackgroundSubtractorGMG

# In[ ]:


import numpy as np
import cv2

cap = cv2.VideoCapture('cat.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()


# ## DAY16

# ### Haar Cascade Object Detection for Face & Eye
# 
# In order to do object recognition/detection with cascade files, you first need cascade files. For the extremely popular tasks, these already exist. Detecting things like faces, cars, smiles, eyes, and license plates for example are all pretty prevalent.

# In[ ]:


import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)


# CascadeClassifier::**detectMultiScale()**
# 
# Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

# In[ ]:


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face_cascade.detectMultiScale() finds faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # (x,y) is starting point and w & h are the distances along x and y direction
    for (x,y,w,h) in faces:
        # (255,0,0) i.e. a blue coloured rectangle for faces
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray is initialised with the region of the face since normally would not want to detect eyes outside the face.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # (0,255,0) i.e. a green coloured rectangle for eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    cv2.imshow('img',img)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

