#!/usr/local//bin/python3
# This is just a sample program to show you how to do
# basic image operations using python and the Pillow library.
#
# By Eriya Terada, based on earlier code by Stefan Lee,
#    lightly modified by David Crandall, 2020

#Import the Image and ImageFilter classes from PIL (Pillow)
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import random

if __name__ == '__main__':
    #Load an image (this one happens to be grayscale)
    im = Image.open("example.jpg")

    #Check its width, height, and number of color channels
    print("Image is %s pixels wide." % im.width)
    print("Image is %s pixels high." % im.height)
    print("Image mode is %s." % im.mode)

    #pixels are accessed via a (X,Y) tuple
    print("Pixel value is %s" % im.getpixel((10,10)))

    #pixels can be modified by specifying the coordinate and RGB value
    im.putpixel((10,10), 20)
    print("New pixel value is %s" % im.getpixel((10,10)))

    #Create a new blank color image the same size as the input
    color_im = Image.new("RGB", (im.width, im.height), color=0)

    # Loops over the new color image and fills in any area that was white in the 
    # first grayscale image  with random colors!
    for x in range(im.width):
        for y in range(im.height):

            if im.getpixel((x,y)) > 200:
                R = random.randint(0,255)
                G = random.randint(0,255)
                B = random.randint(0,255)
                color_im.putpixel((x,y), (R,G,B))
            else:
                color_im.putpixel((x,y), (0,0,0))


    #Save the image
    color_im.save("output.png")

    # Using Pillow's code to create a convolution kernel and apply it to our color image
    # Here, we are applying the box blur, where a kernel of size 3x3 is filled with 1
    # and the result is divided by 9
    # Note: The assignment requires you to implement your own convolution, but
    #   there's nothing stopping you from using Pillow's built-in convolution to check
    #   that your results are correct!
    result = color_im.filter(ImageFilter.Kernel((3,3),[1,1,1,1,1,1,1,1,1],9))

    # Draw a box and add some text. Just for fun!
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("/usr/share/fonts/msttcorefonts/arial.ttf", 16)
    draw.text((0, 0),"Hello!",(0,255,0), font=font)
    draw.rectangle(((100,100), (200,200)), (0,255,0))
    
    result.save("convolved.png")
