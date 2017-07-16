#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016

'''
Part 1 of the problem is to use the simple bayes net. The emission probabilities for each column are used to partially
get the ridgeline.
Part 2 requires us to use the MCMC technique that needs the transition probabilities.
Part 3 uses the MCMC technique along with an input. The method is to use the same strategy from the first column to the
given input and then from that to the last column. This is done for certain number of iterations.
The solution most importantly uses certain methods.
-The get_emission_prob gets the row that reflects the point closest to the ridge. The edge_strength is used to identify
this row by using the fact that each row has a certain probability of being close to the ridge.
-The get_transition_prob gets the probability of how close a particular row is to the previous row. This is similar to
 the comparison of two numbers in a given set of numbers. Here, The height of the image, that is, the number of rows
  and the absolute difference between two rows are used to know the transition probabilities between the rows.
The third part is indicated by green colored line which mostly lies on the blue colored line. So mcmc.jpg is an image
file that will be created to know a clear vision of the blue colored line.
'''

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import math

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return filtered_y**2

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( max(y-thickness/2, 0), min(y+thickness/2, image.size[1]-1 ) ):
            image.putpixel((x, t), color)
    return image

def get_emission_prob(edgestr):

    edgestrprob = []
    for i in range(edgestr.shape[1]):
        edgestrprob.append((edgestr[:, i] + 1) / (sum(edgestr[:, i]) + edgestr.shape[0]))
    edgestrprob = array(edgestrprob).T
    max_row = edgestrprob.argmax(axis=0)

    return  max_row, edgestrprob

def get_transition_prob(imgheight):
    transtionprob = []
    for h in range(1, imgheight + 1):
        temp = []
        for w in range(1, imgheight + 1):
            temp.append(float(imgheight - (abs(w - h))))
        transtionprob.append(temp)
    transtionprob = array(transtionprob)
    transtionprob = transtionprob/transtionprob.sum(axis=0)[:,newaxis]
    return transtionprob

def mcmc(max_row, transtionprob, edgestrprob, gt_row, gt_col):
    height,width = edgestrprob.shape

    max_row = list(max_row)
    if gt_row != None:
        rowsequence = max_row[:gt_col]+[gt_row]+max_row[gt_col+1:]
    else:
        rowsequence = max_row
    for i in range(10):
        for w in range(1, width-1):
            if w != gt_col:
                s=[]
                for h in range(0, height):
                    s.append(transtionprob[rowsequence[w-1]][h]*edgestrprob[h][w]*transtionprob[h][rowsequence[w+1]]*((height-h)**3))
                rowsequence[w] = s.index(max(s))

    return rowsequence

# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)

max_row, edgestrprob = get_emission_prob(edge_strength)

transtionprob = get_transition_prob(edge_strength.shape[0])
ridge_bayes = max_row
ridge_mcmc =mcmc(max_row,transtionprob,edgestrprob,None,-100)
ridge_human = mcmc(max_row,transtionprob,edgestrprob,int(gt_row),int(gt_col))
imsave('edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
t = 'bayes.jpg'
imsave(t, draw_edge(input_image, ridge_bayes, (255, 0, 0), 5))

# output answer
t1 = Image.open(t)

t2 = 'mcmc.jpg'

imsave(t2, draw_edge(t1, ridge_mcmc, (0, 0, 255), 5))

t1 = Image.open(t2)
imsave(output_filename, draw_edge(t1, ridge_human, (0, 255, 0), 5))
