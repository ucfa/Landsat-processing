#!/usr/bin/env python

# This code requires:
# 1. raster_mask, available: 
# https://github.com/jgomezdans/geogg122-1/blob/master/Chapter6_Practical/python/raster_mask.py
# 2. smoothn, available: 
# http://www2.geog.ucl.ac.uk/~plewis/geogg122/Chapter5_Interpolation/python/smoothn.py
# and ensure they are in the same directory as this code.

# Copyright (for whatever that's worth) Benjamin Allen, 2017.
# MIT License.

import numpy as np
import gdal
from raster_mask import *
from smoothn import *


def auto_rmask(input_in, shape_in):
    '''
    This function takes:
    1. An input Landsat 8 image.
    2. An input shapefile
    which will:
    1. Mask the input file to the boundaries of the input shapefile.
    Notes:
    1. Shapefile and input MUST be same co-ordinate system.
    This will typically be in WGS84.
    
    Modified from Prof. Lewis's masking code, UCL.
    '''
      
     
    array = []
    mask = raster_mask2(input_in,target_vector_file=shape_in,\
                        attribute_filter=0)
    
    # Building the boundary.
    rowpix,colpix = np.where(mask == False)
    mincol,maxcol = min(colpix),max(colpix) 
    minrow,maxrow = min(rowpix),max(rowpix) 
    ncol = maxcol - mincol + 1 
    nrow = maxrow - minrow + 1 
    
    # Reading in the Max and min cols
    xoff = int(mincol) 
    yoff = int(minrow) 
    xsize = ncol and int(ncol)
    ysize = nrow and int(nrow)
    
    # Apply the boundary mask.
    small_mask = mask[minrow:minrow+nrow,mincol:mincol+ncol]
    
    # Re-read in the file, this time applying our mask.    
    read_in = gdal.Open(input_in).ReadAsArray(xoff,yoff,xsize,ysize)
    
    # converting our small shapefile mask with our opened and masked HDF files.
    full_mask = ma.array(read_in, mask=small_mask)
                                     
    # Providing an error message.
    ##### Currently non-functional. ####
    if input_in is None:
        print "Problem opening file %s!" % (array)
    else:
        array.append(full_mask)
    
    # Return the appended array, but convert back to 2D
    array2 = array[0]
    return array2

def iterate_landsat():
    new_array = []
    for n in xrange (1,12):
        try:
            a = './LC08_L1TP_203023_20160504_20170325_01_T1_B%d.TIF' %(n)
        
            # auto_rmask already opens using gdal.
        
            new_array.append(auto_rmask(a,shape_in))
        except:
            pass
    bands = {'coast':new_array[0],\
         'blue': new_array[1],\
         'green': new_array[2],\
         'red': new_array[3],\
         'NIR': new_array[4],\
         'SWIR1': new_array[5],\
         'SWIR2': new_array[6],\
         'pan': new_array[7],\
         'cloud': new_array[8],\
         'IR1': new_array[9],\
         'IR2': new_array[10]
        }    
    return bands

def create_comp(in_band3, in_band2, in_band1):
    '''
    Creation of TC image from RGB bands (3).
    Inputs:
    1. Red band
    2. Green band
    3. Blue band
    which outputs:
    1. RGB true colour image.
    
    Adapted from Stack Overflow.
    '''
    from scipy.misc import bytescale
    from skimage import exposure
    
    # Take the array dimensions from band 3.
    img_dim = in_band3.shape
    
    # Define an empty array using np.zeroes.
    img = np.ma.zeros((img_dim[0], img_dim[1], 3), dtype=np.float)
    
    
    img[:,:,0] = in_band3
    img[:,:,1] = in_band2
    img[:,:,2] = in_band1
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return bytescale(img_rescale)

# Cloud masking functionality.

def auto_cmask(in_band, cloud_band):
    '''
    Automatic computation of an LS8 cloud band.
    Takes:
    1. Input band.
    2. The cloud band (9).
    which returns:
    1. A masked input band.
    '''
    min1 = np.min(cloud_band)
    avg = np.mean(cloud_band)
    dif = avg + (avg - min1) 
    cloud_mask = (cloud_band >=dif)
    cloud_mask2 = ma.array(in_band, mask = cloud_mask)    
    return cloud_mask2
