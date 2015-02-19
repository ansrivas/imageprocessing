# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 19:05:42 2015

@author: VigneshRao
"""

import numpy as np
import scipy.misc
import time



class Picture:
    
    def __init__(self, inputImageFileName, rad=None,inrad=None):
        
        self.inputImage = scipy.misc.imread(inputImageFileName)
        
        #if its a colored image, convert to rgb
        if len(self.inputImage.shape) > 2:
            from skimage.color import rgb2gray
            self.inputImage = rgb2gray(self.inputImage)
            
            #Since this greyed image has values from 0 to 1.0 we scale it to 0 to 255
            self.inputImage = self.inputImage * 255
            
        self.rows,self.columns = self.inputImage.shape
        self.radius = rad
        self.inner_radius = inrad
        if self.radius <= self.inner_radius:
            raise Exception ("Cannot have the inner radius greater than or equal to outer radius. Hence no warping can be in this case")
        
        #Because the output image is a matrix which is circle and we need a 
        #matrix of dimesions as it diameter
        self.outputImage = np.zeros((self.radius*2,self.radius*2),int)
        
                
    def radial_warp(self):
        self.rmax,self.cmax = self.outputImage.shape
        for x in xrange(self.rmax):
            for y in xrange(self.cmax):
                
                #Converting the co-ordinates in polar form
                r = np.sqrt(( x - self.radius ) ** 2 + ( y - self.radius ) ** 2 )
                
                #If the euclidean distance is smaller than radius and larger than inner radius, we warp.
                if r < self.radius and r > self.inner_radius:
                    
                    #The phase of the point is ...
                    phi = np.arctan2 ( y - self.radius, x - self.radius )
                    
                    # based on the angle we choose the column
                    v = ((np.degrees(phi)+179.0)/360.0) * self.columns
                    
                    # Based on the distance from center we chose the row  
                    u = self.rows - ((float((r - self.inner_radius)/(self.radius - self.inner_radius)))*float(self.rows)) - 1  
                    
                    
                    #Boundary Check
                    if u < self.rows and v < self.columns:
                        self.outputImage[x][y] = self.inputImage[int(u)][int(v)]#Pull warping by nearest neighbour
                        
                        
                    
    def saveFile(self):
        
        #Taking transpose of the output matrix for a better view/according to the project question
        self.outputImage = np.transpose(self.outputImage)
        
        if self.inner_radius > 0:
            filename = "torus.jpg"
        else:
            filename = "radial.jpg"
            
        scipy.misc.imsave(filename,self.outputImage)
        
        return None
   


if __name__ == '__main__':
    t1=time.time()
    
    #Here rad determines the (outer)radius of the image we want to warp and inrad is the inner radius of the torus.
    #If inrad/ inner radius is 0 then it becomes a radial anamorphic disc
    Image_obj = Picture("test.jpg",rad = 250 ,inrad = 30)
    Image_obj.radial_warp()
    Image_obj.saveFile()
    
    print "Implementation Time : ", np.round(time.time() - t1,3),"Secs"
    