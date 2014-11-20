# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:22:13 2014

@author: Ankur
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


class ImageManipulation():
    """Just a warmup task for image manipulation
       filename : name of the file which is needed to be modified
       rmin = min value
       rmax = max value  
    """
    
    def __init__(self,filename=None,rmin=0,rmax=0):
        if filename is None:
            raise Exception("filename not given as first param")
        if rmin == rmax or rmin < 0 or rmax < 0:
            raise Exception("please change the rmin and rmax values")
            
        #received the image in a nd-numpy-array
        self.image = scipy.misc.imread(filename)
        
        #initialized the image with zeros
        self.new_image = np.zeros(self.image.shape)
        self.rmin = rmin
        self.rmax = rmax
    
    def manipulate(self):
        it = np.nditer(self.image, flags=['multi_index'])
        point_b = np.array([self.image.shape[0],self.image.shape[1]])/2
        
        while not it.finished:
            euc_dist = self.compute_euclidean(it.multi_index,point_b)
            x,y = it.multi_index            
            if euc_dist <= self.rmax and euc_dist >=self.rmin:
                self.new_image[x,y] = 0
            else:
                self.new_image[x,y] = it[0]
            it.iternext()
        
        plt.subplot(222),plt.imshow(self.image, cmap = 'gray')
        plt.title('g(x,y)'),  plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(self.new_image, cmap = 'gray')   
        plt.title("g'(x,y)"),plt.xticks([]), plt.yticks([])
        plt.show()
        
        
        
        self.save_image(self.new_image)
        
     
    def save_image(self,imarray):
        scipy.misc.imsave("task1_1.jpg",imarray)
        print "output has been generated and saved in the current directory"
        
    def compute_euclidean(self,point_a, point_b):
        return np.linalg.norm(point_a-point_b)
        
 

if __name__ == "__main__":     
    obj = ImageManipulation("bauckhage.jpg",20,40)
    obj.manipulate()

