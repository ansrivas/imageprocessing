# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:13:34 2014

@author: Ankur
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


class BandPassFilter():
    """
    Task 2.2: calculating the fft and fftshift for an image
    filename: name of the image
    rmin, rmax: min and max frequencies, outside which everything is 
    to be suppressed
    """
    
    def __init__(self,filename=None,rmin=0,rmax=0):
        if filename is None:
            raise Exception("filename not given as first param")
        if rmin == rmax or rmin < 0 or rmax < 0:
            raise Exception("please change the rmin and rmax values")
            
        #received the image in a nd-numpy-array
        self.image = sc.misc.imread(filename)
        
        #initialized the image with zeros
        self.new_image = np.zeros(self.image.shape)
        self.rmin = rmin
        self.rmax = rmax
        
        #fft of the image and inverse fft of the image
        self.fft = None
        self.shiftedfft = None
        self.image_freq = None
        self.ifft = None

        self.compute_fourier_transform()
        self.compute_inverse_fft()
        self.inverse_fft()
   
    def compute_fourier_transform(self):
        
        self.fft = np.fft.fft2(self.image)
        self.shiftedfft = np.fft.fftshift(self.fft)
        #take the absolute and then the log of the shifted frequency
        self.image_freq = np.log(np.abs(self.shiftedfft))
        
         

        #plotting functions
        plt.subplot(221),plt.imshow(self.image, cmap = 'gray')
        plt.title('g(x,y)')
        plt.xticks([]), plt.yticks([])
        plt.subplot(222),plt.imshow(self.image_freq, cmap = 'gray')
        plt.xticks([]), plt.yticks([])
        plt.title("log|G(u, v)|")
        plt.show()
         
        self.save_image("task1_2_1.jpg",self.image_freq)
           
    def compute_inverse_fft(self):
        it = np.nditer(self.shiftedfft, flags=['multi_index'])
        
        self.new_image = np.copy(self.shiftedfft)
        
        point_b = np.array([self.image.shape[0],self.image.shape[1]])/2
        
        while not it.finished: 
            euc_dist = self.compute_euclidean(it.multi_index,point_b)
            x,y = it.multi_index   
    
            if not (euc_dist >= self.rmin and euc_dist <= self.rmax):
                self.new_image[x,y] = 1
                
            it.iternext()

        self.image_freq = np.log(np.abs(self.new_image))
        
        #plotting functions
        plt.subplot(223),plt.imshow(self.image_freq, cmap = 'gray')
        plt.xticks([]), plt.yticks([])
        plt.title("log|G'(u, v)|")
        plt.show()
        
                 
        self.save_image("task1_2_2.jpg",self.image_freq)
        
     
    def inverse_fft(self):
        
        # shift back (we shifted the center before)
        # inverse fft to get the image back 
    
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(self.new_image)))
        
        
        #plotting functions
        plt.subplot(224),plt.imshow(img_back, cmap = 'gray')
        plt.xticks([]), plt.yticks([])
        plt.title("g'(x,y)")
        plt.show()
        self.save_image("task1_2_3.jpg",img_back)        
     
     
    def save_image(self,name,imarray):
        sc.misc.imsave(name,imarray)
        #print "output has been generated and saved in the current directory"
        
    def compute_euclidean(self,point_a, point_b):
        return np.linalg.norm(point_a-point_b)
        
 

if __name__ == "__main__":     
    obj = BandPassFilter("bauckhage.jpg",20,40)
    

