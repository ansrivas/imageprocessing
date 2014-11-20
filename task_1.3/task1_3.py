# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 00:55:38 2014

@author: Ankur
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt


class PhaseImportance():
    
    def __init__(self,filename1=None,filename2=None):
        if filename1 is None or filename2 is None:
            raise Exception("filename not given as first param")
            
        self.image1 = scipy.misc.imread(filename1)
        self.image2 = scipy.misc.imread(filename2)
        
        self.G = None
        self.H = None
        self.K = None
        self.centerG = None
        self.centerH = None
        
    def run(self):
        self.calculate_fft()
        
        
    def calculate_fft(self):
        #take transform and bring it to origin
        self.G = np.fft.fft2(self.image1)
        self.centerG = np.fft.fftshift(self.G)
        
        #take transform and bring it to origin
        self.H = np.fft.fft2(self.image2)
        self.centerH = np.fft.fftshift(self.H)
        
        self.construct_wave()
    
    def construct_wave(self):
                
        phase = self.find_phase()           
        magnitude = self.magnitude()

        #re = magnitude/np.sqrt(1+np.square(np.tan(phase)))
        re = magnitude*np.cos(phase)
        im = magnitude*np.sin(phase)
        
        final_wave = re + 1j*im

        finimage = np.abs(np.fft.ifft2(final_wave))
        self.save_image("task1_3.jpg",finimage)
        
        #plotting functions
        plt.subplot(221),plt.imshow(finimage, cmap = 'gray')
        plt.xticks([]), plt.yticks([])
        plt.title("Final-Image")
        plt.show()
     
    def magnitude(self):
        return np.abs(self.centerG)
        
    def find_phase(self):
        #this is in radians
        return np.angle(self.centerH)
    
    def save_image(self,name,imarray):
        scipy.misc.imsave(name,imarray)

if __name__ == "__main__":
    obj = PhaseImportance("bauckhage.jpg","clock.jpg")
    obj.run()