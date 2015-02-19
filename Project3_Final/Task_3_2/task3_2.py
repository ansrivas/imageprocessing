# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:49:52 2015

@author: Ankur
"""

 
'''
Task 3.2: Implementing warps
'''
 
import numpy as np
import scipy.misc , scipy.ndimage
import time 


class CWarping:
    def __init__(self, inputImageFileName,freq = 2, amplitude = 80, phase = 20,bothAxesWarping=True,param2ndAxis={}):
        #Get the Images as array
        self.freq=freq
        self.amplitude = amplitude
        self.phase = phase
        self.bothaxis = bothAxesWarping
        self.params2image = param2ndAxis
        self.inputImage = scipy.misc.imread(inputImageFileName)
        #if its a colored image, convert to rgb
        if len(self.inputImage.shape) > 2:
            from skimage.color import rgb2gray
            self.inputImage = rgb2gray(self.inputImage)
     
        self.outputImage = None
        self.rows,self.columns = self.inputImage.shape
 
    def saveimage(self,filename,array): 
        scipy.misc.imsave(filename+".jpg" , array)
        
    def construct_warps(self):
        outputrows = 0        
        if self.freq >=2:
            outputrows = self.rows + self.amplitude + self.amplitude
        else: 
            outputrows = self.rows + self.amplitude
        self.outputImage = np.zeros((outputrows,self.columns))
        
        for col in range(0, self.columns):
            y= self.amplitude*np.sin(((np.pi/self.columns)*self.freq*col)+np.pi*self.phase)
            startrow = (int)(self.amplitude-y)
            endrow = (int)(self.rows-y+self.amplitude)
            print startrow, endrow
            if(endrow-startrow > self.rows):
                print startrow,endrow                
                endrow = startrow +self.rows
                
            self.outputImage[startrow:endrow,col] = self.inputImage[ : , col]

        
        if(self.bothaxis):
            temp = np.transpose(self.outputImage)
            row,col = temp.shape            
            if self.params2image["freq"] >=2:
                outputrows = row+ 2*self.params2image["amplitude"]
            else: 
                outputrows = row + self.params2image["amplitude"]
            self.outputImage = np.zeros((outputrows,col))
            for x in range(0, col):
                y= self.params2image["amplitude"]*np.sin(((np.pi/col)*self.params2image["freq"]*x)+np.pi*self.params2image["phase"])
                startrow = (int)(self.params2image["amplitude"]-y)
                endrow = (int)(row-y+self.params2image["amplitude"])
                if(endrow-startrow > row):
                   endrow = startrow+row
                self.outputImage[startrow:endrow,x] = temp[ : , x]
                #if(x%3==0):
                 #   self.saveimage(str(x),self.outputImage)
            self.outputImage = np.transpose(self.outputImage)
        return self.outputImage 
        

if __name__ == '__main__':
    
 
    total = 0
    maxiter = 1
    for i in xrange(0,maxiter):
        t =time.time()
        image = CWarping("bauckhage.jpg",freq = 1, amplitude = 50, phase = 0,bothAxesWarping=False,param2ndAxis={"freq":3,"amplitude":10,"phase":0.5})
        image.saveimage("out",image.construct_warps()) 
        total = total + time.time()-t 
    print "Run Complete ",total/maxiter
     
  
