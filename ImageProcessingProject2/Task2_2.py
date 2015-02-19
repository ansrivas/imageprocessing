# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 23:15:27 2014

@author: Ankur
"""

# -*- coding: utf-8 -*-
'''
Task 2.2: Implementation of derivative of a gaussian and convolution with real image
'''
import math
import numpy as np
import scipy.misc , scipy.ndimage
import time 

class GaussianFilter:
    def __init__(self, inputImageFileName, kernelSize):
        #Get the Images as array
        self.inputImage = scipy.misc.imread(inputImageFileName)
        if(self.inputImage.shape > 2):
            from skimage.color import rgb2gray
            self.inputImage = rgb2gray(self.inputImage)
            
        self.outputImage = np.zeros(self.inputImage.shape)
        self.noisyimage = None
       
        #Initialize Gaussian filter parameters
        self.KERNEL_SIZE = kernelSize
        
        #SIGMA is calculated from kernel size
        self.SIGMA = (self.KERNEL_SIZE - 1.0) / (2.0 * 2.575)   
        self.gaussianKernel = np.zeros([self.KERNEL_SIZE, self.KERNEL_SIZE])
        
        #Construct the kernel of specified size
        self._constructGaussianKernel()
    
    
    def _constructGaussianKernel(self):
        iterator = np.nditer(self.gaussianKernel, flags=['multi_index'])
        
        sigmaSquare = self.SIGMA ** 2
        constantTerm = (1/ (2 * math.pi * sigmaSquare))
        
                
        while not iterator.finished:
            kx, ky = iterator.multi_index
        
            x = kx - int((self.KERNEL_SIZE / 2))
            y = ky - int((self.KERNEL_SIZE / 2))
            
            self.gaussianKernel[kx,ky] = constantTerm * (math.exp( (-x**2 - y**2) / ( 2 * sigmaSquare)))
            
            iterator.iternext()
        
        #Normalize
        self.gaussianKernel = self.gaussianKernel / self.gaussianKernel.sum()
    
    def _padConstant(self):
        h, w = self.inputImage.shape
        m = self.KERNEL_SIZE
        noOfPadsLeft = noOfPadsRight = noOfPadsUp = noOfPadsDown = 0
        
        if (w - self.KERNEL_SIZE %2 == 0):
            noOfPadsLeft = int((w - m)/2)
            noOfPadsRight = int((w - m)/2)
        else:
            noOfPadsLeft = int((w - m)/2)
            noOfPadsRight = int((w - m)/2) + 1
        
        if (h - self.KERNEL_SIZE %2 == 0):
            noOfPadsUp = int((h - m)/2)
            noOfPadsDown = int((h - m)/2)
        else:
            noOfPadsUp = int((h - m)/2)
            noOfPadsDown = int((h - m))/2 + 1

        self.gaussianKernel = np.pad(self.gaussianKernel, ((noOfPadsUp, noOfPadsDown),(noOfPadsLeft, noOfPadsRight)), 'constant')         
     
     
    def calculateSimpleGradientImage(self):
        
        self.noisyimage =   self.inputImage + 0.8 * self.inputImage.std() * np.random.random(self.inputImage.shape)
        print 0.8 * self.inputImage.std() * np.random.random(self.inputImage.shape)
        self.saveimage("noisyimage",self.noisyimage)
        #self.noisyimage = self.inputImage
        x,y = np.gradient(self.noisyimage) 
        
        #uncomment this to save simple x-y gradient images
        #self.saveimage("simplex",x)
        #self.saveimage("simpley",y)
        
        self.saveimage("without_gaussian",np.sqrt(x**2+ y**2))

        
    def applyGaussianFilter(self):
        startTime = time.clock()
        self.calculateSimpleGradientImage()
        #Here, we convolute image and gaussian kernel in frequency domain
        self.inputImage = self.noisyimage
        #Compute fourier transform of input image
        self.inputImageFFT = np.fft.fft2(self.inputImage)
        
        #Padding
        self._padConstant()
 
        #convolve an array with this kernel, will produce its gradient
        #dx=np.array([[0.0,0,0.0],[1.0,0, -1.0],[0.0,0,0.0]])
        dx = np.array([[-1,0,1]])
        dy=np.transpose(dx)  
        #For calculating the gradient in y-direction the transpose of dx is applied::
        
        #fox,foy : are the x,y derivatives of gaussian kernel
        fox=scipy.ndimage.convolve(self.gaussianKernel ,dx, output=np.float64, mode='nearest')
        foy=scipy.ndimage.convolve(self.gaussianKernel ,dy, output=np.float64, mode='nearest')
        
        self.saveimage("fox.jpg",fox)
        self.saveimage("foy.jpg",foy)
        #Compute fourier transform of gaussian filter mask
        self.gaussianKernelFFTx = np.fft.fft2(fox)
        self.gaussianKernelFFTy = np.fft.fft2(foy)
        
 
        #Multiply fft of image and kernel
        self.resultx = np.fft.fftshift(self.inputImageFFT * self.gaussianKernelFFTx)
        self.resulty = np.fft.fftshift(self.inputImageFFT * self.gaussianKernelFFTy)
        
        #Perform inverse fourier transformation
        self.outputImage1 = np.abs(np.fft.ifftshift(np.fft.ifft2(self.resultx)))
        self.outputImage2 = np.abs(np.fft.ifftshift(np.fft.ifft2(self.resulty)))
        
        #Uncomment this to create the gradient images too    
        #self.saveimage("xgradimage" + str(self.KERNEL_SIZE) ,self.outputImage1)
        #self.saveimage("ygradimage"+ str(self.KERNEL_SIZE) ,self.outputImage2) 
   
        #final = np.abs(self.outputImage1+self.outputImage2)
        final = np.sqrt(self.outputImage1**2+self.outputImage2**2)
        self.saveimage("Kernelsize"+ str(self.KERNEL_SIZE),final)
      
        
        endTime = time.clock()
        return endTime - startTime
        
    def saveimage(self,filename,array):
        dirname = "Task2_2_Outputs/"
        scipy.misc.imsave(dirname+filename+".jpg" , array)
        
    

if __name__ == '__main__':
 
    filterSizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    averageRunTimes = []
   
    for size in filterSizes:
        #Run 10 times
        runTime = 0
        
        for i in range(0, 10):
            image = GaussianFilter("zebra.jpg", size)
            runTime += image.applyGaussianFilter()
             
        runTime = runTime / 10
        averageRunTimes.append(runTime)
         
      
    print "Run Complete"
    
     