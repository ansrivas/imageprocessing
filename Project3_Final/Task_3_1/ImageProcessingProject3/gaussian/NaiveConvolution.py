import math
import numpy as np
import scipy.misc
import time

class NaiveConvolution:
    def __init__(self, inputImageFileName, kernelSize):
        #Get the Images as array
        self.inputImage = scipy.misc.imread(inputImageFileName)
        self.outputImage = np.zeros(self.inputImage.shape)
        
       
        #Initialize Gaussian filter parameters
        self.KERNEL_SIZE = kernelSize
        
        #SIGMA is calculated from kernel size
        self.SIGMA = (self.KERNEL_SIZE - 1.0) / (2.0 * 2.575) 
        
        
        self.gaussianKernel = np.zeros([self.KERNEL_SIZE, self.KERNEL_SIZE])
        
        #Construct the kernel of specified size
        self._constructGaussianKernel()
        
        #Run Time
        self.runTime = 0
    
    
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
    
    def _f(self, x, y):
        #Handle boundary conditions
        if( x < 0 or x >= self.inputImage.shape[0] or y < 0 or y >= self.inputImage.shape[1]):
            return 0
        else:
            return self.inputImage[x,y]
             
        
    def applyGaussianFilter(self):
        startTime = time.clock()
        #Here, we convolute image and gaussian kernel
        iterator = np.nditer(self.inputImage, flags=['multi_index'])
        
        while not iterator.finished:
            x, y = iterator.multi_index
            
            for i in range( int(-self.KERNEL_SIZE / 2), int(self.KERNEL_SIZE / 2) + 1):
                for j in range( int(-self.KERNEL_SIZE / 2), int(self.KERNEL_SIZE / 2) + 1):
                    self.outputImage[x,y] += self._f(x - i, y - j) * self.gaussianKernel[int(self.KERNEL_SIZE /2) - i, int(self.KERNEL_SIZE /2) - j] 
            
            
            iterator.iternext()
            
        
        endTime = time.clock()
        self.runTime = endTime - startTime
    
        #Return the image
        return self.outputImage
