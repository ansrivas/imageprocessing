import math
import numpy as np
import scipy.misc
import time

class SeparableConvolution:
    def __init__(self, inputImageFileName, kernelSize):
        #Get the Images as array
        self.inputImage = scipy.misc.imread(inputImageFileName)
        self.outputImage = np.zeros(self.inputImage.shape)
        
       
        #Initialize Gaussian filter parameters
        self.KERNEL_SIZE = kernelSize
        
        #SIGMA is calculated from kernel size
        self.SIGMA = (self.KERNEL_SIZE - 1.0) / (2.0 * 2.575) 
        
        self.gaussianKernel = np.zeros([self.KERNEL_SIZE, ])
        
        #Construct the kernel of specified size
        self._constructGaussianKernel()
        
        #Run Time
        self.runTime = 0
    
    def _constructGaussianKernel(self):
        #1-D Convolution Kernel
        sigmaSquare = self.SIGMA ** 2
        constantTerm = (1/ (math.sqrt(2 * math.pi) * self.SIGMA))
                  
        for kx in range(0, int(self.KERNEL_SIZE)):     
            x = kx - int((self.KERNEL_SIZE / 2))
            self.gaussianKernel[kx] = constantTerm * (math.exp( (-x**2) / ( 2 * sigmaSquare)))
            
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
        #First, we perform 1 D convolution in x-direction and then in y-direction
        
        #Convolve horizontally
        for i in range(0, self.inputImage.shape[0]):
            for j in range(0, self.inputImage.shape[1]):
                for k in range( int(-self.KERNEL_SIZE / 2), int(self.KERNEL_SIZE / 2) + 1):
                    self.outputImage[i,j] += self._f(i,j - k) * self.gaussianKernel[int((self.KERNEL_SIZE / 2) - k)]
                    
        #Convolve vertically
        for j in range(0, self.inputImage.shape[1]):
            for i in range(0, self.inputImage.shape[0]):
                for k in range( int(-self.KERNEL_SIZE / 2), int(self.KERNEL_SIZE / 2) + 1):
                    self.outputImage[i,j] += self._f(i -k,j) * self.gaussianKernel[int((self.KERNEL_SIZE / 2) - k)]
        
        endTime = time.clock()
        self.runTime = endTime - startTime
    
        #Return the image
        return self.outputImage