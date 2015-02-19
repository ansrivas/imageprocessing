import math
import numpy as np
import scipy.misc
import time

class FourierTransform:
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

        self.gaussianKernel = np.pad(self.gaussianKernel, ((noOfPadsUp, noOfPadsDown), (noOfPadsLeft, noOfPadsRight)), 'constant')         
    def applyGaussianFilter(self):
        startTime = time.clock()
        #Here, we convolute image and gaussian kernel in frequency domain
        
        #Compute fourier transform of input image
        self.inputImageFFT = np.fft.fft2(self.inputImage)
        
        #Padding
        self._padConstant()
        
        #Compute fourier transform of gaussian filter mask
        self.gaussianKernelFFT = np.fft.fft2(self.gaussianKernel)
        
        #Multiply fft of image and kernel
        self.result = np.fft.fftshift(self.inputImageFFT * self.gaussianKernelFFT)
        
        #Perform inverse fourier transformation
        self.outputImage = np.abs(np.fft.ifftshift(np.fft.ifft2(self.result)))
        
        endTime = time.clock()
        self.runTime = endTime - startTime
    
        #Return the image
        return self.outputImage