import math
import numpy as np
import scipy.misc
import time

class RecursiveFilter:
    def __init__(self, inputImageFileName, sigma):
        #Get the Images as array
        self.inputImage = scipy.misc.imread(inputImageFileName)
        self.outputImage = np.zeros(self.inputImage.shape)
        self.SIGMA = sigma 

        #Initialize causal and anti-causal coefficents
        self.causalCoefficientsA = np.zeros(5)
        self.causalCoefficientsB = np.zeros(5)
        self.antiCausalCoefficientsA = np.zeros(5)
        self.antiCausalCoefficientsB = np.zeros(5)
        
        #Calculate coefficients
        self._computeCoefficients()
        
        #Run Time
        self.runTime = 0
    
    def _computeCoefficients(self):
        
        sigma = self.SIGMA
        alpha = [0, 1.6800, -0.6803]
        beta = [0, 3.7350, -0.2598]
        gamma = [0, 1.7830, 1.7230]
        omega = [0, 0.6318, 1.9970]
        
        
        ''' a0+ '''
        
        self.causalCoefficientsA[0] = alpha[1] + alpha[2];
        
        
        
        
        
        ''' a1+ ''' 
        
        #a1 Part 1 of Part 1
        a1p_part1_1 = beta[2]*np.sin(omega[2]/sigma)
       
        #a1 Part 2 of Part 1
        a1p_part1_21 =  alpha[2]+(2*alpha[1])
        a1p_part1_22 = np.cos(omega[2]/sigma)
        
        #a1 Part 1
        a1p_part1 = a1p_part1_1 - (a1p_part1_21 * a1p_part1_22)
       
        #a1 Part 1 of Part 2
        a1p_part2_1 =  beta[1]*np.sin(omega[1]/sigma)
        
        #a1 Part 2 of Part 2
        a1p_part2_21 = alpha[1]+(2*alpha[2])
        a1p_part2_22 = np.cos(omega[1]/sigma)
       
        #a1 Part 2
        a1p_part2 = a1p_part2_1 - (a1p_part2_21 * a1p_part2_22)
        
        #a1 Exponential Part 1
        expo_1 = np.exp(-(gamma[2]/sigma)) 
        
        #a1 Exponential Part 2
        expo_2 = np.exp(-(gamma[1]/sigma)) 
        
        # a1 Final
        self.causalCoefficientsA[1] = (expo_1 * a1p_part1) + (expo_2 * a1p_part2)
        
        
        
        
        
        
        ''' a2+ '''
        
        # a2 PART 1
        a2p_1_part1_1 = alpha[1] + alpha[2]
        a2p_1_part1_2 = np.cos(omega[2]/sigma)
        a2p_1_part1_3 = np.cos(omega[1]/sigma)
       
        a2p_1_part1 = (a2p_1_part1_1) * (a2p_1_part1_2) *  (a2p_1_part1_3)
        
        a2p_1_part2 = (beta[1]) * (np.cos(omega[2]/sigma)) * (np.sin(omega[1]/sigma))
        
        a2p_1_part3 = (beta[2]) * (np.cos(omega[1]/sigma)) * (np.sin(omega[2]/sigma))
        
        # a2 Exponential Part 1
        expo_11 = (gamma[1] + gamma[2])/2
        expo_1 = 2 * np.exp(-(expo_11))
        
        a2p_1 = expo_1 * ((a2p_1_part1) - (a2p_1_part2) - (a2p_1_part3))
        
       
        
        # a2 Exponential Part 2
        expo_21 = gamma[1]/sigma
        expo_2 = np.exp((-2) * expo_21)
        
        #a2 PART 2 
        a2p_2 = expo_2 * alpha[2]
        
        
        
        # a2 Exponential Part 3
        expo_31 = gamma[2]/sigma
        expo_3 = np.exp((-2) * expo_31)
        
        #a2 PART 3 
        a2p_3 = expo_3 * alpha[1]
        
        # a2 Final
        self.causalCoefficientsA[2] = a2p_1 + a2p_2 + a2p_3
        
        
        
        
        
        
        
        ''' a3+ '''
        
        # a3 PART 1 , PART 2
        a3p_part1 = (beta[2] * np.sin(omega[2]/sigma)) - (alpha[2] * np.cos(omega[2]/sigma))
        a3p_part2 = (beta[1] * np.sin(omega[1]/sigma)) - (alpha[1] * np.cos(omega[1]/sigma))
        
        #a3 Exponential Part 1
        expo_111 = 2 * gamma[1]
        expo_11 = (gamma[2] + expo_111)/sigma
        expo_1 = np.exp((-1) * expo_11)
        
        #a3 Exponential Part 2
        expo_222 = 2 * gamma[2]
        expo_22 = (gamma[1] + expo_222)/sigma
        expo_2 = np.exp((-1) * expo_22)
        
        #a3 Final
        self.causalCoefficientsA[3] = (expo_1 * a3p_part1) + (expo_2 * a3p_part2) 
        
        
        
        
        
        
        ''' b1+ '''
        
        # b1 PART1, PART2
        b1p_part1 = np.cos (omega[2]/sigma)
        b1p_part2 = np.cos (omega[1]/sigma)
        
        # b1 Exponential Part 1
        expo_11 = gamma[2] / sigma
        expo_1 = (2) * np.exp((-1) * expo_11)
        
        # b1 Exponential Part 2
        expo_22 = gamma[1] / sigma
        expo_2 = (2) * np.exp((-1) * expo_22)
        
        #b1 Final
        self.causalCoefficientsB[1] = - ((expo_1 * b1p_part1) + (expo_2 * b1p_part2))
        
        
        
        
        
        
        
        ''' b2+ '''
        
        #b2 PART 1
        b2p_part1 = 4 * np.cos(omega[2]/sigma) * np.cos(omega[1]/sigma) * np.exp(- ( (gamma[1] + gamma[2]) / sigma) )
        
        # b2 Final
        self.causalCoefficientsB[2] = (b2p_part1) + np.exp((-2) * (gamma[2]/sigma)) + np.exp((-2) * (gamma[1]/sigma))
        
        
        
        
        
        
        
        ''' b3+ '''
        #b2 PART 1
        b3p_part1 = np.cos(omega[1]/sigma) * np.exp(- ( (gamma[1] + (2*gamma[2]) ) / sigma) )
        
        #b2 PART 2
        b3p_part2 = np.cos(omega[2]/sigma) * np.exp(- ( (gamma[2] + (2*gamma[1]) ) / sigma) )
        
        #b2 Final
        self.causalCoefficientsB[3] = - ((2 * b3p_part1) + (2 * b3p_part2))
        
        
        
        
        
        ''' b4+ '''
        
        #b4 Final
        self.causalCoefficientsB[4] = np.exp(- ( ( (2*gamma[2]) + (2*gamma[1]) ) / sigma) )
        
        
        
        ''' a1-'''
        
        #a1- Final
        self.antiCausalCoefficientsA[1] = self.causalCoefficientsA[0] - (self.causalCoefficientsB[1] * self.causalCoefficientsA[0])
        
        
        
        ''' a2-'''
        
        #a2- Final
        self.antiCausalCoefficientsA[2] = self.causalCoefficientsA[2] - (self.causalCoefficientsB[2] * self.causalCoefficientsA[0])
        
        
        
        ''' a3-'''
        
        #a3- Final
        self.antiCausalCoefficientsA[3] = self.causalCoefficientsA[3] - (self.causalCoefficientsB[3] * self.causalCoefficientsA[0])
        
        
        
        
        ''' a4-'''
        
        #a4- Final
        self.antiCausalCoefficientsA[4] = -self.causalCoefficientsB[4]*self.causalCoefficientsA[0]
        
        
        
        
        ''' b1- '''
        
        #b1- Final
        self.antiCausalCoefficientsB[1] = self.causalCoefficientsB[1]
        
        
        
        
        ''' b2- '''
        
        #b2- Final
        self.antiCausalCoefficientsB[2] = self.causalCoefficientsB[2]
        
        
        
        
        ''' b3- '''
        
        #b3- Final
        self.antiCausalCoefficientsB[3] = self.causalCoefficientsB[3]
        
        
        
        
        
        ''' b4- '''
        
        #b4- Final
        self.antiCausalCoefficientsB[4] = self.causalCoefficientsB[4]
        
        
        
        
        
        
    def _x(self, x, y):
        #Handle boundary conditions
        if( x < 0 or x >= self.inputImage.shape[0] or y < 0 or y >= self.inputImage.shape[1]):
            return 0
        else:
            return self.inputImage[x,y]
    
    def _get(self, array, i):
        #Handle boundary conditions
        if(i < 0 or i >= array.shape[0]):
            return 0
        else:
            return array[i]
                 
    def applyGaussianFilter(self):
        startTime = time.clock()
        
        constTerm = 1 / (self.SIGMA * np.sqrt(2 * np.pi))
        #First, we perform 1 D convolution in x-direction and then in y-direction
        
        #Convolve horizontally
        for i in range(0, self.inputImage.shape[0]):
            for j in range(0, self.inputImage.shape[1]):
                causal = antiCausal = np.zeros(self.inputImage.shape[1])
                for m in range(1, 5):                  
                    #Causal
                    causal[j] += (self.causalCoefficientsA[m-1] * self._x(i, j-(m-1))) - (self.causalCoefficientsB[m] * self._get(causal, (j-m)))
                    #Anti-Causal
                    antiCausal[j] += (self.antiCausalCoefficientsA[m] * self._x(i, j+m)) - (self.antiCausalCoefficientsB[m] * self._get(antiCausal, (j+m)))
                self.outputImage[i,j] = constTerm * (causal[j] + antiCausal[j])
        
        self.inputImage = self.outputImage
               
        #Convolve vertically
        for j in range(0, self.inputImage.shape[1]):
            for i in range(0, self.inputImage.shape[0]):
                causal = antiCausal = np.zeros(self.inputImage.shape[0])
                for m in range(1, 5):                    
                    #Causal
                    causal[i] += (self.causalCoefficientsA[m-1] * self._x(i-(m-1), j)) - (self.causalCoefficientsB[m] * self._get(causal, (i-m)))
                    #Anti-Causal
                    antiCausal[i] += (self.antiCausalCoefficientsA[m] * self._x(i+m, j)) - (self.antiCausalCoefficientsB[m] * self._get(antiCausal, (i+m)))
                self.outputImage[i,j] = constTerm * (causal[i] + antiCausal[i])
        
        endTime = time.clock()
        self.runTime = endTime - startTime
    
        #Return the image
        return self.outputImage