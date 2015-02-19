# -*- coding: utf-8 -*-
'''
Task 1.3: Comparison of different methods to apply gaussian filter
'''

import scipy.misc
import matplotlib.pyplot as plt
from gaussian import FourierTransform as ft
from gaussian import NaiveConvolution as nc
from gaussian import SeparableConvolution as sc

def plotResults(x, y, filePath, titleText):
    fig = plt.figure()
    fig.suptitle('Gaussian Filter using' + titleText, fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Plot of filter size vs run time')

    ax.set_xlabel('Filter size')
    ax.set_ylabel('Average run time (s)')
    ax.bar(x, y)
    plt.savefig(filePath + 'RunTimePlot.pdf')


def plotComparisonChart(x, y1, y2, y3, legend):
    fig = plt.figure()
    fig.suptitle('Comparison of different run times', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Plot of filter size vs run time')

    #Axis Labels
    ax.set_xlabel('Filter size')
    ax.set_ylabel('Average run time (s)')
    
    #Plot
    lines = plt.plot(x, y1, x, y2, x, y3)
    
    #Line colors
    plt.setp(lines[0], 'color', 'r', 'linewidth', 5.0)
    plt.setp(lines[1], 'color', 'b', 'linewidth', 10.0)
    plt.setp(lines[2], 'color', 'g', 'linewidth', 15.0)
    
    #Legend
    plt.legend(legend, loc='upper left')
    plt.savefig('Outputs/Task1_4_Outputs/RunTimePlot.pdf')

def run(function, filePath, titleText):
    filterSizes = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
    fileName = "bauckhage.jpg"
    averageRunTimes = []
    for size in filterSizes:
        #Run 10 times
        runTime = 0
        for i in range(0, 10):
            image = function(fileName, size)
            outputImage = image.applyGaussianFilter()    
            #Save the image
            imageName = filePath + 'KernelSize' + str(int(size)) + fileName
            scipy.misc.imsave(imageName,outputImage)
               
            runTime += image.runTime
        runTime = runTime / 10
        averageRunTimes.append(runTime)
    
    #Plot results (bar charts)
    plotResults(filterSizes, averageRunTimes, filePath, titleText)
        
    return averageRunTimes

if __name__ == '__main__':
    filterSizes = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
    fourierTransformRunTime = run(ft.FourierTransform, 'Outputs/Task1_3_Outputs/', ' Fourier Transform')
    separableConvolutionRunTime = run(sc.SeparableConvolution, 'Outputs/Task1_2_Outputs/', ' Separable Convolution')
    naiveConvolutionRunTime = run(nc.NaiveConvolution, 'Outputs/Task1_1_Outputs/', ' Naive Convolution')
   
    
    
    plotComparisonChart(filterSizes, naiveConvolutionRunTime, separableConvolutionRunTime, fourierTransformRunTime, ['Naive Convolution', 'Separable Convolution', 'Fourier Transform'])
    print "Comparison Done!"
    