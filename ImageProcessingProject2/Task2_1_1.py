# -*- coding: utf-8 -*-
'''
Task 1.1: Implementation of Gaussian Filter using Naive Convolution in Space Domain
'''
import matplotlib.pyplot as plt
import scipy.misc
from gaussian import NaiveConvolution as NV
  
def plotResults(x, y):
    fig = plt.figure()
    fig.suptitle('Gaussian Filter using naive convolution', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Plot of filter size vs run time')

    ax.set_xlabel('Filter size')
    ax.set_ylabel('Average run time (s)')
    ax.bar(x, y)
    plt.savefig('Outputs/Task1_1_Outputs/RunTimePlot.pdf')

if __name__ == '__main__':
    filterSizes = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
    fileName = "bauckhage.jpg"
    averageRunTimes = []
    for size in filterSizes:
        #Run 10 times
        runTime = 0
        for i in range(0, 10):
            image = NV.NaiveConvolution(fileName, size)
            outputImage = image.applyGaussianFilter()   
             
            #Save the image
            imageName = 'Outputs/Task1_1_Outputs/KernelSize' + str(int(size)) + fileName
            scipy.misc.imsave(imageName,outputImage)
               
            runTime += image.runTime
        runTime = runTime / len(filterSizes)
        averageRunTimes.append(runTime)
    
    #Plot the results
    plotResults(filterSizes, averageRunTimes)
    
    print "Run Complete"

