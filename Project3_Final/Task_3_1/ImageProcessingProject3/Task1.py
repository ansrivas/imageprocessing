'''
Task 1: Implementation of Recursive Gaussian Filter
'''
import matplotlib.pyplot as plt
import scipy.misc
from gaussian import RecursiveFilter as rf

    
def plotResults(x, y):
    fig = plt.figure()
    fig.suptitle('Gaussian Filter using recursion', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Plot of sigma vs run time')

    ax.set_xlabel('Sigma')
    ax.set_ylabel('Average run time (s)')
    ax.bar(x, y, width=0.01)
    plt.savefig('Outputs/Task1_Outputs/RunTimePlot.png')

if __name__ == '__main__':
    sigmas = [0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 1.09, 1.13, 1.16, 1.19, 1.22, 1.25, 1.28, 1.31, 1.34, 1.37, 1.40, 1.43, 1.46, 1.49 ]
    #sigmas = [1.10, 1.13, 1.16, 1.19, 1.22, 1.25, 1.28, 1.31, 1.34, 1.37, 1.40, 1.43, 1.46, 1.49 ]
    #sigmas = [1.10, 1.13, 1.16, 1.19, 1.22 ]
    fileName = "cat.png"
    averageRunTimes = []
    for sigma in sigmas:
        #Run 10 times
        runTime = 0
        for i in range(0, 10):
            image = rf.RecursiveFilter(fileName, sigma)
            outputImage = image.applyGaussianFilter()   
               
            #Save the image
            imageName = 'Outputs/Task1_Outputs/Sigma' + str(sigma) + fileName
            scipy.misc.imsave(imageName,outputImage)
            
            print "Running for sigma : " + str(sigma)
                 
            runTime += image.runTime
        runTime = runTime / 10
        averageRunTimes.append(runTime)
      
    #Plot the results
    plotResults(sigmas, averageRunTimes)
    
    print "Run Complete"

