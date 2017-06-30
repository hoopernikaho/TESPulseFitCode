"""
Statistical functions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size']=20
def FWHM(X,Y,plot=True):
    X=np.array(X)
    Y=np.array(Y)
    half_max = np.max(Y) / 2
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[:-1])) - np.sign(half_max - np.array(Y[1:]))
    # print d
    # plt.plot(X,d*100) #if you are interested
    #find the left and right most indexes
    
    left_idx = np.where(d > 0)
    right_idx = np.where(d < 0)

    left = X[left_idx][0]
    right = X[right_idx][-1]
    print left, right
    fwhm = right-left
    if plot:
        plt.annotate('', (left, half_max), (right, half_max), arrowprops={'arrowstyle':'<->'}, weight='5')
        plt.text((left+right)/2, half_max, '{:.0f} ns'.format(fwhm), ha='center', va='bottom', weight='bold')
    return right-left #return the difference (full width)
