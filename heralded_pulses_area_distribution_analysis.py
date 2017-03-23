"""
Heralded Pulse Area Analysis
This script tries to calculate the area distribution of n=0,1,2,... traces collted by the TES, from the signal arm of SPDC.
The pulses are heralded at an efficiency eta, with each subsequently having some probability of including an accidental.
Jianwei 2016

Questions:
1) Should n_bar (used for calculating the accidental probability) be reduced if the pulses are heralded?:
The heralded pulse at the beginning of the trace 'penalises' the n_bar to be used for calculating accidental rates at the later part of the pulse.
"""

import numpy as np

import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

eta = .67 #TES efficiency
n_bar = 44.7e3*8e-6 #TES singles rate
poiss = lambda k, miu: miu**k*np.exp(-1*miu)/np.math.factorial(k)
npoiss = lambda k, miu: poiss(k,miu)/(1-poiss(0,miu))

term0 = lambda k: (1-eta)**k*npoiss(k,n_bar/eta)
term1 = lambda k: k*eta*(1-eta)**(k-1)*npoiss(k,n_bar/eta)
term2 = lambda k: nCr(k,2)*eta**2*(1-eta)**(k-2)*npoiss(k,n_bar/eta)
term3 = lambda k: nCr(k,3)*eta**3*(1-eta)**(k-3)*npoiss(k,n_bar/eta)

p0 = np.sum(map(term0, np.arange(1,10)))
p1 = np.sum(map(term1, np.arange(1,10)))
p2 = np.sum(map(term2, np.arange(2,10)))
p3 = np.sum(map(term3, np.arange(3,10)))

print((p1+p2+p3)/p0)


q0=(1-eta)*poiss(0,n_bar)
q1=eta*poiss(0,n_bar)+(1-eta)*poiss(1,n_bar)
q2=eta*poiss(1,n_bar)+(1-eta)*poiss(2,n_bar)
print((q1+q2)/q0)