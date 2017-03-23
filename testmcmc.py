"""
Implements
"""
import pymc
import numpy as np
import matplotlib.pyplot as plt

# create some test data
x = np.arange(100) * 0.3
f_true = 0.1 * x**2 - 2.6*x - 1.5
np.random.seed(76523654)
noise = np.random.normal(size=100) * 1     # create some Gaussian noise
f = f_true + noise                                # add noise to the data
f_error = np.ones_like(f_true)*noise.std()

z = np.polyfit(x, f, 2)   # the traditional chi-square fit
print("The chi-square result:{}".format(z))

def model(x, f): 
	#priors
	sig = pymc.Uniform("sig", 0.0, 100.0, value=1.)
	a = pymc.Uniform("a", -10.0, 10.0, value= 0.0)
	b = pymc.Uniform("b", -10.0, 10.0, value= 0.0)
	c = pymc.Uniform("c", -10.0, 10.0, value= 0.0)

	#model
	@pymc.deterministic(plot=False)
	def mod_quadratic(x=x, a=a, b=b, c=c):
	      return a*x**2 + b*x + c

	#likelihood
	y = pymc.Normal("y", mu=mod_quadratic, tau= 1.0/sig**2, value=f, observed=True)
	return locals()

MDL = pymc.MCMC(model(x,f), db='pickle') # The sample is stored in a Python serialization (pickle) database
MDL.use_step_method(pymc.AdaptiveMetropolis,MDL.sig) # use AdaptiveMetropolis to "learn" how to step
MDL.sample(iter=2e4, burn=1e4, thin=2)  # run chain longer since there are more dimensions

# extract and plot results
y_fit = MDL.mod_quadratic.value
y_min = MDL.stats()['mod_quadratic']['quantiles'][2.5]
y_max = MDL.stats()['mod_quadratic']['quantiles'][97.5]


# plt.figure()
# plt.plot(x,f_true,'b', marker='None', ls='-', lw=1, label='True')
# plt.errorbar(x,f,yerr=f_error, color='r', marker='.', ls='None', label='Observed')
# plt.plot(x,y_fit,'k', marker='+', ls='None', ms=5, mew=2, label='Fit')
# plt.fill_between(x, y_min, y_max, color='0.5', alpha=0.5)
# plt.legend()

plt.figure()
plt.plot(x,f_true,'b', marker='None', ls='-', lw=1, label='True')
plt.errorbar(x,f,yerr=f_error, color='b', marker='o', ls='None', label='Observed')
plt.plot(x,y_fit,'k', marker='+', ls='--', ms=5, mew=2, label='Bayesian Fit Values')
plt.legend()

pymc.Matplot.plot(MDL)
plt.show()

