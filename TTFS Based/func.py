import numpy as np

class Accordian(object):
	def __init__(self, T, upper, lower):
		self.T = T
		self.T_half = T/2
		self.upper = upper
		self.lower = lower

	def __call__(self, x):
		x %= self.T
		dw = (self.upper-self.lower)*np.pi/(self.T_half)
		if x <= self.T_half:
			w = self.lower*np.pi+x*dw
			return np.sin(w*x/(self.T_half))
		if x > self.T_half:
			w = self.upper*np.pi-(x-self.T_half)*dw
			return -np.sin(w*(self.T-x)/self.T_half)


def input_spike(t):
	if t <= 0.05:
		return 1
	else:
		return 0