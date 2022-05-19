import sys
import os

sys.path.insert(1, "..")
from snn_ttfs import *

from func import *

import numpy as np
import matplotlib.pyplot as plt

# general network parameters
N_neurons = 300
dt = 0.00005
alpha = 100*dt

N_inputs = 1
input_dims = 1

N_outputs = 1
output_dims = 1

N_hints = 0
hint_dims = 0

# parameters
gg = gp = 1.5
Qg = Qp = 0.01
tm_g = tm_p = 0.01
td_g = td_p = 0.02
tr_g = tr_p = 0.002
ts_g = ts_p = 0.01
E_Lg = E_Lp = -60
v_actg = v_actp = -45
bias_g = bias_p = v_actg
std_Jg = std_Jp = gg/np.sqrt(N_neurons)
mu_wg = mu_wp = 0
std_wg = std_wp = 1
std_uing = std_uinp = 0.2

# testing parameters
dur = 2
dur = int(dur/dt)
test_trials = 50
init_trials = 1
trial_num = 100


def test(Network, f, f_out, h, test_trials, init_trials, dur):
	Network.Gen.reset_activity()
	Network.Per.reset_activity()

	for i in range(init_trials):
		for t in range(dur):
			Network.step(f, t)

	x = np.zeros(dur)
	
	total_error = 0
	E_out = 0
	V_targ = 0

	Network.reset_spike_count()

	for trial in range(test_trials):
		print(f"Test trial: {trial+1}")

		for t in range(0, dur):
			x[t] = Network.step(f, t)
			total_error += (x[t]-f_out[t][0][0])**2 
			E_out = E_out + np.dot(np.transpose(x[t]-f_out[t]), x[t]-f_out[t])
			V_targ = V_targ + np.dot(np.transpose(f_out[t]), f_out[t])

	E_norm = E_out.flatten()/V_targ.flatten()
	print("Normalized error: %g" % E_norm)
	print(f"-- MSE for test run : {total_error/(dur*test_trials)} --")
	print(f"-- Avg spike-Rate: {np.mean(Network.Per.spike_count)/(dur*Network.dt*test_trials)} Hz")

	spacing = np.linspace(0, len(f_out)*dt, len(f_out)).flatten()

	plt.plot(spacing, x, label="output")
	plt.plot(spacing, f_out.flatten(), label="target", linestyle="dashed")
	plt.plot(spacing, f.flatten(), label="input", linestyle="dashed")
	plt.title("SNN Testing, trial 10")
	plt.legend(loc="upper right")
	plt.show()


N = S_RNN(N_neurons=N_neurons, \
		N_inputs=N_inputs, \
		input_dims=input_dims, \
		N_outputs=N_outputs, \
		output_dims=output_dims, \
		alpha=alpha, \
		gg=gg, \
		gp=gp, \
		Qg=Qg, \
		Qp=Qp, \
		dt=dt, \
		tm_g=tm_g, \
		tm_p=tm_p, \
		td_g=td_g, \
		td_p=td_p, \
		tr_g=tr_g, \
		tr_p=tr_p, \
		ts_g=ts_g, \
		ts_p=ts_p, \
		E_Lg=E_Lg, \
		E_Lp=E_Lp, \
		v_actg=v_actg, \
		v_actp=v_actp, \
		bias_g=bias_g, \
		bias_p=bias_p, \
		std_Jg=std_Jg, \
		std_Jp=std_Jp, \
		mu_wg=mu_wg, \
		mu_wp=mu_wp, \
		std_wg=std_wg, \
		std_wp=std_wp, \
		std_uing=std_uing, \
		std_uinp=std_uinp, \
		hints=False, \
		N_hints=N_hints, \
		hint_dims=hint_dims)


# parameters for accordian function
target_T = 2
upper = 6
lower = 2

f = np.zeros((dur, N_inputs, input_dims))

for t in range(dur):
	f[t][0][0] = input_spike(t*dt)

accord = Accordian(target_T, upper, lower)
f_out = np.zeros((dur, N_outputs, output_dims))

for t in range(f_out.shape[0]):
	f_out[t][0][0] = accord(t*dt)

h = None 


N.Per.u_in = np.load(os.path.join("model", "u_in", "u_in.npy"))
N.Per.J = np.load(os.path.join("model", "J", f"J_{trial_num}.npy"))
N.Per.w = np.load(os.path.join("model", "w", f"w_{trial_num}.npy"))

test(Network=N, f=f, f_out=f_out, h=h, test_trials=test_trials, init_trials=init_trials, dur=dur)
