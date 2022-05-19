import sys
import os

sys.path.insert(1, "..")
from snn_ttfs import *

from func import *

import numpy as np

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

# training parameters
dur = 2
dur = int(dur/dt)
init_trials = 5
trials = 100
step = 40
save_int = -1


def training(Network, f, f_out, h, init_trials, trials, dur, save_int, p):

	print("----- TRAINING -----")
	print("--- Initializing ---")

	np.save(os.path.join("model", "u_in", "u_in.npy"), Network.Per.u_in)

	Network.Per.J = Network.Gen.J.copy()

	Network.Gen.reset_activity()
	Network.Per.reset_activity()

	for i in range(init_trials):
		for t in range(dur):

			Network.Gen.step(f, t, f_out=f_out, h=h)
			Network.Per.step(f, t)


	print("STARTING TRAINING")
	for trial in range(trials):
		print(f"- Trial {trial+1} -")
		Network.train_once(dur, f, f_out, h, p)
		print(f"--- Avg Spike-Rate: {np.mean(Network.Per.spike_count)/(dur*Network.dt)} Hz")

		if save_int == -1:
			if trial == trials-1:
				np.save(os.path.join("model", "w", f"w_{trial+1}"), Network.Per.w)
				np.save(os.path.join("model", "J", f"J_{trial+1}"), Network.Per.J)
		else:
			if trial%save_int == 0 or trial == trials-1:
				np.save(os.path.join("model", "w", f"w_{trial+1}"), Network.Per.w)
				np.save(os.path.join("model", "J", f"J_{trial+1}"), Network.Per.J)


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

training(Network=N, f=f, f_out=f_out, h=h, init_trials=init_trials, trials=trials, dur=dur, save_int=save_int, p=step)
