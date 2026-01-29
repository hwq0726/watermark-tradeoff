import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import multiprocessing
from multiprocessing import Pool, cpu_count
import argparse
from functools import partial
import time
from utils import *


parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--s',default=1000000,type=int, help='simulation times')
parser.add_argument('--l',default=100,type=int, help='linespace of gamma and theta')
parser.add_argument('--q_k',default=0,type=int, help='the index of Q')
parser.add_argument('--p_k',default=0,type=int, help='the index of P')
parser.add_argument('--method',default='gumbel',type=str, help='the method of decoding')
parser.add_argument('--two_keys',default=None,type=str, help='the method of two keys')
parser.add_argument('--m',default=30,type=int, help='the number of layers for synthid')
parser.add_argument('--synthid_output',default=None,type=str, help='the output of synthid')
args = parser.parse_args()
print(args)

P_s = [np.array([0.25, 0.45, 0.15, 0.1, 0.05]),
       np.array([0.24, 0.35, 0.15, 0.13, 0.13]),
       np.array([0.25,0.18,0.155,0.115,0.085,0.065,0.055,0.04,0.03,0.025]),
       np.array([0.1,0.13,0.155,0.115,0.235,0.065,0.055,0.05,0.06,0.035])]
P = P_s[args.p_k]
Q_s = [np.array([0.2, 0.48, 0.12, 0.1, 0.1]),
       np.array([0.1, 0.2, 0.05, 0.48, 0.17]),
       np.array([0.27,0.20,0.15,0.11,0.08,0.06,0.05,0.035,0.025,0.02]),
       np.array([0.4,0.10,0.12,0.11,0.08,0.06,0.05,0.035,0.025,0.02])]
Q = Q_s[args.q_k]

if args.method == 'gumbel':
    decoder = gumbel_sampling
    m = None
    synthid_output = None
elif args.method == 'synthid':   
    decoder = partial(synthid_sampling, output=args.synthid_output)
    assert args.m is not None
    m = args.m
    synthid_output = args.synthid_output
# define a function to get the entropy
def simulation(g, p, q, s=1000):
    x = np.zeros(s)
    y = np.zeros(s)
    for i in range(s):
        if args.method == 'gumbel':
            U = np.random.uniform(0,1, size=(p.shape[0]))
        elif args.method == 'synthid':
            U = np.random.uniform(0,1, size=(p.shape[0], m))
        q_zeta = decoder(U, q)
        pp_zeta = compute_P_zeta(q_zeta, p, q, two_keys=args.two_keys, m=m, synthid_output=synthid_output)
        p_zeta = decoder(U, p)
        target = (1-g)*pp_zeta + g*p_zeta
        wm = entropy(p) - entropy(target)
        se = sum_of_min(target, q_zeta)
        x[i] = wm
        y[i] = se

    return np.mean(x), np.mean(y)


# Parallel computation of overlap values for a given g
def compute_t_values(l, steps):
    gs = np.linspace(0, 1, l)
    
    # Create a list of tasks for multiprocessing
    tasks = [(g, P, Q, steps) for g in gs]
    
    # Use tqdm to track progress
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.starmap(simulation, tasks),
            total=len(tasks),
            desc="Computing simulations",
            unit="sim"
        ))
    
    # Unpack results
    wm_values = [result[0] for result in results]
    se_values = [result[1] for result in results]
    
    return wm_values, se_values

num_cpus = multiprocessing.cpu_count()
print(f"Number of available CPUs: {num_cpus}")

wm_values, se_values = compute_t_values(args.l, args.s)


# Plotting the results
plt.plot(se_values, wm_values)
plt.xlabel('Speculative Sampling Efficiency')
plt.ylabel('Watermark Strength')

# Save as PDF
if args.synthid_output is not None:
    # when the synthid output is real not limit
    save_name = f"trade-off_s-{args.s}_l-{args.l}_{args.method}{m}_Hu_{args.two_keys}_Q{args.q_k}_P{args.p_k}"
else:
    save_name = f"trade-off_s-{args.s}_l-{args.l}_{args.method}_inf_Hu_{args.two_keys}_Q{args.q_k}_P{args.p_k}"

plt.savefig(f"simulation_results/plots/{save_name}.pdf")

results_para = {'wm': wm_values, 'se': se_values}
with open(f"simulation_results/{save_name}.pkl", 'wb') as f:
    pickle.dump(results_para, f)




