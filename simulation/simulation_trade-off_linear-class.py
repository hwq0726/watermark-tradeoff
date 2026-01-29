import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import multiprocessing
from multiprocessing import Pool, cpu_count
import argparse
from utils import *


parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--s',default=10000,type=int, help='simulation times')
parser.add_argument('--l',default=1000,type=int, help='linespace of gamma and theta')
parser.add_argument('--decoder1',default='gumbel',type=str, help='the first decoder')
parser.add_argument('--decoder2',default='gumbel',type=str, help='the second decoder')
parser.add_argument('--q_k',default=0,type=int, help='the index of Q')
parser.add_argument('--p_k',default=0,type=int, help='the index of P')
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

def f(a,p): # the expected entropy of the watermarked distribution
    I = np.eye(p.shape[0])
    x = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        x[i] = entropy((1-a)*p + a*I[i])
    return np.dot(p, x)


# def synthid_sampling(U, P):
#     m = U.shape[1]  # Number of layers
#     M = 2 ** m  # Total number of initial samples
    
#     # Step 1: Sample M tokens according to P
#     sampled_tokens = np.random.choice(len(P), size=M, p=P)
    
#     # Step 2-3: Iteratively reduce the number of tokens in layers
#     for layer in range(m):
#         indices = np.arange(len(sampled_tokens))
#         np.random.shuffle(indices)  # Shuffle indices to maintain correct score-token pairing
#         sampled_tokens = sampled_tokens[indices]
#         scores = U[sampled_tokens, layer]  # Get scores for this layer
        
#         new_sampled_tokens = []
#         for i in range(0, len(sampled_tokens), 2):
#             token1, token2 = sampled_tokens[i], sampled_tokens[i+1]
#             score1, score2 = scores[i], scores[i+1]
            
#             if score1 > score2:
#                 new_sampled_tokens.append(token1)
#             elif score2 > score1:
#                 new_sampled_tokens.append(token2)
#             else:  # Tie case: break randomly
#                 new_sampled_tokens.append(np.random.choice([token1, token2]))
        
#         sampled_tokens = np.array(new_sampled_tokens)
    
#     # Final winner
#     assert len(sampled_tokens) == 1
#     return sampled_tokens[0]

def overlap(g, t, p, q, decoder1, decoder2, s=1000):
    x = np.zeros(s)
    if decoder1 == 'gumbel' and decoder2 == 'gumbel':
        for i in range(s):
            U = np.random.uniform(0,1, size=(p.shape[0]))
            p_z = one_hot_vector(p.shape[0], gumbel_sampling(U, p))
            q_z = one_hot_vector(q.shape[0], gumbel_sampling(U, q))
            x[i] = sum_of_min((1-g)*p + g*p_z, (1-t)*q + t*q_z)
    
    elif decoder1 == 'gumbel' and decoder2 == 'synthid':
        for i in range(s):
            U_gumbel = np.random.uniform(0,1, size=(p.shape[0]))
            U_synthid = np.random.uniform(0,1, size=(p.shape[0], 30))   # 30 layers for synthid
            p_z = one_hot_vector(p.shape[0], gumbel_sampling(U_gumbel, p))
            q_z = one_hot_vector(q.shape[0], synthid_sampling(U_synthid, q))
            x[i] = sum_of_min((1-g)*p + g*p_z, (1-t)*q + t*q_z)
    
    elif decoder1 == 'synthid' and decoder2 == 'gumbel':
        for i in range(s):
            U_synthid = np.random.uniform(0,1, size=(p.shape[0], 30))
            U_gumbel = np.random.uniform(0,1, size=(p.shape[0]))
            p_z = one_hot_vector(p.shape[0], synthid_sampling(U_synthid, p))
            q_z = one_hot_vector(q.shape[0], gumbel_sampling(U_gumbel, q))
            x[i] = sum_of_min((1-g)*p + g*p_z, (1-t)*q + t*q_z)
    
    elif decoder1 == 'synthid' and decoder2 == 'synthid':
        for i in range(s):
            U = np.random.uniform(0,1, size=(p.shape[0], 30))
            p_z = one_hot_vector(p.shape[0], synthid_sampling(U, p))
            q_z = one_hot_vector(q.shape[0], synthid_sampling(U, q))
            x[i] = sum_of_min((1-g)*p + g*p_z, (1-t)*q + t*q_z)

    return np.mean(x)


# Parallel computation of overlap values for a given g
def compute_t_values(g, l, decoder1, decoder2, steps):
    ts = np.linspace(0, 1, l)
    with Pool(processes=cpu_count()) as pool:  # Use max available CPUs
        t_values = pool.starmap(overlap, [(g, t, P, Q, decoder1, decoder2, steps) for t in ts])
    
    max_index = np.argmax(t_values)
    max_t_value = t_values[max_index]
    max_t = ts[max_index]
    return max_t_value, max_t

num_cpus = multiprocessing.cpu_count()
print(f"Number of available CPUs: {num_cpus}")

gamma = np.linspace(0,1,args.l)
wm = [entropy(P) - f(i,P) for i in gamma]
se = []
theta = []

for g in tqdm(gamma):
    max_t_value, max_t = compute_t_values(g, args.l, args.decoder1, args.decoder2, args.s)
    se.append(max_t_value)
    theta.append(max_t)


# Plotting the results
plt.plot(se, wm)
plt.xlabel('Speculative Sampling Efficiency')
plt.ylabel('Watermark Strength')

# Save as PDF
plt.savefig(f"simulation_results/plots/trade-off_s-{args.s}_l-{args.l}_{args.decoder1}_{args.decoder2}_Q{args.q_k}_P{args.p_k}.pdf")

results_para = {'gamma': gamma, 'wm': wm, 'se': se, 'theta': theta}
with open(f"simulation_results/trade-off_s-{args.s}_l-{args.l}_{args.decoder1}_{args.decoder2}_Q{args.q_k}_P{args.p_k}.pkl", 'wb') as f:
    pickle.dump(results_para, f)




