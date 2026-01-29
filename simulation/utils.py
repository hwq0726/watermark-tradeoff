import numpy as np

def gumbel_sampling(U, P, eps=1e-12):
    # Add epsilon to avoid division by zero
    safe_P = np.maximum(P, eps)
    ratio = np.log(U) / safe_P
    return one_hot_vector(P.shape[0], np.argmax(ratio))

def synthid_sampling(U, p, output=None):

    k, m = U.shape  # Extract dimensions
    p = p.copy()  # Avoid modifying the original input
    
    for i in range(m):
        g = (U[:, i] >= 0.5).astype(int)  # Generate binary random vector (tournament sampling)
        
        # Compute the transformation for each probability element
        sum_p_g1 = np.sum(p[g == 1])  # Sum of probabilities where g_w' = 1
        p = p * (1 + g - sum_p_g1)  # Apply transformation
    
    if output == 'real':
        return p    # Return the real probability vector
    output = np.argmax(p)  # Return the index of the maximum probability
    return one_hot_vector(p.shape[0], output)

def one_hot_vector(size, index):
    arr = np.zeros(size)  # Create an array of zeros
    arr[index] = 1  # Set the specified index to 1
    return arr

def sum_of_min(p, q):
    return np.sum(np.minimum(p, q))

def entropy(p):
    return -np.sum(p * np.log(p + 1e-10))

def compute_P_zeta(Q_zeta, P, Q, two_keys=None, m=None, synthid_output=None):
    """
    Compute P_zeta from the given probability vectors Q_zeta, P, Q.

    Parameters
    ----------
    Q_zeta : np.ndarray
        Probability vector Q_zeta over the vocabulary (shape: [V]).
    P : np.ndarray
        Probability vector P over the vocabulary (shape: [V]).
    Q : np.ndarray
        Probability vector Q over the vocabulary (shape: [V]).

    Returns
    -------
    P_zeta : np.ndarray
        Probability vector P_zeta over the vocabulary (shape: [V]).
    """
    # Ensure numpy arrays
    Q_zeta = np.array(Q_zeta, dtype=float)
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    # Avoid division by zero by masking
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(Q > 0, P / Q, np.inf)

    # First term: Q_zeta(w) * min(1, P(w)/Q(w))
    first_term = Q_zeta * np.minimum(1, ratio)
    # Normalization constant
    norm_const = np.sum(first_term)
    # Coefficient for (P - Q)_+(w)
    coeff = 1 - norm_const
    # Positive part: (P - Q)_+(w)
    positive_part = np.maximum(P - Q, 0)
    if two_keys == 'gumbel':
        U = np.random.uniform(0,1, size=(P.shape[0]))
        positive_part = gumbel_sampling(U, positive_part)
    elif two_keys == 'synthid':
        assert m is not None
        U = np.random.uniform(0,1, size=(P.shape[0], m))
        positive_part = synthid_sampling(U, positive_part, output=synthid_output)
    
    # Second term
    second_term = coeff * positive_part
    # Final result
    P_zeta = first_term + second_term
    # Normalize to ensure it's a probability distribution (optional safety)

    return P_zeta


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