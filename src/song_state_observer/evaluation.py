import numpy as np

def states_to_boundaries(states, hop = 0.5):
    states = np.asarray(states, dtype = int)
    if states.ndim != 1:
        raise ValueError("states must be a 1-D array of decoded state labels.")
    boundaries = (np.flatnonzero(np.diff(states)) + 1) * hop
    return boundaries

def boundary_agreement(predicted, true, tolerance = 2.0):
    predicted = np.sort(np.asarray(predicted, dtype = float))
    true = np.sort(np.asarray(true, dtype = float))
    if predicted.ndim != 1 or true.ndim != 1:
        raise ValueError("predicted and true must be 1-D arrays.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    n_predicted = len(predicted)
    n_true = len(true)

    p = 0
    t = 0
    hits = 0
    
    while p < n_predicted and t < n_true:
        if abs(predicted[p] - true[t]) <= tolerance:
            hits += 1
            p += 1
            t += 1
        elif predicted[p] < true[t]:
            p += 1
        else:
            t += 1

    precision = hits / n_predicted if n_predicted > 0 else 0.0
    recall = hits / n_true if n_true > 0 else 0.0
    f_measure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "hits": hits,
        "n_predicted": n_predicted,
        "n_true": n_true,
    }

def dwell_times(states, hop = 0.5):
    states = np.asarray(states, dtype = int)
    if states.ndim != 1:
        raise ValueError("states must be a 1-D array.")
    if not np.isfinite(hop) or hop <= 0:
        raise ValueError("hop must be positive.")
    T = len(states)
    
    if T == 0:
        return {}
    change_idx = np.flatnonzero(np.diff(states)) + 1
    run_starts = np.concatenate([[0], change_idx])
    run_ends   = np.concatenate([change_idx, [T]])

    dwell = {}
    for start, end in zip(run_starts, run_ends):
        state = int(states[start])
        duration = (end - start) * hop
        dwell.setdefault(state, []).append(duration)
    
    return dwell