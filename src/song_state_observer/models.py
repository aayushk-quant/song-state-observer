import numpy as np

def log_gaussian_diag(x, mean, var):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    var = np.asarray(var, dtype=float)

    return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2 / var))

def logsumexp(values):
    values = np.asarray(values, dtype=float)

    largest = np.max(values)
    
    if np.isneginf(largest):
        return -np.inf
    
    return largest + np.log(np.sum(np.exp(values - largest)))

def forward_log(log_pi, log_A, log_B):
    T, K = log_B.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_B[0]

    for t in range (1, T):
        for new_state in range(K):
            previous_scores = log_alpha[t-1] + log_A[:, new_state]
            log_alpha[t, new_state] = logsumexp(previous_scores) + log_B[t, new_state]
    
    log_likelihood = logsumexp(log_alpha[-1])
    return log_alpha, log_likelihood

def backward_log(log_A, log_B):
    T, K = log_B.shape
    log_beta = np.empty((T, K))

    log_beta[-1] = 0.0

    for t in range(T - 2, -1, -1):
        for current_state in range(K):
            next_scores = log_A[current_state] + log_B[t + 1] + log_beta[t + 1]
            log_beta[t, current_state] = logsumexp(next_scores)
    
    return log_beta

def posteriors_log(log_alpha, log_beta, log_A, log_B, log_likelihood):
    log_alpha = np.asarray(log_alpha, dtype=float)
    log_beta = np.asarray(log_beta, dtype=float)
    log_A = np.asarray(log_A, dtype=float)
    log_B = np.asarray(log_B, dtype=float)
    T, K = log_alpha.shape
    
    log_gamma = log_alpha + log_beta - log_likelihood

    log_xi = np.empty((T - 1,K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                log_xi[t, i, j] = (
                 log_alpha[t, i] + 
                 log_A[i, j] + 
                 log_B[t+1, j] + 
                 log_beta[t+1, j] - 
                 log_likelihood
                )

    return log_gamma, log_xi

def compute_log_B(observations, means, vars_):
    observations = np.asarray(observations, dtype=float)
    means = np.asarray(means, dtype=float)
    vars_ = np.asarray(vars_, dtype=float)

    if observations.ndim != 2:
        raise ValueError("observations must have shape (T, D).")

    if means.ndim != 2:
        raise ValueError("means must have shape (K, D).")

    if vars_.shape != means.shape:
        raise ValueError("vars_ must have the same shape as means.")

    if observations.shape[1] != means.shape[1]:
        raise ValueError(
            "observations and means must have the same number of features."
            )
    
    T = observations.shape[0]
    K = means.shape[0]
    log_B = np.empty((T, K))
    
    for t in range(T):
        for k in range(K):
            log_B[t, k] = log_gaussian_diag(observations[t], means[k], vars_[k])
    
    return log_B

def m_step(gamma, xi, observations, var_floor=1e-6):
    gamma = np.asarray(gamma, dtype = float)
    xi = np.asarray(xi, dtype = float)
    observations = np.asarray(observations, dtype = float)

    T, K = gamma.shape
    D = observations.shape[1]

    if observations.shape[0] != T:
        raise ValueError(
            "gamma and observations must have same number of frames"
            )
    
    if xi.shape != (T - 1, K, K):
        raise ValueError(f"xi must have shape {(T - 1, K, K)}")
    
    pi = gamma[0].copy()
    xi_sum = xi.sum(axis=0)
    A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
    Nk = gamma.sum(axis = 0)

    means = gamma.T @ observations / Nk[:, None]

    vars_ = np.empty((K, D))

    for k in range (K):
        difference = observations - means[k]

        vars_[k] = (gamma[:, k][:, None] * difference ** 2).sum(axis=0) / Nk[k]

    vars_ = np.maximum(vars_, var_floor)

    return pi, A, means, vars_

def fit_from_initial_params(
    observations,
    pi,
    A,
    means,
    vars_,
    n_iter=100,
    tol=1e-4,
    var_floor=1e-6,
):

    observations = np.asarray(observations, dtype=float)
    pi = np.asarray(pi, dtype=float).copy()
    A = np.asarray(A, dtype=float).copy()
    means = np.asarray(means, dtype=float).copy()
    vars_ = np.asarray(vars_, dtype=float).copy()

    with np.errstate(divide="ignore"):
        log_pi = np.log(pi)
        log_A = np.log(A)

    ll_history = []
    previous_ll = -np.inf

    for iteration in range(n_iter):
        # E-step
        log_B = compute_log_B(
            observations,
            means,
            vars_,
        )

        log_alpha, ll = forward_log(
            log_pi,
            log_A,
            log_B,
        )

        log_beta = backward_log(
            log_A,
            log_B,
        )

        log_gamma, log_xi = posteriors_log(
            log_alpha,
            log_beta,
            log_A,
            log_B,
            ll,
        )

        ll_history.append(ll)

        assert ll >= previous_ll - 1e-6, (
            f"LL decreased at iteration {iteration}: "
            f"{previous_ll} -> {ll}"
        )

        if (
            iteration > 0
            and ll - previous_ll < tol
        ):
            break

        previous_ll = ll

        # M-step
        gamma = np.exp(log_gamma)
        xi = np.exp(log_xi)

        pi, A, means, vars_ = m_step(
            gamma,
            xi,
            observations,
            var_floor=var_floor,
        )

        with np.errstate(divide="ignore"):
            log_pi = np.log(pi)
            log_A = np.log(A)

    return {
        "pi": pi,
        "A": A,
        "means": means,
        "vars": vars_,
        "log_likelihoods": np.asarray(ll_history),
        "log_gamma": log_gamma,
    }

def fit(observations, K, n_iter = 100, tol = 1e-4, seed = None):
    observations = np.asarray(observations, dtype = float)
    if observations.ndim != 2:
        raise ValueError(
            "observations must have shape (T, D)."
        )

    T, D = observations.shape

    if K < 1 or K > T:
        raise ValueError(
            "K must be between 1 and the number of frames."
        )
    
    rng = np.random.default_rng(seed)

    chosen_frames = rng.choice(T, size = K, replace = False)

    means = observations[chosen_frames].copy()

    global_var = np.maximum(observations.var(axis = 0), 1e-6)

    vars_ = np.tile(global_var, (K, 1))

    pi = np.full(K, 1.0 / K)

    A = np.full((K, K), 1.0 / K)

    return fit_from_initial_params(
        observations, 
        pi, 
        A, 
        means, 
        vars_, 
        n_iter=n_iter, 
        tol=tol
        )

def viterbi_log(log_pi, log_A, log_B):

    log_pi = np.asarray(log_pi, dtype = float)
    log_A = np.asarray(log_A, dtype = float)
    log_B = np.asarray(log_B, dtype = float)

    T, K = log_B.shape

    if log_pi.shape != (K,):
        raise ValueError(f"log_pi must have shape {(K,)}.")

    if log_A.shape != (K, K):
        raise ValueError(f"log_A must have shape {(K, K)}.")
    
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype = int)

    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        for j in range(K):
            scores = delta[t - 1] + log_A[:, j]

            best_previous_state = np.argmax(scores)

            psi[t, j] = best_previous_state

            delta[t, j] = scores[best_previous_state] + log_B[t, j]

    states = np.empty(T, dtype = int)
    states[-1] = np.argmax(delta[-1])

    best_log_prob = float(delta[-1, states[-1]])

    for t in range(T - 1, 0, -1):
        states[t - 1] = psi[t, states[t]]

    return states, best_log_prob