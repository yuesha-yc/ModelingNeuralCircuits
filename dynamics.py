import numpy as np

def dynamics_solve(f, D=1, t_0=0.0, s_0=1.0, h=0.1, N=100, method="Euler"):
    """ Solves for dynamics of a given dynamical system
    
    - User must specify dimension D of phase space.
    - Includes Euler, RK2, RK4, that user can choose from using the keyword "method"
    
    Args:
        f: A python function f(t, s) that assigns a float to each time and state representing
        the time derivative of the state at that time.
        
    Kwargs:
        D: Phase space dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        s_0: Initial state (float for D=1, ndarray for D>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4"
    
    Returns:
        T: Numpy array of times
        S: Numpy array of states at the times given in T
    """
    T = np.linspace(t_0, t_0 + N * h, N + 1)
    S = np.zeros((N + 1, D)) if D > 1 else np.zeros(N + 1)
    S[0] = s_0

    for n in range(N):
        if method == "Euler":
            S[n + 1] = S[n] + h * f(T[n], S[n])

        elif method == "RK2":
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + h / 2, S[n] + k1 / 2)
            S[n + 1] = S[n] + k2

        elif method == "RK4":
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + h / 2, S[n] + k1 / 2)
            k3 = h * f(T[n] + h / 2, S[n] + k2 / 2)
            k4 = h * f(T[n] + h, S[n] + k3)
            S[n + 1] = S[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        else:
            raise ValueError("method must be 'Euler', 'RK2', or 'RK4'")

    return T, S


def hamiltonian_solve(d_qH, d_pH, d=1, t_0=0.0, q_0=0.0, p_0=1.0,
                      h=0.1, N=100, method="Euler"):
    """ Solves for dynamics of Hamiltonian system
    
    - User must specify dimension d of configuration space.
    - Includes Euler, RK2, RK4, Symplectic Euler (SE) and Stormer Verlet (SV) 
      that user can choose from using the keyword "method"
    
    Args:
        d_qH: Partial derivative of the Hamiltonian with respect to coordinates (float for d=1, ndarray for d>1)
        d_pH: Partial derivative of the Hamiltonian with respect to momenta (float for d=1, ndarray for d>1)
        
    Kwargs:
        d: Spatial dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        q_0: Initial position (float for d=1, ndarray for d>1) set to 0.0 as default
        p_0: Initial momentum (float for d=1, ndarray for d>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4", "SE", "SV"
    
    Returns:
        T: Numpy array of times
        Q: Numpy array of positions at the times given in T
        P: Numpy array of momenta at the times given in T
    """
    T = np.linspace(t_0, t_0 + N * h, N + 1)
    Q = np.zeros((N + 1, d)) if d > 1 else np.zeros(N + 1)
    P = np.zeros((N + 1, d)) if d > 1 else np.zeros(N + 1)
    Q[0], P[0] = q_0, p_0

    for n in range(N):
        if method == "Euler":  # explicit Euler
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n])
            P[n + 1] = P[n] - h * d_qH(Q[n], P[n])

        elif method == "RK2":
            k1_q = h * d_pH(Q[n], P[n])
            k1_p = -h * d_qH(Q[n], P[n])
            k2_q = h * d_pH(Q[n] + k1_q / 2, P[n] + k1_p / 2)
            k2_p = -h * d_qH(Q[n] + k1_q / 2, P[n] + k1_p / 2)
            Q[n + 1] = Q[n] + k2_q
            P[n + 1] = P[n] + k2_p

        elif method == "RK4":
            k1_q = h * d_pH(Q[n], P[n])
            k1_p = -h * d_qH(Q[n], P[n])
            k2_q = h * d_pH(Q[n] + k1_q / 2, P[n] + k1_p / 2)
            k2_p = -h * d_qH(Q[n] + k1_q / 2, P[n] + k1_p / 2)
            k3_q = h * d_pH(Q[n] + k2_q / 2, P[n] + k2_p / 2)
            k3_p = -h * d_qH(Q[n] + k2_q / 2, P[n] + k2_p / 2)
            k4_q = h * d_pH(Q[n] + k3_q, P[n] + k3_p)
            k4_p = -h * d_qH(Q[n] + k3_q, P[n] + k3_p)
            Q[n + 1] = Q[n] + (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6
            P[n + 1] = P[n] + (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6

        elif method == "SE":  # symplectic Euler (momentum first)
            P[n + 1] = P[n] - h * d_qH(Q[n], P[n])
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n + 1])

        elif method == "SV":  # Stormer–Verlet
            P_half = P[n] - 0.5 * h * d_qH(Q[n], P[n])
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P_half)
            P[n + 1] = P_half - 0.5 * h * d_qH(Q[n + 1], P_half)

        else:
            raise ValueError("method must be 'Euler', 'RK2', 'RK4', 'SE', or 'SV'")

    return T, Q, P