# src/NBS.py
import numpy as np


def ADMM(ESP,MDs):
    N = len(MDs)
    Dmax, p, lamb = 0, np.zeros(N), np.zeros(N)
    p_hat, lamb_hat,Dmax_hat = np.zeros(N), np.zeros(N), np.zeros(N)
    alpha, beta = np.zeros(N), np.zeros(N)
    epsilon = 1e-7
    rho = 10
    while True:
        Dmax_old, p_old, lambda_old = Dmax, p,lamb
        # ESP's global subproblem
        ESP.solve_global_subproblem()
        # MDs' local subproblem
        for md in MDs:
            md.solve_local_subproblem()
        # dual variable update
        alpha += rho*(lamb_hat-lamb)
        beta += rho*(Dmax_hat-[Dmax for i in range(N)])

        if abs(Dmax-Dmax_old)<epsilon and (abs(p_old-p)<epsilon).all() \
              or (abs(lambda_old-lamb)<epsilon).all():
            break
    return

def negotiation(ESP,MDs):
    N = len(MDs)
    Q_star,L_star = 1, np.ones(N)
    M,S_star = 1e5, Q_star-np.sum(L_star)
    gamma_high, gamma_low = M, ESP.omega_0/S_star
    epsilon = 1e-7
    while True:
        gamma = (gamma_high+gamma_low)/2
        r0 = Q_star-ESP.omega_0/gamma
        r = [L_star[i]+md.omega_n/gamma for (i,md) in enumerate(MDs)]
        if r0 > np.sum(r)+epsilon:
            gamma_high = gamma
        elif r0 < np.sum(r)-epsilon:
            gamma_low = gamma
        else:
            break
    pass