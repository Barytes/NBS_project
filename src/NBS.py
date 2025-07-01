# src/NBS.py
import numpy as np
from scipy.optimize import minimize

def solve_ESP_subproblem(ESP, N, rho, lamb_hat, Dmax_hat, alpha, beta):
    """
    全局子问题：给定 lamb_hat (N,), Dmax_hat (N,), alpha (N,), beta (N,),
    返回更新后的 lamb (N,) 和 Dmax (scalar)。
    """
    D0  = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o

    lamb_hat  = np.asarray(lamb_hat)
    Dmax_hat  = np.asarray(Dmax_hat)
    alpha     = np.asarray(alpha)
    beta      = np.asarray(beta)

    # 初始猜测：全局 lambda 从 lamb_hat 开始，Dmax 用平均
    x0 = np.concatenate([lamb_hat, Dmax_hat.mean()])

    # 增广拉格朗日目标
    def obj(x):
        lam = x[:N]
        Dmax   = x[N]
        Dmax_arr = np.ones(N) * Dmax
        # ESP 的目标项
        term_esp = -lambda0*theta + o/(D0-Dmax)
        # 乘子线性项 + 二次罚项
        term_l = alpha.dot(lam - lamb_hat) + (rho/2)*np.sum((lam - lamb_hat)**2)
        term_D = beta.dot(Dmax_arr - Dmax_hat)   + (rho/2)*np.sum((Dmax_arr - Dmax_hat)**2)
        return term_esp + term_l + term_D

    # 约束：sum(lam)=lambda0；0<=lam；0<=D<=Q−ε
    cons = ({
        'type': 'eq',
        'fun': lambda x: np.sum(x[:N]) - lambda0
    },)
    bounds = [(0, None)]*N + [(0, D0 - 1e-9)]

    sol = minimize(
        obj, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-6, 'disp': False}
    )

    lamb = sol.x[:N]
    Dmax = sol.x[N]
    return lamb, Dmax

def solve_MD_subproblem(MDs, rho, p_old, lamb, Dmax, alpha, beta):
    N = len(MDs)
    Dmax_arr = np.ones(N) * Dmax
    # ---- 把旧值拼成初始猜测 x0 ----
    x0 = np.concatenate([p_old, lamb, Dmax_arr])     # shape = (3N,)

    # ---- 目标函数：各 MD 项求和 ----
    def obj_joint(x):
        p_vec  = x[0:N]
        lamh   = x[N:2*N]
        Dh     = x[2*N:3*N]

        # ∑L_n  —— 效用项
        term_u = np.sum([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p_vec)])

        # ∑  α_i(λ_i-λ̂_i) + ½ρ(λ_i-λ̂_i)²
        term_l = np.dot(alpha, (lamb - lamh)) + (rho/2)*np.sum((lamb - lamh)**2)

        # ∑  β_i(D - D̂_i) + ½ρ(D-D̂_i)²
        term_D = np.dot(beta, (Dmax_arr - Dh)) + (rho/2)*np.sum((Dmax_arr - Dh)**2)

        return term_u + term_l + term_D

    # ---- 约束与边界 ----
    bounds = []
    ineq_cons = []

    for i, md in enumerate(MDs):
        SiLi = md.s * md.l      # s_i * l_i

        # 变量索引
        idx_p   = i
        idx_lam = N + i
        idx_D   = 2*N + i

        # 1) 0 ≤ p_i ≤ F_i
        bounds.append((0, md.F_n))

        # 2) 0 ≤ λ̂_i ≤ p_i/(s_i l_i)
        bounds.append((0, None))                 # λ̂_i 下界
        def lam_upper(x, SiLi=SiLi):
            return  x[idx_p]/SiLi - x[idx_lam]-1e-9   # ≥0
        ineq_cons.append({'type': 'ineq', 'fun': lam_upper})
        def Dn_Dh(x, SiLi=SiLi):
            return md.s/md.Rn+1/(x[idx_p]/SiLi - x[idx_lam])
        ineq_cons.append({'type': 'ineq', 'fun': Dn_Dh})

        # 3) 0 ≤ D̂_i ≤ Dmax
        bounds.append((0, Dmax))

        # p_i 上界通过 bounds； λ̂_i 下界 0 已在 bounds； D̂_i 下界 0 已在 bounds
        # D̂_i 上界 Dmax 已在 bounds

    # SLSQP 求解
    sol = minimize(
        obj_joint, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=ineq_cons,
        options={'ftol': 1e-6, 'disp': False}
    )

    # 拆分回三个 ndarray
    x_opt = sol.x
    p        = np.asarray(x_opt[0:N])
    lamb_hat = np.asarray(x_opt[N:2*N])
    D_hat    = np.asarray(x_opt[2*N:3*N])

    return p, lamb_hat, D_hat


def ADMM(ESP,MDs):
    N = len(MDs)
    Dmax, p, lamb = 0, np.zeros(N), np.zeros(N)
    lamb_hat,Dmax_hat = np.zeros(N), np.zeros(N)
    alpha, beta = np.zeros(N), np.zeros(N)
    epsilon = 1e-7
    rho = 10
    while True:
        Dmax_old, p_old, lambda_old = Dmax, p,lamb
        # ESP's global subproblem
        lamb,Dmax = solve_ESP_subproblem(ESP,N,rho,lamb_hat,Dmax_hat,alpha,beta)
        # MDs' local subproblem
        p, lamb_hat,Dmax_hat = solve_MD_subproblem(MDs,rho, p_old, lamb, Dmax, alpha, beta)
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