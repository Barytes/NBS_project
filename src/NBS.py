# src/NBS.py
import numpy as np
from scipy.optimize import minimize

def solve_ESP_subproblem(ESP, N, rho, last_lamb, last_Dmax, lamb_hat, Dmax_hat, alpha, beta):
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
    # x0 = np.concatenate([lamb_hat+1, [Dmax_hat.mean()+1]])
    x0 = np.concatenate([last_lamb, [last_Dmax]])

    # 增广拉格朗日目标
    def obj(x):
        lam = x[:N]
        Dmax   = x[N]
        Dmax_arr = np.ones(N) * Dmax
        # ESP 的目标项
        # term_esp = -lambda0*theta + o/(D0-Dmax)
        term_esp = -(np.sum(lam) * theta - o / (D0 - Dmax))
        # 乘子线性项 + 二次罚项
        term_l = alpha.dot(lam - lamb_hat) + (rho/2)*np.sum((lam - lamb_hat)**2)
        term_D = beta.dot(Dmax_arr - Dmax_hat) + (rho/2)*np.sum((Dmax_arr - Dmax_hat)**2)
        return term_esp + term_l + term_D

    # 约束：sum(lam)=lambda0；0<=lam；0<=D<=D0−ε
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
        options={'ftol': 1e-6, 'maxiter': 1000, 'disp': False}
    )

    # if not sol.success:
    #     raise RuntimeError("SLSQP failed: "+sol.message)

    lamb = sol.x[:N]
    Dmax = sol.x[N]
    return lamb, Dmax

def solve_MD_subproblem(MDs, rho, last_p, last_lambhat, last_Dhat, lamb, Dmax, alpha, beta):
    N = len(MDs)
    Dmax_arr = np.ones(N) * Dmax
    # ---- 把旧值拼成初始猜测 x0 ----
    # x0 = np.ones(3 * N)  # shape = (3N,)
    x0 = np.concatenate([last_p, last_lambhat, last_Dhat])

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

    # 先加所有 p_i 的 bounds
    for md in MDs:
        bounds.append((0, md.Fn))
    # 再加所有 λ̂_i 的 bounds
    for _ in MDs:
        bounds.append((0, None))
    # 再加所有 D̂_i 的 bounds
    for _ in MDs:
        bounds.append((0, None))

    for i, md in enumerate(MDs):
        # 变量索引
        idx_p   = i
        idx_lam = N + i
        idx_D   = 2*N + i

        def lam_upper(x):
            return  x[idx_p]/(md.s * md.l) - x[idx_lam]-1e-8   # ≥0
        ineq_cons.append({'type': 'ineq', 'fun': lam_upper})
        def Dn_Dh(x):
            return x[idx_D] - md.s/md.Rn-1/(x[idx_p]/(md.s * md.l) - x[idx_lam]+1e-8)  # ≥0
        ineq_cons.append({'type': 'ineq', 'fun': Dn_Dh})

    # SLSQP 求解
    sol = minimize(
        obj_joint, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=ineq_cons,
        options={'ftol': 1e-6, 'maxiter': 1000, 'disp': False}
    )

    # if not sol.success:
    #     raise RuntimeError("SLSQP failed: "+sol.message)

    # 拆分回三个 ndarray
    x_opt = sol.x
    p        = np.asarray(x_opt[0:N])
    lamb_hat = np.asarray(x_opt[N:2*N])
    D_hat    = np.asarray(x_opt[2*N:3*N])

    return p, lamb_hat, D_hat


def ADMM(ESP,MDs):
    N = len(MDs)
    Dmax, p, lamb = ESP.D0/2, np.array([md.Fn/2 for md in MDs]), np.array([0.1 for _ in range(N)])
    lamb_hat,Dmax_hat = np.ones(N), np.ones(N)
    alpha, beta = np.ones(N), np.ones(N)
    eps_abs, eps_rel = 1e-3, 1e-3
    rho     = 100.0        # 初值
    mu, tau = 10, 2      # Boyd 推荐：μ=10, τ=2
    Dmax_old, p_old, lamb_old = 0.01, 0.01, 0.01
    lamb_hat_old, Dmax_hat_old = np.zeros(N), np.zeros(N)
    while True:
        Dmax_old, p_old, lamb_old = Dmax, p, lamb
        lamb_hat_old, Dmax_hat_old = lamb_hat, Dmax_hat
        # ESP's global subproblem
        lamb,Dmax = solve_ESP_subproblem(ESP,N,rho, lamb_old, Dmax_old,lamb_hat,Dmax_hat,alpha,beta)
        # MDs' local subproblem
        p, lamb_hat,Dmax_hat = solve_MD_subproblem(MDs,rho, p_old, lamb_hat_old, Dmax_hat_old, lamb, Dmax, alpha, beta)
        # dual variable update
        alpha += rho*(lamb-lamb_hat)
        beta += rho*([Dmax for i in range(N)]-Dmax_hat)
        # --- residuals (after primal updates) ---
        r = np.hstack([lamb_hat - lamb,        # size N
                    Dmax_hat    - Dmax])       # size N

        # Δhat 记号
        d_lamh = lamb_hat - lamb_hat_old          # Δ λ̂  (shape N,)
        d_Dh   = Dmax_hat - Dmax_hat_old          # Δ D̂  (shape N,)

        # 对偶残差（p 块在 A^T B 中系数为 0，应当被忽略）
        s_lambda = -rho * d_lamh                  # shape N
        s_Dmax   = -rho * np.sum(d_Dh)            # 标量
        s = np.hstack([s_lambda, s_Dmax])         # shape N+1

        # --- tolerances ---
        eps_pri  = np.sqrt(2*N)*eps_abs + \
                eps_rel * max(np.linalg.norm(lamb),
                                np.linalg.norm(lamb_hat),
                                abs(Dmax)*np.sqrt(N),
                                np.linalg.norm(Dmax_hat))

        eps_dual = np.sqrt(N+1)*eps_abs + \
                eps_rel * np.linalg.norm(np.hstack([alpha, beta]))

        # --- stopping test ---
        if np.linalg.norm(r,2) <= eps_pri and np.linalg.norm(s,2) <= eps_dual:
            break

        if np.linalg.norm(r,2) > mu*np.linalg.norm(s,2):
            rho *= tau
        elif np.linalg.norm(s,2) > mu*np.linalg.norm(r,2):
            rho /= tau

    print("lamb:", lamb, lamb_hat)
    print("Dmax:", Dmax, Dmax_hat)
    print("p:", p)
    return lamb, p, Dmax

def negotiation(ESP,MDs,lamb,p,Dmax):
    N = len(MDs)
    Q_star,L_star = ESP.Q(Dmax), [md.Ln(p[i]) for i,md in enumerate(MDs)]
    M,S_star = 1e5, Q_star-np.sum(L_star)
    gamma_high, gamma_low = M, ESP.omega_0/S_star
    epsilon = 1
    r0 = -1
    r = np.array([])
    while True:
        gamma = (gamma_high+gamma_low)/2
        r0 = Q_star-ESP.omega_0/gamma
        r = np.asarray([L_star[i]+md.omega_n/gamma for (i,md) in enumerate(MDs)])
        if r0 > np.sum(r)+epsilon:
            gamma_high = gamma
        elif r0 < np.sum(r)-epsilon:
            gamma_low = gamma
        else:
            break
    return r