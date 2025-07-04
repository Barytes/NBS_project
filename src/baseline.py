# src/baseline.py()
import numpy as np
from scipy.optimize import minimize

def uniform_baseline(ESP, MDs):
    N = len(MDs)
    lambda_n = ESP.lambda0/N
    s, l = MDs[0].s, MDs[0].l
    pn = lambda_n*s*l+1e7
    # solve NBP problem for r
    pass
    return

def proportional_baseline(ESP,MDs):
    N = len(MDs)
    s, l = MDs[0].s, MDs[0].l
    sum_F = np.sum([md.Fn for md in MDs])
    proportion = np.array([md.Fn/sum_F for md in MDs])
    lamb = proportion * ESP.lambda0
    p = lamb*s*l+1e7
    # solve NBP problem for r
    pass
    return

def non_cooperative_baseline(ESP,MDs):
    pass

def contract_baseline(ESP,MDs):
    pass

def social_welfare_maximization(ESP, MDs):
    N = len(MDs)
    D0  = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    w0 = ESP.omega_0
    w = [md.omega_n for md in MDs]
    eps = 1e-6
    F = np.array([md.Fn for md in MDs])  # (N,)
    s, l = MDs[0].s, MDs[0].l  # 所有MD的s和l相同
    R = np.array([md.Rn for md in MDs])  # (N,)

    # ----------- 目标函数 -----------
    def Q(Dmax):
        return lambda0*theta - o / (D0 - Dmax)

    def L(p):
        return np.sum([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)])

    def objective(x):
        lam = x[0:N]
        p   = x[N:2*N]
        Dm  = x[-1]
        term_esp = Q(Dm)
        term_md  = L(p)
        return -(term_esp + term_md)          # 最大化 → 取负

    # ----------- 约束 -----------
    def Dn(lam, p):        # (N,)
        Tx = s / R
        Tc = 1.0 / (p / (s * l) - lam)
        return Tx + Tc

    # eq: Σλ = λ0
    def g_eq(x):
        return np.sum(x[0:N]) - lambda0

    # ineq list
    def g_ineq(x):
        lam = x[0:N]
        p   = x[N:2*N]
        Dm  = x[-1]
        res = []
        # 12c  λ ≤ p/(s l) - ε
        res.extend(p / (s * l) - lam - eps)
        # 12d  p ≤ F_n
        res.extend(F - p)
        # 12e  D_n(λ,p) ≤ D_max - ε
        res.extend(Dm - Dn(lam, p))
        # Dm ≤ D0 - ε
        res.append(D0 - eps - Dm)
        return np.asarray(res)

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F * 0.5
    Dm0  = 0.5 * D0
    x0   = np.concatenate([lam0, p0, [Dm0]])

    # bounds
    lam_bounds = [(0, lambda0)] * N
    p_bounds   = [(eps, Fi) for Fi in F]
    D_bounds   = [(eps, D0 - eps)]
    bounds = lam_bounds + p_bounds + D_bounds

    # print("lam0:", lam0)
    # print("p0:", p0)
    # print("Dm0:", Dm0)
    # print("g_eq(x0):", g_eq(x0))
    # print("g_ineq(x0):", g_ineq(x0))

    cons = (
        {'type': 'eq',   'fun': g_eq},
        {'type': 'ineq', 'fun': g_ineq},
    )

    sol = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-7, 'maxiter': 1000, 'disp': False}
    )

    # if not sol.success:
    #     raise RuntimeError("SLSQP failed: "+sol.message)

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, Dmax

def optimal_NBP(ESP,MDs):
    N = len(MDs)
    D0  = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    w0 = ESP.omega_0
    w = [md.omega_n for md in MDs]
    eps = 1e-6
    F = np.array([md.Fn for md in MDs])  # (N,)
    s, l = MDs[0].s, MDs[0].l  # 所有MD的s和l相同
    R = np.array([md.Rn for md in MDs])  # (N,)

    # ----------- 目标函数 -----------
    def Q(Dmax):
        return lambda0*theta - o / (D0 - Dmax)

    def L(p):
        return np.asarray([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)])

    def objective(x):
        lam = x[0:N]
        p   = x[N:2*N]
        r   = x[2*N:3*N]
        Dm  = x[-1]
        term_esp = w0 * np.log(np.maximum(Q(Dm) - np.sum(r),1e-6))
        term_md  = np.dot(w, np.log(np.maximum(r - L(p), 1e-6)))
        return -(term_esp + term_md)          # 最大化 → 取负

    # ----------- 约束 -----------
    def Dn(lam, p):        # (N,)
        Tx = s / R
        Tc = 1.0 / (p / (s * l) - lam)
        return Tx + Tc

    # eq: Σλ = λ0
    def g_eq(x):
        return np.sum(x[0:N]) - lambda0

    # ineq list
    def g_ineq(x):
        lam = x[0:N]
        p   = x[N:2*N]
        r   = x[2*N:3*N]
        Dm  = x[-1]
        res = []
        # 12c  λ ≤ p/(s l) - ε
        res.extend(p / (s * l) - lam - eps)
        # 12d  p ≤ F_n
        res.extend(F - p)
        # 12e  D_n(λ,p) ≤ D_max - ε
        res.extend(Dm - Dn(lam, p))
        # 12f  r_n ≥ 0
        res.extend(r)
        # Dm ≤ D0 - ε
        res.append(D0 - eps - Dm)
        # r_0 argument log(Q - Σr) : Q(Dm)-Σr  ≥ eps
        res.append(Q(Dm) - np.sum(r) - eps)
        return np.asarray(res)

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F * 0.5
    r0   = L(p0) + 1  # 确保 r0 > L(p0)，避免 log(0) 问题
    Dm0  = 0.5 * D0
    x0   = np.concatenate([lam0, p0, r0, [Dm0]])

    # bounds
    lam_bounds = [(0, lambda0)] * N
    p_bounds   = [(eps, Fi) for Fi in F]
    r_bounds   = [(eps, None)] * N
    D_bounds   = [(eps, D0 - eps)]
    bounds = lam_bounds + p_bounds + r_bounds + D_bounds

    # print("lam0:", lam0)
    # print("p0:", p0)
    # print("r0:", r0)
    # print("Dm0:", Dm0)
    # print("g_eq(x0):", g_eq(x0))
    # print("g_ineq(x0):", g_ineq(x0))

    cons = (
        {'type': 'eq',   'fun': g_eq},
        {'type': 'ineq', 'fun': g_ineq},
    )

    sol = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-7, 'maxiter': 1000, 'disp': False}
    )

    # if not sol.success:
    #     raise RuntimeError("SLSQP failed: "+sol.message)

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    r_opt   = sol.x[2*N:3*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, r_opt, Dmax