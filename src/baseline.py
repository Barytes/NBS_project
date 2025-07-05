# src/baseline.py()
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.optimize._hessian_update_strategy import BFGS

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
    eps = 1e-6
    F = np.array([md.Fn for md in MDs])  # (N,)
    s, l = MDs[0].s, MDs[0].l  # 所有MD的s和l相同
    R = np.array([md.Rn for md in MDs])  # (N,)

    # ----------- 目标函数 -----------
    def Q(Dmax):
        return lambda0*theta - o / (D0 - Dmax)

    def sum_L(p):
        # return np.sum([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)])
        res = []
        for md, pn in zip(MDs, p):
            delta = md.Fn - pn
            if delta < 0:
                delta = 0
            res.append(md.cn*(pn**2) + (md.Fn)**md.kn - (delta)**md.kn)
        return np.sum(res)

    def objective(x):
        lam = x[0:N]
        p   = x[N:2*N]
        Dm  = x[-1]
        term_esp = Q(Dm)
        term_md  = sum_L(p)
        return -(term_esp + term_md)/10000         # 最大化 → 取负

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
        res1 = p / (s * l) - lam - 1e-3
        res2 = F - p
        res3 = Dm - Dn(lam, p)
        res4 = D0 - eps - Dm
        res5 = Q(Dm) - sum_L(p) - eps
        # print("约束1:", res1)
        # print("约束2:", res2)
        # print("约束3:", res3)
        # print("约束4:", res4)
        # print("约束5:", res5)
        res.extend(res1)
        res.extend(res2)
        res.extend(res3)
        res.append(res4)
        res.append(res5)
        return np.asarray(res)

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F * 0.5
    Dm0  = 0.5 * D0
    x0   = np.concatenate([lam0, p0, [Dm0]])

    def feas_obj(x): return np.sum(np.minimum(g_ineq(x), 0)**2) + g_eq(x)**2
    x0_feas = minimize(feas_obj, x0, method='SLSQP', options={'ftol':1e-9}).x
    x0_feas = np.nan_to_num(x0_feas)

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
    # print("L(p0):",L(p0))

    cons = (
        {'type': 'eq',   'fun': g_eq},
        {'type': 'ineq', 'fun': g_ineq},
    )

    sol = minimize(
        objective, x0_feas,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={
        'ftol': 1e-9,
        'maxiter': 2000,
        'disp': True
        }
    )
    # # 等式约束：g_eq(x) == 0
    # nl_eq = NonlinearConstraint(g_eq, 0.0, 0.0)

    # # 不等式约束：g_ineq(x) >= 0
    # # （如果你想给上界也加∞，就写 (0, np.inf)）
    # nl_ineq = NonlinearConstraint(g_ineq, 0.0, np.inf)

    # sol = minimize(
    #     objective, x0_feas,
    #     method='trust-constr',
    #     jac='2-point',
    #     hess=BFGS(),               # or hess='3-point'
    #     bounds=bounds,
    #     constraints=[nl_eq, nl_ineq],
    #     options={
    #         'verbose':2,
    #         'xtol':1e-9,
    #         'gtol':1e-9,
    #         'barrier_tol':1e-9,
    #         'maxiter':20000
    #     }
    # )

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, Dmax

def solve_r_NBP(ESP, MDs, Dm, lam, p):
    N = len(MDs)
    D0  = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    w0 = ESP.omega_0
    w = [md.omega_n for md in MDs]
    eps = 1e-6

    # ----------- 目标函数 -----------
    def Q(Dmax):
        return lambda0*theta - o / (D0 - Dmax)

    def L(p_vec):
        # return np.asarray([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)])
        res = []
        for md, pn in zip(MDs, p_vec):
            delta = md.Fn - pn
            if delta < 0:
                delta = 0
            res.append(md.cn*(pn**2) + (md.Fn)**md.kn - (delta)**md.kn)
        return np.asarray(res)

    def objective(x):
        r = x[0:N]
        r = np.nan_to_num(r)  # 确保 r 中没有 NaN
        esp_arg = max(Q(Dm) - np.sum(r) + eps, eps)
        md_arg = np.maximum(r - L(p) + eps, eps)
        term_esp = w0 * np.log(esp_arg)
        term_md  = np.dot(w, np.log(md_arg))
        return -(term_esp + term_md)/10000          # 最大化 → 取负

    # ineq list
    def g_ineq(x):
        r = x[0:N]
        res = []
        # 12f  r_n ≥ 0
        res.extend(r)
        # Q(D_max)−Σr ≥ ε
        res.append( Q(Dm) - np.sum(r) - eps )
        # r_n ≥ L_n(p) + ε
        res.extend( (r - L(p) - eps).tolist() )
        return res

    # ----------- 初始可行点 -----------
    r0   = L(p) + 1  # 确保 r0 > L(p0)，避免 log(0) 问题
    x0   = r0

    # bounds
    r_bounds   = [(eps, None)] * N

    # cons = (
    #     {'type': 'ineq', 'fun': g_ineq}
    # )

    sol = minimize(
        objective, x0,
        method='SLSQP',
        bounds=r_bounds,
        options={
            'ftol': 1e-9,
            'maxiter': 2000,
            'disp': True
        }
    )

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    r_opt = sol.x[0:N]
    return r_opt

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
        # return np.asarray([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)])
        res = []
        for md, pn in zip(MDs, p):
            delta = md.Fn - pn
            if delta < 0:
                delta = 0
            res.append(md.cn*(pn**2) + (md.Fn)**md.kn - (delta)**md.kn)
        return np.asarray(res)

    def objective(x):
        lam = x[0:N]
        p   = x[N:2*N]
        r   = x[2*N:3*N]
        Dm  = x[-1]
        esp_arg = max(Q(Dm) - np.sum(r) + eps, eps)
        md_arg = np.maximum(r - L(p) + eps, eps)
        term_esp = w0 * np.log(esp_arg)
        term_md  = np.dot(w, np.log(md_arg))
        return -(term_esp + term_md)/10000          # 最大化 → 取负

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
        res.extend(p / (s * l) - lam - 1e-3)
        # 12d  p ≤ F_n
        res.extend(F - p)
        # 12e  D_n(λ,p) ≤ D_max
        res.extend(Dm - Dn(lam, p))
        # 12f  r_n ≥ 0
        res.extend(r)
        # Dm ≤ D0 - ε
        res.append(D0 - eps - Dm)
        # Q(D_max)−Σr ≥ ε
        res.append( Q(Dm) - np.sum(r) - eps )
        # r_n ≥ L_n(p) + ε
        res.extend( (r - L(p) - eps).tolist() )
        return res

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F * 0.5
    r0   = L(p0) + 1  # 确保 r0 > L(p0)，避免 log(0) 问题
    Dm0  = 0.5 * D0
    x0   = np.concatenate([lam0, p0, r0, [Dm0]])

    def feas_obj(x): return np.sum(np.minimum(g_ineq(x), 0)**2) + g_eq(x)**2
    x0_feas = minimize(feas_obj, x0, method='SLSQP', options={'ftol':1e-9}).x
    x0_feas = np.nan_to_num(x0_feas)

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
        objective, x0_feas,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={
            'ftol': 1e-9,
            'maxiter': 2000,
            'disp': True
        }
    )

    # # 等式约束：g_eq(x) == 0
    # nl_eq = NonlinearConstraint(g_eq, 0.0, 0.0)

    # # 不等式约束：g_ineq(x) >= 0
    # # （如果你想给上界也加∞，就写 (0, np.inf)）
    # nl_ineq = NonlinearConstraint(g_ineq, 0.0, 1e10)
    # sol = minimize(
    #     objective, x0_feas,
    #     method='trust-constr',
    #     jac='2-point',
    #     hess=BFGS(),               # or hess='3-point'
    #     bounds=bounds,
    #     constraints=[nl_eq, nl_ineq],
    #     options={
    #         'verbose':2,
    #         'xtol':1e-9,
    #         'gtol':1e-9,
    #         'barrier_tol':1e-9,
    #         'maxiter':20000
    #     }
    # )

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    r_opt   = sol.x[2*N:3*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, r_opt, Dmax