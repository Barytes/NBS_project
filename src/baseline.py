# src/baseline.py()
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, minimize_scalar
from scipy.optimize._hessian_update_strategy import BFGS

def uniform_baseline(ESP, MDs,seed=None):
    N = len(MDs)
    D0  = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    s, l = MDs[0].s, MDs[0].l  # 所有MD的s和l相同
    R = np.array([md.Rn for md in MDs])  # (N,)
    rng = np.random.default_rng(seed)

    lam_uni = np.full(N, lambda0 / N)  # 均匀分配 λ
    basic_p = [min(lam_uni[i]*s*l+s*l/(D0-s/R[i]),md.Fn) for i,md in enumerate(MDs)] 
    p_uni = [basic_p[i]+rng.uniform(0,0.2*(md.Fn-basic_p[i])) for i,md in enumerate(MDs)]
    Dmax = max([md.delay(p_uni[i], lam_uni[i]) for i, md in enumerate(MDs)])
    Q = lambda0 * theta - o / (D0 - Dmax)
    sum_L = np.sum([md.Ln(p_uni[i]) for i, md in enumerate(MDs)])
    r_uni = [np.maximum(Q - sum_L, 0) / N for md in MDs]  # 均匀分配 r
    return lam_uni, p_uni, r_uni, Dmax

def proportional_baseline(ESP,MDs,seed=None):
    sum_F = np.sum([md.Fn for md in MDs])
    proportion = np.array([md.Fn/sum_F for md in MDs])
    lamb = proportion * ESP.lambda0
    rng = np.random.default_rng(seed)
    basic_p = [min(lamb[i]*md.s*md.l+md.s*md.l/(ESP.D0-md.s/md.Rn),md.Fn) for i,md in enumerate(MDs)] 
    p = [basic_p[i]+rng.uniform(0,(md.Fn-basic_p[i])) for i,md in enumerate(MDs)]
    Dmax = max([md.delay(p[i], lamb[i]) for i, md in enumerate(MDs)])
    Q = ESP.lambda0 * ESP.theta - ESP.o / (ESP.D0 - Dmax)
    sum_L = np.sum([md.Ln(p[i]) for i, md in enumerate(MDs)])
    r = np.maximum(Q - sum_L, 0) * proportion  # 均匀分配 r
    return lamb, p, r, Dmax

def non_cooperative_baseline(ESP,MDs):
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

    def L(p):
        for i,md in enumerate(MDs):
            if p[i] < 0: p[i] = eps
            elif p[i] > md.Fn: p[i] = md.Fn - eps
        return [md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)]
    
    def p_star(lamb,Dmax):
        return [lamb[i]*s*l+s*l/(Dmax-s/R[i]) for i, md in enumerate(MDs)]

    def objective(x):
        lam = x[0:N]
        r   = x[N:2*N]
        Dm  = x[-1]
        return -Q(Dm)+np.sum(r)
        # return -Q(Dm)+np.sum(r)+0.1*np.linalg.norm(lam,2)

    # ----------- 约束 -----------
    # eq: Σλ = λ0
    def g_eq(x):
        lam = x[0:N]
        return np.sum(lam) - lambda0

    # ineq list
    def g_ineq(x):
        lam = x[0:N]
        r   = x[N:2*N]
        Dm  = x[-1]
        p_s = p_star(lam,Dm)
        res = []
        res.extend(r - L(p_s))
        res.extend(F - p_s)
        res.append(D0 - eps - Dm)
        return res
    
    # bounds
    lam_bounds = [(0, lambda0)] * N
    r_bounds   = [(0, lambda0*theta)] * N
    max_trans_delay = max([s/md.Rn for md in MDs])
    D_bounds   = [(max_trans_delay+eps, D0 - eps)]
    bounds = lam_bounds + r_bounds + D_bounds

    cons = (
        {'type': 'eq',   'fun': g_eq},
        {'type': 'ineq', 'fun': g_ineq},
    )

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    r0   = np.full(N, 0.01)
    Dm0  = D0 / 2
    x0   = np.concatenate([lam0, r0, [Dm0]])

    nl_eq   = NonlinearConstraint(g_eq, 0, 0)
    nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)

    def zero_obj(x): 
        return 0.0

    x0_feas = minimize(
        zero_obj, x0,
        method='trust-constr',
        jac = '2-point',
        hess=BFGS(),
        constraints=[nl_eq, nl_ineq],
        options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    ).x

    # 1. 等式残差
    heq = np.asarray(g_eq(x0_feas))
    # 2. 不等式残差
    hineq = np.asarray(g_ineq(x0_feas))
    # 3. 判断是否都在容差内
    tol_eq   = 1e-8
    tol_ineq = -1e-8  # 小于零一点点可以接受
    if abs(heq) <= tol_eq and hineq.min() >= tol_ineq:
        print("✅ 这个点严格满足所有约束（在容差范围内）。")
    else:
        print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
        raise ValueError("初始可行点不满足约束条件！")

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

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    lam_opt = sol.x[0:N]
    r_opt   = sol.x[N:2*N]
    Dmax    = sol.x[-1]
    p_opt = p_star(lam_opt, Dmax)
    for i, md in enumerate(MDs):
        pn = lam_opt[i]*s*l+s*l/(Dmax-s/R[i])
        if lam_opt[i]<=1e-2: pn = 0
        p_opt[i] = pn
    return lam_opt, p_opt, r_opt, Dmax


def non_cooperative_baseline1(ESP,MDs):
    # ----------------- 固定参数 ----------------- #
    N = len(MDs)
    D0  = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    eps = 1e-5
    F = np.array([md.Fn for md in MDs])  # (N,)
    s, l = MDs[0].s, MDs[0].l  # 所有MD的s和l相同
    R = np.array([md.Rn for md in MDs])  # (N,)
    c = np.array([md.cn for md in MDs])
    k = np.array([md.kn for md in MDs])
    bigM     = 1e20                    # 惩罚

    D_cap = D0 - 1e-3
    lambda_cap = np.sum(np.maximum(0, F/(s*l) - 1/(D_cap - s/R)))
    if lambda_cap < lambda0:
        raise ValueError(f"λ0={lambda0} already exceeds physical limit {lambda_cap:.2f}")

    def Q(Dmax):                       # ESP 收益
        return lambda0*theta - o/(D0-Dmax)

    # ----------------- follower best response ----------------- #
    def p_star(pi, lam, Dmax, idx):
        p_lo = s*l*lam + s*l/(Dmax - s/R[idx])      # 时延下界
        if p_lo > F[idx]:                # 不可行
            print(f"p_lo > F[idx]: MD {idx} : {p_lo}>{F[idx]}")
            return None

        # 目标: 取负号 → 转成最小化
        def negU(p):
            return -(pi*p - c[idx]*p**2 + (F[idx]-p)**k[idx])

        res = minimize_scalar(
                negU, bounds=(p_lo, F[idx]-eps), method='bounded',
                options={'xatol':1e-8})
        if res.status != 0:
            print(f"p_star 求解失败：{res.status} : {res.message}")
        return res.x

    # ----------------- 待优化目标 ----------------- #
    def objective(x):
        pi   = x[0]
        lam  = x[1:-1]
        Dmax = x[-1]

        p_sum = 0.0
        for n in range(N):
            p_n = p_star(pi, lam[n], Dmax, n)
            if p_n is None:                       # 不可行 → 重罚
                return bigM
            p_sum += p_n
        return -(Q(Dmax) - pi*p_sum)              # SLSQP 最小化

    # ----------------- 约束与边界 ----------------- #
    def g_eq(x):      # Σλ = λ0
        return np.sum(x[1:-1]) - lambda0
    
    def g_ineq(x):
        lam  = x[1:-1]            # λ₁…λ_N
        Dmax = x[-1]
        res = []
        res.extend(F - (s*l*lam + s*l/(Dmax - s/R))-1e-3)
        res.extend((Dmax - s/R) - 1e-3)
        return res  # element-wise

    cons = [
        {'type': 'eq',   'fun': g_eq},  # Σλ = λ0
        {'type': 'ineq', 'fun': g_ineq},      # Dmax ≥ s/R + ε, slλ + sl/(Dmax-s/R) ≤ F
    ]

    pi_bound = [(0,None)]
    lam_bounds = [(0, None)]*N
    Dmin = np.max(s/R) + eps
    D_bounds   = [(Dmin, D0 - eps)]
    bounds = pi_bound + lam_bounds + D_bounds

    # ----------------- 初始猜测 ----------------- #
    x0 = np.zeros(N+2)
    x0[0]   = 0.1                       # π
    x0[1:-1]= lambda0/N
    x0[-1]  = 0.5*(Dmin+D0-eps)

    nl_eq   = NonlinearConstraint(g_eq, 0, 0)
    nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)

    def zero_obj(x): 
        return 0.0

    x0_feas = minimize(
        zero_obj, x0,
        method='trust-constr',
        jac = '2-point',
        hess=BFGS(),
        constraints=[nl_eq, nl_ineq],
        bounds=bounds,
        options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    ).x

    # 1. 等式残差
    heq = np.asarray(g_eq(x0_feas))
    # 2. 不等式残差
    hineq = np.asarray(g_ineq(x0_feas))
    # 3. 判断是否都在容差内
    tol_eq   = 1e-8
    tol_ineq = -1e-8  # 小于零一点点可以接受
    if abs(heq) <= tol_eq and hineq.min() >= tol_ineq:
        print("✅ 这个点严格满足所有约束（在容差范围内）。")
    else:
        print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
        raise ValueError("初始可行点不满足约束条件！")

    # ----------------- 求解 ----------------- #
    # sol = minimize(
    #     objective, x0_feas,
    #     method='trust-constr',
    #     jac = '2-point',
    #     hess=BFGS(),
    #     constraints=[nl_eq, nl_ineq],
    #     bounds=bounds,
    #     options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    # )
    
    sol = minimize(objective, x0_feas, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'ftol':1e-9, 'maxiter':2000,'disp':True})

    if sol.status != 0:
        print(f"ESP leader 求解失败：{sol.status} : {sol.message}")

    pi_opt   = sol.x[0]
    lam_opt  = sol.x[1:-1]
    D_opt    = sol.x[-1]
    p_opt    = np.array([p_star(pi_opt, lam_opt[i], D_opt, i) for i in range(N)])
    print(lam_opt, p_opt, pi_opt, D_opt)
    return lam_opt, p_opt, pi_opt*p_opt, D_opt

def stackelberg_br(ESP, MDs):
    # 一、预处理
    N       = len(MDs)
    lambda0 = ESP.lambda0
    D0      = ESP.D0
    theta,o = ESP.theta, ESP.o
    s,l     = MDs[0].s, MDs[0].l
    max_iter=100
    tol,eps=1e-9,1e-6

    R = np.array([md.Rn for md in MDs])
    F = np.array([md.Fn for md in MDs])
    c = np.array([md.cn for md in MDs])
    k = np.array([md.kn for md in MDs])

    # 初始化 leader 策略
    lam  = np.ones(N) * (lambda0/N)*0.1
    Dmax = (np.max(s/R) + D0)/2
    pi   = 1.0
    p = np.zeros(N)+0.1

    # 收敛标志
    for it in range(max_iter):
        print(f"iteration {it}")
        # —————— 3. MD 最优响应 ——————
        p_new = np.zeros(N)
        for i in range(N):
            # 1) 计算可行最小功率（由延迟约束决定）
            p_lo = s*l*lam[i] + s*l/(Dmax - s/R[i])
            # 如果下界 > Fn，说明 MD 无法服务任何任务——直接 p=0
            if p_lo >= F[i]:
                print(f"p_lo > F[idx]: MD {i} : {p_lo}>{F[i]}")
                p_new[i] = 0.0
            else:
                negU = lambda x: -(pi*x - c[i]*x**2 - (F[i])**k[i] + (F[i]-x)**k[i])
                sol  = minimize_scalar(
                    negU, bounds=(p_lo, F[i]), method='bounded',
                    options={'xatol':1e-9})
                p_star = sol.x
            # 3) 计算两种方案的真实效用
            U_star = pi*p_star - (c[i]*p_star**2 + F[i]**k[i] - (F[i]-p_star)**k[i])
            U_zero = F[i]**k[i]  # 如果不贡献算力，效用就是 0

            # 4) 只要 p_star 带来的效用非正，就退回 0
            p_new[i] = p_star if U_star > U_zero else 0.0

        # 若任何 MD 都不可行，退出
        if any(pi_val is None for pi_val in p_new):
            raise RuntimeError("Follower infeasible under current leader strategy")

        # —————— 4. ESP 优化更新 ——————
        # 决策向量 x = [π, λ1...λN, Dmax]
        def leader_obj(x):
            pi_c   = x[0]
            lam_c  = x[1:1+N]
            Dm_c   = x[-1]
            # ESP 收益 = Q(Dm_c) - π * sum(p)
            Qval = lambda0*theta - o/(D0 - Dm_c)
            return -(Qval - pi_c * np.sum(p))

        def g_eq(x):      # Σλ = λ0
            return np.sum(x[1:1+N]) - lambda0
        
        def g_ineq(x):
            pi = x[0]
            lam  = x[1:1+N]            # λ₁…λ_N
            Dmax = x[-1]
            res = []
            res.extend(F - (s*l*lam + s*l/(Dmax - s/R))-eps)
            res.extend((Dmax - s/R) - eps)
            res.extend(Dmax-(s/R+1/(p_new/(s*l)-lam)))
            # res.extend(pi-p_new)
            return res  # element-wise

        cons = [
            {'type': 'eq',   'fun': g_eq},  # Σλ = λ0
            {'type': 'ineq', 'fun': g_ineq},      # Dmax ≥ s/R + ε, slλ + sl/(Dmax-s/R) ≤ F
        ]

        pi_bound = [(0,None)]
        lam_bounds = [(0, None)]*N
        Dmin = np.max(s/R) + eps
        D_bounds   = [(Dmin, D0 - eps)]
        bounds = pi_bound + lam_bounds + D_bounds

        # ----------------- 初始猜测 ----------------- #
        x0 = np.zeros(N+2)
        x0[0]   = 0.1                       # π
        x0[1:-1]= lambda0/N
        x0[-1]  = 0.5*(Dmin+D0-eps)

        nl_eq   = NonlinearConstraint(g_eq, 0, 0)
        nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)

        def zero_obj(x): 
            return 0.0

        x0_feas = minimize(
            zero_obj, x0,
            method='trust-constr',
            jac = '2-point',
            hess=BFGS(),
            constraints=[nl_eq, nl_ineq],
            bounds=bounds,
            options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
        ).x

        # 1. 等式残差
        heq = np.asarray(g_eq(x0_feas))
        # 2. 不等式残差
        hineq = np.asarray(g_ineq(x0_feas))
        # 3. 判断是否都在容差内
        tol_eq   = 1e-8
        tol_ineq = -1e-8  # 小于零一点点可以接受
        if abs(heq) <= tol_eq and hineq.min() >= tol_ineq:
            print("✅ 这个点严格满足所有约束（在容差范围内）。")
        else:
            print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
            raise ValueError("初始可行点不满足约束条件！")

        # ----------------- 求解 ----------------- #
        sol = minimize(
            leader_obj, x0_feas,
            method='trust-constr',
            jac = '2-point',
            hess=BFGS(),
            constraints=[nl_eq, nl_ineq],
            bounds=bounds,
            options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
        )

        # 提取新策略
        pi_new   = sol.x[0]
        lam_new  = sol.x[1:1+N]
        Dmax_new = sol.x[-1]

        # —————— 5. 收敛检测 ——————
        if (np.all(np.abs(lam_new-lam)<tol) and
            abs(Dmax_new-Dmax)<tol and
            abs(pi_new-pi)<tol and
            np.all(np.abs(p_new-p))<tol ):
            lam, pi, Dmax, p = lam_new, pi_new, Dmax_new, p_new
            break

        lam, pi, Dmax, p = lam_new, pi_new, Dmax_new, p_new

    return lam, p, pi*p, Dmax

def contract_baseline(ESP,MDs):
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
        return -(Q(Dm) - np.sum(r))         # 最大化 → 取负

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
        # 12e  D_n(λ,p) ≤ D_max
        res.extend(Dm - Dn(lam, p))
        # 12f  r_n - Ln(p_n) ≥ 0
        res.extend(r - L(p))
        # Dm ≤ D0 - ε
        res.append(D0 - eps - Dm)
        # —— Incentive Compatibility 约束 —— #
        # 对每一对 i≠j: (r[i] - L_i(p[i])) - (r[j] - L_i(p[j])) ≥ 0
        for i, md_i in enumerate(MDs):
            # 计算 md_i 的成本函数对任意 p 的值
            def L_i(p_val):
                if p_val < 0: p[i] = eps
                elif p_val > md_i.Fn: p_val = md_i.Fn - eps
                return md_i.cn * p_val**2 + md_i.Fn**md_i.kn - (md_i.Fn - p_val)**md_i.kn
            for j, md_j in enumerate(MDs):
                if i == j:
                    continue
                # 左侧： r_i - L_i(p_i) - (r_j - L_i(p_j))
                res.append((r[i] - L_i(p[i])) - (r[j] - L_i(p[j]))-eps)

        return res
    
    # bounds
    lam_bounds = [(eps, lambda0)] * N
    p_bounds   = [(eps, Fi) for Fi in F]
    r_bounds   = [(eps, None)] * N
    D_bounds   = [(eps, D0 - eps)]
    bounds = lam_bounds + p_bounds + r_bounds + D_bounds

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F * 0.5
    r0   = L(p0) + 1  # 确保 r0 > L(p0)，避免 log(0) 问题
    Dm0  = 0.5 * D0
    x0   = np.concatenate([lam0, p0, r0, [Dm0]])

    nl_eq   = NonlinearConstraint(g_eq, 0, 0)
    nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)

    def zero_obj(x): 
        return 0.0

    x0_feas = minimize(
        zero_obj, x0,
        method='trust-constr',
        jac = '2-point',
        hess=BFGS(),
        constraints=[nl_eq, nl_ineq],
        bounds=bounds,
        options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    ).x

    # 1. 等式残差
    heq = np.asarray(g_eq(x0_feas))
    # 2. 不等式残差
    hineq = np.asarray(g_ineq(x0_feas))
    # 3. 判断是否都在容差内
    tol_eq   = 1e-8
    tol_ineq = -1e-8  # 小于零一点点可以接受
    if abs(heq) <= tol_eq and hineq.min() >= tol_ineq:
        print("✅ 这个点严格满足所有约束（在容差范围内）。")
    else:
        print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
        raise ValueError("初始可行点不满足约束条件！")

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

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    r_opt   = sol.x[2*N:3*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, r_opt, Dmax

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
        for i,md in enumerate(MDs):
            if p[i] < 0: p[i] = eps
            elif p[i] > md.Fn: p[i] = md.Fn - eps
        return np.sum([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)])

    def objective(x):
        lam = x[0:N]
        p   = x[N:2*N]
        Dm  = x[-1]
        term_esp = Q(Dm)
        term_md  = sum_L(p)
        # return -(term_esp - term_md)
        return -(term_esp - term_md)+0.1*np.linalg.norm(lam,2)       # 最大化 → 取负

    # ----------- 约束 -----------
    def Dn(lam, p):        # (N,)
        Tx = s / R
        Tc = 1.0 / (p / (s * l) - lam)
        return np.asarray(Tx + Tc)

    # eq: Σλ = λ0
    def g_eq(x):
        lam = x[0:N]
        return np.sum(lam) - lambda0

    # ineq list
    def g_ineq(x):
        lam = x[0:N]
        p   = x[N:2*N]
        Dm  = x[-1]
        res = []
        res.extend(p / (s * l) - lam - eps)
        res.extend(F - p)
        res.extend(Dm - Dn(lam, p)- eps)
        res.append(D0 - eps - Dm)
        return res
    
    # bounds
    lam_bounds = [(eps, lambda0)] * N
    p_bounds   = [(eps, Fi) for Fi in F]
    D_bounds   = [(eps, D0 - eps)]
    bounds = lam_bounds + p_bounds + D_bounds

    cons = (
        {'type': 'eq',   'fun': g_eq},
        {'type': 'ineq', 'fun': g_ineq},
    )

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F / 2
    Dm0  = D0 / 2
    x0   = np.concatenate([lam0, p0, [Dm0]])

    nl_eq   = NonlinearConstraint(g_eq, 0, 0)
    nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)

    def zero_obj(x): 
        return 0.0

    x0_feas = minimize(
        zero_obj, x0,
        method='trust-constr',
        jac = '2-point',
        hess=BFGS(),
        constraints=[nl_eq, nl_ineq],
        bounds=bounds,
        options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    ).x

    # 1. 等式残差
    heq = np.asarray(g_eq(x0_feas))
    # 2. 不等式残差
    hineq = np.asarray(g_ineq(x0_feas))
    # 3. 判断是否都在容差内
    tol_eq   = 1e-8
    tol_ineq = -1e-8  # 小于零一点点可以接受
    if abs(heq) <= tol_eq and hineq.min() >= tol_ineq:
        print("✅ 这个点严格满足所有约束（在容差范围内）。")
    else:
        print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
        raise ValueError("初始可行点不满足约束条件！")

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
        return np.asarray([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p_vec)])

    def objective(x):
        r = x[0:N]
        r = np.nan_to_num(r)  # 确保 r 中没有 NaN
        esp_arg = np.maximum(Q(Dm) - np.sum(r), eps)
        md_arg = np.maximum(r - L(p) + eps, eps)
        term_esp = w0 * np.log(esp_arg)
        term_md  = np.dot(w, np.log(md_arg))
        return -(term_esp + term_md)          # 最大化 → 取负

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

    cons = (
        {'type': 'ineq', 'fun': g_ineq}
    )

    sol = minimize(
        objective, x0,
        method='SLSQP',
        constraints=cons,
        bounds=r_bounds,
        options={
            'ftol': 1e-8,
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
        return np.asarray([md.cn*(pn**2)+(md.Fn)**md.kn-(md.Fn-pn)**md.kn for md, pn in zip(MDs, p)])

    def objective(x):
        lam = x[0:N]
        p   = x[N:2*N]
        r   = x[2*N:3*N]
        Dm  = x[-1]
        esp_arg = np.maximum(Q(Dm) - np.sum(r), eps)
        md_arg = np.maximum(r - L(p), eps)
        term_esp = w0 * np.log(esp_arg)
        term_md  = np.dot(w, np.log(md_arg))
        # return -(term_esp + term_md)          # 最大化 → 取负
        return -(term_esp + term_md)+0.1*np.linalg.norm(lam,2)          # 最大化 → 取负

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
        # 12e  D_n(λ,p) ≤ D_max
        res.extend(Dm - Dn(lam, p))
        # 12f  r_n ≥ 0
        res.extend(r)
        # Dm ≤ D0 - ε
        res.append(D0 - eps - Dm)
        return res

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F * 0.5
    r0   = L(p0) + 1  # 确保 r0 > L(p0)，避免 log(0) 问题
    Dm0  = 0.5 * D0
    x0   = np.concatenate([lam0, p0, r0, [Dm0]])

    nl_eq   = NonlinearConstraint(g_eq, 0, 0)
    nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)

    def zero_obj(x): 
        return 0.0

    x0_feas = minimize(
        zero_obj, x0,
        method='trust-constr',
        jac = '2-point',
        hess=BFGS(),
        constraints=[nl_eq, nl_ineq],
        options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    ).x

    # 1. 等式残差
    heq = np.asarray(g_eq(x0_feas))
    # 2. 不等式残差
    hineq = np.asarray(g_ineq(x0_feas))
    # 3. 判断是否都在容差内
    tol_eq   = 1e-8
    tol_ineq = -1e-8  # 小于零一点点可以接受
    if abs(heq) <= tol_eq and hineq.min() >= tol_ineq:
        print("✅ 这个点严格满足所有约束（在容差范围内）。")
    else:
        print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
        raise ValueError("初始可行点不满足约束条件！")

    # bounds
    lam_bounds = [(eps, lambda0)] * N
    p_bounds   = [(eps, Fi) for Fi in F]
    r_bounds   = [(eps, None)] * N
    D_bounds   = [(eps, D0 - eps)]
    bounds = lam_bounds + p_bounds + r_bounds + D_bounds

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

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    r_opt   = sol.x[2*N:3*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, r_opt, Dmax