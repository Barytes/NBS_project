# src/baseline.py()
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
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
    p_uni = [basic_p[i]+rng.uniform(0,(md.Fn-basic_p[i])) for i,md in enumerate(MDs)]
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

def non_cooperative_baseline(ESP,MDs,verbose=False):
    # ----------------- 固定参数 ----------------- #

    N   = len(MDs)
    s,l = MDs[0].s, MDs[0].l
    R   = np.array([m.Rn for m in MDs])
    F   = np.array([m.Fn for m in MDs])
    c   = np.array([m.cn for m in MDs])
    k   = np.array([m.kn for m in MDs])
    sR = s / R
    sl = s * l
    D0  = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    eps = 1e-5
    bigM     = 1e20                    # 惩罚

    if lambda0*s*l>np.sum([md.Fn for md in MDs]):
        raise ValueError(f"λ0={lambda0} already exceeds physical limit {np.sum([md.Fn for md in MDs])/(s*l):.2f}")
    
    Dmin = np.max(s/R) + 1e-3

    def Q(Dmax):
        return  lambda0*theta - o/(D0 - Dmax)

    def L_i(i, p):
        return c[i]*p**2 + F[i]**k[i] - (F[i]-p)**k[i]
    
    def g_i(i, p, pi):
        """stationarity residual  g_i=∂U/∂p"""
        return pi - 2*c[i]*p - k[i]*(F[i]-p)**(k[i]-1)
    
    def pmin_i(i, lam_i, Dmax):
        return sl*lam_i + sl/(Dmax - sR[i])
    
    def sgap_i(i, p, pi):
        return pi*p - L_i(i, p)

    def phi_mu(a,b, mu):
        return np.sqrt(a*a + b*b + mu*mu) - a - b
    
    # ---------- 4.  decision vector packing / unpacking ----------
    # z = [pi | lam( N ) | Dmax | p( N ) | alpha( N ) | beta( N ) | s_gap( N )]
    idx_pi = 0
    idx_lam = slice(1, 1+N)
    idx_D   = 1 + N
    idx_p   = slice(2+N, 2+2*N)
    idx_al  = slice(2+2*N, 2+3*N)
    idx_be  = slice(2+3*N, 2+4*N)
    # idx_sg  = slice(2+4*N, 2+5*N)

    # dim = 2 + 5*N          # total length of z
    dim = 2 + 4*N          # total length of z

    def unpack(z):
        pi   = z[idx_pi]
        lam  = z[idx_lam]
        Dmax = z[idx_D]
        p    = z[idx_p]
        alpha= z[idx_al]
        beta = z[idx_be]
        # sgap = z[idx_sg]
        return pi, lam, Dmax, p, alpha, beta
        # return pi, lam, Dmax, p, alpha, beta, sgap

    # ---------- 5.  outer continuation parameters ----------
    outer_max_iter = 4
    mu0  = 1e-2
    rho0 = 1e2

    # ---------- 6.  bounds & equality constraint ----------
    bounds = [(0, None)]                   # pi>=0
    bounds += [(0, lambda0)]*N                # λ
    bounds += [(Dmin, ESP.D0-1e-3)]        # Dmax
    bounds += [(0, F[i]-1e-3) for i in range(N)] # p
    bounds += [(0, None)]*N                # alpha
    bounds += [(0, None)]*N                # beta
    # bounds += [(None, None)]*N             # s_gap  (free)

    def cons_lamsum(z):
        return np.sum(z[idx_lam]) - ESP.lambda0
    
    def g_ineq(z):
        res = []
        # res.extend(F - z[idx_lam]*sl-sl/(z[idx_D]-sR)-1e-3)
        res.extend((F - sl*z[idx_lam]) * (z[idx_D] - sR) - sl-1e-3* (z[idx_D] - sR))
        # res.extend(z[idx_p] - z[idx_lam]*sl-sl/(z[idx_D]-sR)-1e-3)
        res.extend((z[idx_p] - sl*z[idx_lam]) * (z[idx_D] - sR) - sl-1e-3* (z[idx_D] - sR))
        res.extend(z[idx_p] - z[idx_lam]*sl - 1e-3)
        return res

    eq_cons = {'type': 'eq', 'fun': cons_lamsum}
    ineq_cons = {'type': 'ineq', 'fun': g_ineq}

    # ---------- 7.  outer loop ----------
    #  initial guess
    lam0  = np.full(N, ESP.lambda0/N)
    p0    = np.minimum(F/2, 0.5)           # any positive
    z = np.zeros(dim)
    z[idx_pi]  = 1.0
    z[idx_lam] = lam0
    z[idx_D]   = 0.5*(Dmin+ESP.D0)
    z[idx_p]   = p0
    # z[idx_sg]  = -1.0                       # initial gap

    for it in range(outer_max_iter):
        mu  = mu0  * (0.8  ** it)
        rho = min(rho0 * (1.05 ** it), 1e2)
        if verbose:
            print(f"\n--- Outer iter {it}: mu={mu:g}, rho={rho:g} ---")
        # --- objective ---
        def obj(zvec):
            pi, lam, Dmax, p, al, be = unpack(zvec)
            leader_part = -(Q(Dmax) - pi*np.sum(p))
            pen = 0.0
            for i in range(N):
                g   = g_i(i, p[i], pi)
                a   = p[i] - pmin_i(i, lam[i], Dmax)
                b   = F[i] - p[i]
                pen += g**2 \
                    + phi_mu(a,  al[i], mu)**2 \
                    + phi_mu(b,  be[i], mu)**2 \
                    + phi_mu(p[i], -sgap_i(i,p[i],pi), mu)**2
            return leader_part + rho*pen
        # --- solve NLP ---
        # res = minimize(obj, z, method='trust-constr',
        #             constraints=[eq_cons, ineq_cons],
        #             bounds=bounds,
        #             options={'gtol':1e-8,'xtol':1e-8,'maxiter':2000,'verbose':0})
        res = minimize(obj, z, method='SLSQP',
                bounds=bounds, constraints=[eq_cons, ineq_cons],
                options={'ftol':1e-9, 'maxiter':2000,'disp':verbose})
        if not res.success:
            print("  NLP failed:", res.message)
            # break
        z = res.x
        if verbose:
            print("  inner NLP status =", res.message)
        # gap = np.max(np.abs(res.grad[idx_p]))       # crude gap measure
        # print("  inner NLP status =", res.message, " | max grad p =", gap)
        # if gap < 1e-5:
        #     break

    # ---------- 8.  extract solution ----------
    pi, lam, Dmax, p, *_ = unpack(z)
    if verbose:
        print("\n===  Final solution  ===")
        print("pi =", pi)
        print("lambda =", lam)
        print("p =", p)
        print("Dmax =", Dmax)
        print("IR gaps  =", [sgap_i(i,p[i],pi) for i in range(N)])
        print("sum lambda (should=λ0) =", lam.sum())    
    return lam, p, pi*p, Dmax

def contract_baseline_alt(ESP, MDs, verbose=False):
    """
    Contract theory baseline with major performance and stability enhancements.
    1. Sorts MDs by cost to simplify N^2 IC constraints to N-1.
    2. Removes the overly strict 'g_tight' constraint.
    3. Reformulates the delay constraint to be numerically stable (no division).
    """
    # ===================================================================
    # 1. 性能优化：按成本排序MDs以简化IC约束
    # ===================================================================
    
    # 将原始MDs列表和它们的原始索引打包
    indexed_MDs = list(enumerate(MDs))
    
    # 根据成本系数 c_n 对MDs进行升序排序 (高效的在前)
    sorted_indexed_MDs = sorted(indexed_MDs, key=lambda item: item[1].cn)
    
    # 提取排序后的MDs对象和它们的原始索引，用于最后恢复顺序
    sorted_MDs = [item[1] for item in sorted_indexed_MDs]
    original_indices = [item[0] for item in sorted_indexed_MDs]

    # --- 后续所有计算都基于排序后的 'sorted_MDs' ---
    N = len(sorted_MDs)
    D0 = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    eps = 1e-6
    
    # 使用排序后的MDs更新参数数组
    F = np.array([md.Fn for md in sorted_MDs])
    s, l = sorted_MDs[0].s, sorted_MDs[0].l
    R = np.array([md.Rn for md in sorted_MDs])

    # ----------- 目标函数 -----------
    def Q(Dmax):
        return lambda0 * theta - o / (D0 - Dmax)

    def L(p_vec, mds_list):
        # 成本函数，注意它依赖于具体的MDs列表
        return np.asarray([md.cn * (pn**2) + md.Fn**md.kn - (md.Fn - pn)**md.kn for md, pn in zip(mds_list, p_vec)])

    def objective(x):
        Dm = x[-1]
        r = x[2*N:3*N]
        # 增加一个微小的L2正则化项，提高数值稳定性
        return -(Q(Dm) - np.sum(r)) + 1e-5 * np.sum(x**2)

    # ----------- 约束 -----------
    # eq: Σλ = λ0
    def g_eq(x):
        return np.sum(x[0:N]) - lambda0

    # ineq list
    def g_ineq(x):
        # 注意：这里的 lam, p, r 变量都是按照 sorted_MDs 的顺序
        lam = x[0:N]
        p = x[N:2*N]
        r = x[2*N:3*N]
        Dm = x[-1]
        
        res = []
        
        # 基础约束 (Individual Rationality & Physical Constraints)
        # 1. λ_n <= p_n/(s*l) - ε
        res.extend(p / (s * l) - lam - eps)
        # 2. p_n <= F_n
        res.extend(F - p)
        # 3. r_n >= L_n(p_n)
        res.extend(r - L(p, sorted_MDs))
        # 4. D_max <= D0 - ε
        res.append(D0 - eps - Dm)

        # ===================================================================
        # 2. 稳定性优化：重构延迟约束，避免除法
        # ===================================================================
        # 原约束: Dm - (s/R + 1/(p/(sl)-lam)) >= 0
        # 新约束: (Dm - s/R) * (p/(sl) - lam) >= 1
        Tx = s / R
        processing_margin = p / (s * l) - lam
        res.extend((Dm - Tx) * processing_margin - 1.0)
        
        # ===================================================================
        # 1. 性能优化续：简化的IC约束 (O(N) 复杂度)
        # ===================================================================
        current_L_values = L(p, sorted_MDs)
        for i in range(N - 1):
            # 确保类型为 i 的 MD 不想模仿紧邻的、效率更低的类型 i+1 的 MD
            # U_i(p_i, r_i) >= U_i(p_{i+1}, r_{i+1})
            #  => r_i - L_i(p_i) >= r_{i+1} - L_i(p_{i+1})
            
            md_i = sorted_MDs[i]
            p_i_plus_1 = p[i+1]
            
            # 计算 md_i 在 p_{i+1} 下的成本 L_i(p_{i+1})
            # 为避免负数power导致nan，增加一个检查
            fn_minus_p = md_i.Fn - p_i_plus_1
            if fn_minus_p < 0: fn_minus_p = 0 # 物理上不可能，但为数值稳定性增加
            
            L_i_at_p_i_plus_1 = md_i.cn * p_i_plus_1**2 + md_i.Fn**md_i.kn - fn_minus_p**md_i.kn
            
            res.append((r[i] - current_L_values[i]) - (r[i+1] - L_i_at_p_i_plus_1))
            
        return res
    
    # ----------- 边界和初始点 -----------
    # bounds 和 x0 现在是基于排序后的MDs
    lam_bounds = [(0, lambda0)] * N
    p_bounds = [(0, Fi) for Fi in F]
    r_bounds = [(0, None)] * N
    D_bounds = [(0, D0 - eps)]
    bounds = lam_bounds + p_bounds + r_bounds + D_bounds

    lam0 = np.full(N, lambda0 / N)
    p0 = F * 0.5
    r0 = L(p0, sorted_MDs) + 1  # 确保初始点满足 r > L(p)
    Dm0 = 0.5 * D0
    x0 = np.concatenate([lam0, p0, r0, [Dm0]])

    # ----------- 求解器设置 -----------
    nl_eq = NonlinearConstraint(g_eq, 0, 0)
    nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)
    
    # 移除了过于严苛的 g_tight 约束
    constraints = [nl_eq, nl_ineq]

    sol = minimize(
        objective, x0,
        method='trust-constr',
        jac='2-point',
        hess=BFGS(),
        constraints=constraints,
        bounds=bounds,
        options={'verbose': 1 if verbose else 0, 'xtol': 1e-8, 'gtol': 1e-8, 'maxiter': 2000}
    )

    # ----------- 恢复原始顺序并返回结果 -----------
    if sol.success:
        # 解是按排序后的顺序得到的
        lam_sorted = sol.x[0:N]
        p_sorted = sol.x[N:2*N]
        r_sorted = sol.x[2*N:3*N]
        Dmax = sol.x[-1]

        # 创建一个空数组来存放恢复顺序后的结果
        lam_opt = np.zeros(N)
        p_opt = np.zeros(N)
        r_opt = np.zeros(N)

        # 使用 original_indices 将结果放回原位
        for i in range(N):
            original_idx = original_indices[i]
            lam_opt[original_idx] = lam_sorted[i]
            p_opt[original_idx] = p_sorted[i]
            r_opt[original_idx] = r_sorted[i]

        return lam_opt, p_opt, r_opt, Dmax
    else:
        # 如果求解失败，也返回一个符合维度的空数组或错误标识
        print(f"Contract求解失败 (N={N}): {sol.status} : {sol.message}")
        return np.zeros(N), np.zeros(N), np.zeros(N), 0

def contract_baseline(ESP,MDs,verbose=False):
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
    
    def max_Dn_smooth(lam, p, gamma=30.0):
        Dvec = Dn(lam, p)           # (N,)
        return (1.0/gamma) * np.log(np.sum(np.exp(gamma * Dvec)))

    # eq: Σλ = λ0
    def g_eq(x):
        return np.sum(x[0:N]) - lambda0
    
    def g_tight(x):
        lam = x[0:N]; p = x[N:2*N]; Dm = x[-1]
        return (Dm - max_Dn_smooth(lam,p))*100
    
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
        # # —— Incentive Compatibility 约束 —— #
        # # 对每一对 i≠j: (r[i] - L_i(p[i])) - (r[j] - L_i(p[j])) ≥ 0
        for i, md_i in enumerate(MDs):
            # 计算 md_i 的成本函数对任意 p 的值
            def L_i(p_val):
                if p_val < 0: return 1e20
                elif p_val > md_i.Fn: return 1e20
                return md_i.cn * p_val**2 + md_i.Fn**md_i.kn - (md_i.Fn - p_val)**md_i.kn
            for j, md_j in enumerate(MDs):
                if i == j:
                    continue
                # 左侧： r_i - L_i(p_i) - (r_j - L_i(p_j))
                # res.append((r[i] - L_i(p[i])) - (r[j] - L_i(p[j])))
                pass
        return res
    
    # bounds
    lam_bounds = [(0, lambda0)] * N
    p_bounds   = [(0, Fi) for Fi in F]
    r_bounds   = [(0, None)] * N
    D_bounds   = [(0, D0 - eps)]
    bounds = lam_bounds + p_bounds + r_bounds + D_bounds

    # ----------- 初始可行点 -----------
    lam0 = np.full(N, lambda0 / N)
    p0   = F * 0.5
    r0   = L(p0) + 1  # 确保 r0 > L(p0)，避免 log(0) 问题
    Dm0  = 0.5 * D0
    x0   = np.concatenate([lam0, p0, r0, [Dm0]])

    nl_eq   = NonlinearConstraint(g_eq, 0, 0)
    nl_ineq = NonlinearConstraint(g_ineq, 0, np.inf)
    nl_eq_tight = NonlinearConstraint(g_tight, 0, 0)
    constraints=[nl_eq, nl_ineq, nl_eq_tight]

    # def zero_obj(x): 
    #     return 0.0

    # x0_feas = minimize(
    #     zero_obj, x0,
    #     method='trust-constr',
    #     jac = '2-point',
    #     hess=BFGS(),
    #     constraints=[nl_eq, nl_ineq],
    #     bounds=bounds,
    #     options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    # ).x

    # # 1. 等式残差
    # heq = np.asarray(g_eq(x0_feas))
    # # 2. 不等式残差
    # hineq = np.asarray(g_ineq(x0_feas))
    # # 3. 判断是否都在容差内
    # tol_eq   = 1e-6
    # tol_ineq = -1e-6  # 小于零一点点可以接受
    # if abs(heq) <= tol_eq and hineq.min() >= tol_ineq:
    #     print("✅ 这个点严格满足所有约束（在容差范围内）。")
    # else:
    #     print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
        # raise ValueError("初始可行点不满足约束条件！")

    # cons = (
    #     {'type': 'eq',   'fun': g_eq},
    #     {'type': 'ineq', 'fun': g_ineq},
    # )

    # sol = minimize(
    #     objective, x0,
    #     method='SLSQP',
    #     bounds=bounds,
    #     constraints=cons,
    #     options={
    #         'ftol': 1e-9,
    #         'maxiter': 2000,
    #         'disp': True
    #     }
    # )

    sol = minimize(
        objective, x0,
        method='trust-constr',
        jac = '2-point',
        hess=BFGS(),
        constraints=constraints,
        bounds=bounds,
        options={'verbose': 0, 'xtol':1e-9, 'gtol':1e-9, 'maxiter':2000}
    )

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    r_opt   = sol.x[2*N:3*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, r_opt, Dmax

def social_welfare_maximization(ESP, MDs,verbose=False):
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
        if verbose: print("✅ 这个点严格满足所有约束（在容差范围内）。")
    else:
        if verbose: print("❌ 约束未全部满足，需要进一步调试或增大可行域容差。")
        # raise ValueError("初始可行点不满足约束条件！")

    sol = minimize(
        objective, x0_feas,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={
        'ftol': 1e-9,
        'maxiter': 2000,
        'disp': verbose
        }
    )

    if sol.status != 0:
        print(f"求解失败：{sol.status} : {sol.message}")

    lam_opt = sol.x[0:N]
    p_opt   = sol.x[N:2*N]
    Dmax    = sol.x[-1]
    return lam_opt, p_opt, Dmax

def solve_r_NBP(ESP, MDs, Dm, lam, p, verbose=False):
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
            'disp': verbose
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
        # raise ValueError("初始可行点不满足约束条件！")

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