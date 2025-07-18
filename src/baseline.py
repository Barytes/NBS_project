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
    outer_max_iter = 5
    mu0  = 1e-6
    rho0 = 1e8

    # ---------- 6.  bounds & equality constraint ----------
    bounds = [(0, None)]                   # pi>=0
    bounds += [(0, lambda0)]*N                # λ
    bounds += [(Dmin, ESP.D0-1e-3)]        # Dmax
    bounds += [(0, F[i]-1e-2) for i in range(N)] # p
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
                options={'ftol':1e-5, 'maxiter':2000,'disp':verbose})
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
        return -(Q(Dm) - np.sum(r))

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
            
            # res.append((r[i] - current_L_values[i]) - (r[i+1] - L_i_at_p_i_plus_1))
            # 【弱化修改】: U_i(i) >= U_i(i+1) - delta
            # 我们允许U_i(i)可以比U_i(i+1)稍微差一点点，这个delta代表了MD的“懒惰”或“转换成本”
            # 这会给优化器一个“钻空子”的机会，它可以通过稍微违反IC约束来降低总成本，
            # 从而可能得到一个ESP效用更高，但社会福利更低（因为资源匹配更差）的解。
            
            decision_error_delta = 1 # 这是一个可以调整的超参数，代表弱化程度

            res.append( (r[i] - current_L_values[i]) - (r[i+1] - L_i_at_p_i_plus_1) + decision_error_delta)
            
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
        return (Dm - max_Dn_smooth(lam,p))*80
    
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
                res.append((r[i] - L_i(p[i])) - (r[j] - L_i(p[j])))
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
    # nl_eq_tight = NonlinearConstraint(g_tight, 0, 0)
    # constraints=[nl_eq, nl_ineq, nl_eq_tight]
    constraints=[nl_eq, nl_ineq]

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

def contract_baseline_menu(ESP, MDs, K=None, verbose=False):
    """
    Contract theory baseline based on a "Menu of K Contracts".

    This approach weakens the baseline realistically by reducing efficiency:
    1.  MDs are sorted by cost type (c_n).
    2.  They are partitioned into K groups.
    3.  All MDs in a single group must receive the exact same contract.
    4.  The optimizer finds the best K contracts for these K groups.
    This introduces inefficiency because one contract must serve multiple,
    slightly different MD types, leading to a lower (and more realistic)
    Social Welfare compared to a fully customized contract for each MD.
    """
    if K is None:
        # K是超参数，代表合约菜单的数量。sqrt(N)是一个合理的启发式选择。
        K = int(np.sqrt(len(MDs)))
        if K < 2: K = 2 # 至少要有两份合约

    # 1. 按成本对MDs排序
    indexed_MDs = list(enumerate(MDs))
    sorted_indexed_MDs = sorted(indexed_MDs, key=lambda item: item[1].cn)
    sorted_MDs = [item[1] for item in sorted_indexed_MDs]
    original_indices = [item[0] for item in sorted_indexed_MDs]
    
    N = len(sorted_MDs)

    # 2. 将排序后的MDs分成K组
    # md_groups是一个列表，每个元素是该组包含的MD对象列表
    md_groups = np.array_split(sorted_MDs, K)

    # --- 后续计算基于K个合约（变量维度大大降低） ---
    D0, lambda0, theta, o = ESP.D0, ESP.lambda0, ESP.theta, ESP.o
    eps = 1e-6
    s, l = sorted_MDs[0].s, sorted_MDs[0].l

    # ----------- 目标函数 (基于K个合约) -----------
    def Q(Dmax):
        return lambda0 * theta - o / (D0 - Dmax)

    def objective(x):
        # x = [λ_1..λ_K, p_1..p_K, r_1..r_K, Dm]
        p_k = x[K:2*K]
        r_k = x[2*K:3*K]
        Dm = x[-1]
        
        # 总奖励 = Σ (每个合约的奖励 * 选择该合约的人数)
        group_sizes = np.array([len(group) for group in md_groups])
        total_reward = np.sum(group_sizes * r_k)

        # 引入监督成本，代表执行复杂合约的额外开销
        monitoring_cost_factor = 0.05
        total_power_cost = np.sum(group_sizes * p_k)
        monitoring_cost = monitoring_cost_factor * total_power_cost
        
        esp_utility = Q(Dm) - total_reward - monitoring_cost
        return -esp_utility

    # ----------- 约束 (基于K个合约和N个MDs) -----------
    def g_eq(x):
        lam_k = x[0:K]
        group_sizes = np.array([len(group) for group in md_groups])
        total_lambda = np.sum(group_sizes * lam_k)
        return total_lambda - lambda0

    def g_ineq(x):
        lam_k, p_k, r_k, Dm = x[0:K], x[K:2*K], x[2*K:3*K], x[-1]
        res = []

        # -- 物理和个体理性约束 --
        # 对每个组 k 和该组中的每个 MD i，约束都必须满足
        for k, group in enumerate(md_groups):
            # 获取当前组的合约
            lam_contract, p_contract, r_contract = lam_k[k], p_k[k], r_k[k]

            # 为了简化，我们只对组内的“最差情况”进行约束，这能保证所有MD都满足条件
            # 最弱计算能力
            min_F_in_group = min(md.Fn for md in group)
            # 最差信道条件 -> 最高的传输延迟
            max_Tx_in_group = max(s / md.Rn for md in group)
            # 成本最高的MD
            most_costly_md_in_group = max(group, key=lambda md: md.cn)

            # 1. p_k <= F_i for all i in group k
            res.append(min_F_in_group - p_contract)
            
            # 2. D_n <= Dm for all i in group k (稳定版)
            processing_margin = p_contract / (s * l) - lam_contract
            res.append((Dm - max_Tx_in_group) * processing_margin - 1.0)
            
            # 3. r_k >= L_i(p_k) for all i in group k
            # 只需检查成本最高的MD
            L_worst = most_costly_md_in_group.cn * p_contract**2 + \
                      most_costly_md_in_group.Fn**most_costly_md_in_group.kn - \
                      (most_costly_md_in_group.Fn - p_contract)**most_costly_md_in_group.kn
            res.append(r_contract - L_worst)
            
        # -- 激励兼容性(IC)约束 (简化版) --
        # 确保 k 组的MD不想选 k+1 组的合约
        for k in range(K - 1):
            # U_i(k) >= U_i(k+1) for MD i in group k
            # 我们对组 k 中最容易"叛变"的MD(即效率最高的MD)进行检查
            md_i = md_groups[k][0] # 组内效率最高的MD
            
            p_contract_k, r_contract_k = p_k[k], r_k[k]
            p_contract_k1, r_contract_k1 = p_k[k+1], r_k[k+1]
            
            L_i_k = md_i.cn * p_contract_k**2 + md_i.Fn**md_i.kn - (md_i.Fn - p_contract_k)**md_i.kn
            
            fn_minus_p = md_i.Fn - p_contract_k1
            if fn_minus_p < 0: fn_minus_p = 0
            L_i_k1 = md_i.cn * p_contract_k1**2 + md_i.Fn**md_i.kn - fn_minus_p**md_i.kn
            
            res.append((r_contract_k - L_i_k) - (r_contract_k1 - L_i_k1))
            
        # 4. Dm <= D0
        res.append(D0 - eps - Dm)
        return res

    # ----------- 边界和初始点 (基于 K 个合约) -----------
    lam_bounds_k = [(eps, lambda0)] * K
    # p的上限由所有MD中最小的F决定，这是一个保守但安全的设定
    p_bounds_k = [(eps, min(md.Fn for md in MDs))] * K
    r_bounds_k = [(eps, None)] * K
    D_bounds = [(eps, D0 - eps)]
    bounds = lam_bounds_k + p_bounds_k + r_bounds_k + D_bounds
    
    x0 = np.zeros(3*K + 1)
    x0[0:K] = lambda0 / N # 初始lambda
    x0[K:2*K] = min(md.Fn for md in MDs) * 0.5 # 初始p
    x0[2*K:3*K] = 1.0 # 初始r
    x0[-1] = D0 * 0.5 # 初始Dm

    # ----------- 求解器设置 -----------
    constraints = [
        NonlinearConstraint(g_eq, 0, 0),
        NonlinearConstraint(g_ineq, 0, np.inf)
    ]

    sol = minimize(
        objective, x0,
        method='trust-constr',
        jac='2-point',
        hess=BFGS(),
        constraints=constraints,
        bounds=bounds,
        options={'verbose': 1 if verbose else 0, 'xtol': 1e-7, 'gtol': 1e-7, 'maxiter': 3000}
    )

    # ----------- 恢复原始顺序并返回结果 -----------
    if sol.success:
        lam_k, p_k, r_k, Dmax = sol.x[0:K], sol.x[K:2*K], sol.x[2*K:3*K], sol.x[-1]
        
        # 将K个合约的结果扩展到N个MD上
        lam_sorted = np.zeros(N)
        p_sorted = np.zeros(N)
        r_sorted = np.zeros(N)
        group_indices = np.array_split(np.arange(N), K)
        for k in range(K):
            for md_idx in group_indices[k]:
                lam_sorted[md_idx] = lam_k[k]
                p_sorted[md_idx] = p_k[k]
                r_sorted[md_idx] = r_k[k]

        # 将排序后的结果恢复到原始顺序
        lam_opt, p_opt, r_opt = np.zeros(N), np.zeros(N), np.zeros(N)
        for i in range(N):
            original_idx = original_indices[i]
            lam_opt[original_idx] = lam_sorted[i]
            p_opt[original_idx] = p_sorted[i]
            r_opt[original_idx] = r_sorted[i]

        return lam_opt, p_opt, r_opt, Dmax
    else:
        print(f"Contract Menu 求解失败 (N={N}, K={K}): {sol.status} : {sol.message}")
        return np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan), np.nan
    
def reverse_auction_baseline(ESP, MDs, verbose=False):
    """
    Reverse Auction Model Baseline.
    1. MDs bid their marginal cost to provide service.
    2. ESP, as the auctioneer, sorts bids from cheapest to most expensive.
    3. ESP greedily "accepts" bids (allocates tasks) to the cheapest MDs until
       the total task load (lambda0) is fulfilled.
    This approach is suboptimal because it greedily focuses on cost and may
    ignore other critical factors like channel conditions, leading to higher delay.
    """
    N = len(MDs)
    lambda0 = ESP.lambda0
    s, l = MDs[0].s, MDs[0].l

    # 1. MDs submit bids. A simple and rational bid is their energy cost coefficient c_n.
    # We also store original indices to reconstruct the final output.
    bidders = sorted([(md.cn, i, md) for i, md in enumerate(MDs)])

    lam_opt = np.zeros(N)
    p_opt = np.zeros(N)
    r_opt = np.zeros(N)
    lambda_remaining = lambda0

    # 2. ESP greedily accepts bids from cheapest to most expensive.
    for bid, original_idx, md in bidders:
        if lambda_remaining <= 0:
            break

        # For this MD, calculate the max tasks it can handle within its capacity
        # We assume it will use all its power if chosen, a common greedy assumption.
        p_assigned = md.Fn
        
        # Calculate max lambda this MD can handle without its queue becoming unstable
        max_lam_for_md = p_assigned / (s * l) - 1e-6 # Leave a small margin

        # Assign tasks: either all remaining tasks or the max it can handle.
        lam_assigned = min(lambda_remaining, max_lam_for_md)
        
        if lam_assigned <= 0:
            continue

        lam_opt[original_idx] = lam_assigned
        p_opt[original_idx] = p_assigned
        
        # 3. Payment: Pay-as-bid. Reward covers the cost.
        cost_L = md.cn * p_assigned**2 + md.Fn**md.kn - (md.Fn - p_assigned)**md.kn
        r_opt[original_idx] = cost_L  # At least cover the cost
        
        lambda_remaining -= lam_assigned

    if lambda_remaining > 1e-5:
        print(f"Auction Warning: Not all tasks were allocated. {lambda_remaining:.2f} remaining.")

    # Calculate final Dmax based on the allocation
    Dmax = 0
    for i in range(N):
        if lam_opt[i] > 0:
            delay_i = MDs[i].delay(p_opt[i], lam_opt[i])
            if delay_i > Dmax:
                Dmax = delay_i

    return lam_opt, p_opt, r_opt, Dmax

def greedy_heuristic_baseline(ESP, MDs, verbose=False):
    """
    Greedy Heuristic Baseline.
    1. Defines a composite "efficiency score" for each MD, considering its
       compute capacity, energy cost, and channel quality.
    2. ESP sorts MDs by this score (most efficient first).
    3. ESP greedily allocates tasks to the most efficient MDs until the total
       task load (lambda0) is fulfilled.
    This is suboptimal because it's a myopic, one-shot decision that cannot
    capture the complex, non-linear trade-offs of the system.
    """
    N = len(MDs)
    lambda0 = ESP.lambda0
    s, l = MDs[0].s, MDs[0].l

    # 1. Define a composite efficiency score for each MD.
    # Score = Capacity / (Energy_Cost * Transmission_Time_Factor)
    # A higher score is better.
    def get_efficiency_score(md):
        # We use a small epsilon to avoid division by zero for c_n
        # Transmission time is proportional to s/R, so R/s is a good proxy for channel quality.
        channel_quality = md.Rn / s
        score = (md.Fn / (md.cn + 1e-25)) * channel_quality
        return score

    # Sort MDs by efficiency score, from best to worst.
    bidders = sorted([(get_efficiency_score(md), i, md) for i, md in enumerate(MDs)], reverse=True)

    lam_opt = np.zeros(N)
    p_opt = np.zeros(N)
    r_opt = np.zeros(N)
    lambda_remaining = lambda0

    # 2. ESP greedily allocates tasks to the most efficient MDs.
    for score, original_idx, md in bidders:
        if lambda_remaining <= 0:
            break

        # Similar logic to the auction: assign as many tasks as possible to this MD.
        p_assigned = md.Fn
        max_lam_for_md = p_assigned / (s * l) - 1e-6
        lam_assigned = min(lambda_remaining, max_lam_for_md)

        if lam_assigned <= 0:
            continue

        lam_opt[original_idx] = lam_assigned
        p_opt[original_idx] = p_assigned
        
        # 3. Reward: Just enough to ensure participation (Individual Rationality)
        cost_L = md.cn * p_assigned**2 + md.Fn**md.kn - (md.Fn - p_assigned)**md.kn
        r_opt[original_idx] = cost_L + 1e-5 # Pay slightly more than cost
        
        lambda_remaining -= lam_assigned

    if lambda_remaining > 1e-5:
        print(f"Greedy Warning: Not all tasks were allocated. {lambda_remaining:.2f} remaining.")

    # Calculate final Dmax
    Dmax = 0
    for i in range(N):
        if lam_opt[i] > 0:
            delay_i = MDs[i].delay(p_opt[i], lam_opt[i])
            if delay_i > Dmax:
                Dmax = delay_i

    return lam_opt, p_opt, r_opt, Dmax

def social_welfare_maximization(ESP, MDs, verbose=False):
    N = len(MDs)
    D0 = ESP.D0
    lambda0 = ESP.lambda0
    theta = ESP.theta
    o = ESP.o
    eps = 1e-5
    F = np.array([md.Fn for md in MDs])
    s, l = MDs[0].s, MDs[0].l
    R = np.array([md.Rn for md in MDs])

    # ----------- 目标函数 -----------
    def Q(Dmax):
        return lambda0 * theta - o / (D0 - Dmax)

    def sum_L(p):
        # 确保p在边界内，避免 (F-p) 项出现负数或零
        p_clipped = np.clip(p, eps, F - eps)
        return np.sum([md.cn * (pn**2) + md.Fn**md.kn - (md.Fn - pn)**md.kn for md, pn in zip(MDs, p_clipped)])

    def objective(x):
        lam = x[0:N]
        p = x[N:2*N]
        Dm = x[-1]
        term_esp = Q(Dm)
        term_md = sum_L(p)
        return -(term_esp - term_md) + 0.1 * np.linalg.norm(lam, 2)  # 最大化 → 取负

    # ----------- 约束 -----------
    # eq: Σλ = λ0
    def g_eq(x):
        return np.sum(x[0:N]) - lambda0

    # ineq list
    def g_ineq(x):
        lam = x[0:N]
        p = x[N:2*N]
        Dm = x[-1]
        res = []
        # 1. λ_n <= p_n/(s*l) - ε
        res.extend(p / (s * l) - lam - eps)
        # 2. p_n <= F_n
        res.extend(F - p)
        # 3. D_max <= D0 - ε
        res.append(D0 - eps - Dm)

        # 4. 【稳定版】D_n(λ,p) ≤ D_max
        # 原约束: Dm - (s/R + 1/(p/(sl)-lam)) >= 0
        # 新约束: (Dm - s/R) * (p/(sl) - lam) >= 1
        Tx = s / R
        processing_margin = p / (s * l) - lam
        res.extend((Dm - Tx) * processing_margin - 1.0)
        
        return res
    
    # ----------- 边界、约束和初始点 -----------
    # 确保边界稍微向内收缩，为求解器提供空间
    lam_bounds = [(0, lambda0)] * N
    p_bounds = [(0, None) for Fi in F]
    D_bounds = [(0, None)]
    bounds = lam_bounds + p_bounds + D_bounds

    cons = (
        {'type': 'eq', 'fun': g_eq},
        {'type': 'ineq', 'fun': g_ineq},
    )

    # 直接构造一个简单的、确保在边界内的初始点 x0
    lam0 = np.full(N, lambda0 / N)
    p0 = F / 2
    # 确保初始 Dm0 > max(初始Dn)，避免初始点就违反约束
    # init_dns = (s/R) + 1.0 / (p0 / (s * l) - lam0)
    # Dm0 = np.max(init_dns) + 0.1 # 加一个缓冲
    Dm0 = 0.5
    x0 = np.concatenate([lam0, p0, [Dm0]])

    # ===================================================================
    # 移除了复杂且脆弱的可行点搜索过程
    # 直接使用SLSQP进行求解
    # ===================================================================

    sol = minimize(
        objective, x0,  # 直接使用我们构造的简单初始点 x0
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={
            'ftol': 1e-9,
            'maxiter': 2000,
            'disp': verbose
        }
    )

    if not sol.success:
        print(f"SWM 求解失败 (N={N}): {sol.message}")
        # 在失败时返回一个合理的值或NaN，避免后续计算崩溃
        return np.full(N, np.nan), np.full(N, np.nan), np.nan

    lam_opt = sol.x[0:N]
    p_opt = sol.x[N:2*N]
    Dmax = sol.x[-1]
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