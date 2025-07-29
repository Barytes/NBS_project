function optimal_NBP() % <--- 【核心修改】将脚本包裹成一个函数

    % --- 0. 初始化环境 ---
    % clear; clc; close all; % clear现在是函数内部的，更安全
    
    % --- 1. 从JSON文件加载参数 ---
    fprintf('Loading parameters from params.json...\n');
    project_path = 'c:\Users\a1831\Desktop\NBS_Project'; % 您的项目路径
    json_file_path = fullfile(project_path, 'params.json');
    json_string = fileread(json_file_path);
    params = jsondecode(json_string);
    
    % 将参数提取到MATLAB变量中
    N = length(params.mds);
    D0 = params.esp.D0;
    theta = params.esp.theta;
    o = params.esp.o;
    w0 = params.esp.omega_0;
    s = params.esp.s;
    l = params.esp.l;
    
    F = [params.mds.Fn]';
    c = [params.mds.cn]';
    k = [params.mds.kn]';
    R = [params.mds.Rn]';
    w = [params.mds.omega_n]';
    
    fprintf('Parameters loaded for N = %d MDs.\n', N);
    fprintf('========================================\n\n');
    
    % --- 2. 实验循环设置 ---
    lambda0_list = 50:10:150;
    results_sw = nan(1, length(lambda0_list));
    results_dmax = nan(1, length(lambda0_list));
    
    % --- 3. 遍历 lambda0 进行求解 ---
    for i = 1:length(lambda0_list)
        lambda0 = lambda0_list(i);
        fprintf('===== Solving for lambda0 = %d =====\n', lambda0);
        
        % ... (内部的所有代码，包括x0, lb, ub, Aeq, beq的定义都保持不变)
        eps = 1e-7;
        lb = zeros(3*N + 1, 1);
        ub = inf(3*N + 1, 1);
        lb(1:N) = eps;          ub(1:N) = lambda0;
        lb(N+1:2*N) = eps;      ub(N+1:2*N) = F - eps;
        lb(2*N+1:3*N) = 0;
        lb(3*N+1) = eps;        ub(3*N+1) = D0 - eps;
        Aeq = [ones(1, N), zeros(1, 2*N + 1)];
        beq = lambda0;
        
        x0 = zeros(3*N + 1, 1);
        lam0 = repmat(lambda0 / N, N, 1);
        p0 = ones(N,1);
        L_p0 = c .* p0.^2 + F.^k - (F - p0).^k;
        r0 = L_p0 + 0.1;
        Tx = s ./ R;
        processing_margin0 = p0./(s*l) - lam0;
        init_dns = Tx + 1./max(processing_margin0, eps);
        Dm0 = max(init_dns) + 0.1;
        while (lambda0*theta - o/(D0-Dm0)) <= sum(r0)
            Dm0 = (Dm0 + D0) / 2;
            if Dm0 >= D0 - eps, Dm0 = D0 - eps; break; end
        end
        x0(1:N) = lam0;
        x0(N+1:2*N) = p0;
        x0(2*N+1:3*N) = r0;
        x0(3*N+1) = Dm0;
        
        options = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
                               'Display', 'none', 'MaxFunctionEvaluations', 200000);
                           
        try
            [x_sol, ~, exitflag] = fmincon(@objective_fcn, x0, [], [], Aeq, beq, lb, ub, @nonlcon, options);
            
            if exitflag > 0
                p_opt = x_sol(N+1:2*N);
                Dmax_opt = x_sol(3*N+1);
                Q_val = lambda0 * theta - o / (D0 - Dmax_opt);
                L_val = c .* p_opt.^2 + F.^k - (F - p_opt).^k;
                sw = Q_val - sum(L_val);
                
                results_sw(i) = sw;
                results_dmax(i) = Dmax_opt;
                
                fprintf('  Success! Social Welfare = %.4f, Dmax = %.4f\n\n', sw, Dmax_opt);
            else
                fprintf('  Solver did not converge. Exitflag: %d\n\n', exitflag);
            end
            
        catch ME
            fprintf('  An error occurred during optimization: %s\n\n', ME.message);
        end
    end
    
    % --- 4. 显示最终结果 ---
    fprintf('========================================\n');
    fprintf('Final Results:\n');
    disp('Lambda0 values:');
    disp(lambda0_list);
    disp('Social Welfare results:');
    disp(results_sw);
    disp('Maximum Delay results:');
    disp(results_dmax);
    
    % --- 嵌套函数定义 (现在它们可以正确地访问外部变量) ---
    function f = objective_fcn(x)
        p = x(N+1:2*N);
        r = x(2*N+1:3*N);
        Dm = x(3*N+1);
        Q_Dm = lambda0 * theta - o / (D0 - Dm);
        L_p = c .* p.^2 + F.^k - (F - p).^k;
        esp_utility = Q_Dm - sum(r);
        md_utility = r - L_p;
        f = -(w0 * log(esp_utility) + sum(w .* log(md_utility)));
    end

    function [c_ineq, ceq] = nonlcon(x)
        lam = x(1:N);
        p = x(N+1:2*N);
        r = x(2*N+1:3*N);
        Dm = x(3*N+1);
        Q_Dm = lambda0 * theta - o / (D0 - Dm);
        L_p = c .* p.^2 + F.^k - (F - p).^k;
        Tx = s ./ R;
        processing_margin = p./(s*l) - lam;
        
        c_ineq = [
            lam - p./(s*l) + eps;
            1 - (Dm - Tx) .* processing_margin;
            eps - (Q_Dm - sum(r));
            eps - (r - L_p);
        ];
        ceq = [];
    end

end % <--- 【核心修改】函数的结束标志