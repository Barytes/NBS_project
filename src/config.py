# src/config.py
import json
import os
import numpy as np

class Config:
    def __init__(self, config_file):
        self.load_from_file(config_file) 

    def generate_md_param(self,rng):
        """
        生成一个随机的MD参数
        """
        md = {
            "cn": rng.uniform(*self.cn_range),  # 能耗系数
            "Fn": rng.uniform(*self.Fn_range),  # 算力总量 (GHz)
            "kn": rng.uniform(*self.kn_range),  # 算力敏感度
            "omega_n": self.omega_n,  # 议价能力
            "Rn": self.generate_transmission_rate(rng),  # 传输速率 (MBps)
        } 
        return md

    def generate_transmission_rate(self,rng):
        """
        生成一个随机的传输速率
        """
        B = 1e7 # 10 MHz
        h = rng.exponential(scale=1.0) # small-scale fading channel power gain
        g0_dB=-40 # path-loss constant
        d0=1 # reference distance (m)
        varpi=4 # path-loss exponent
        distance=rng.uniform(100, 150) # user to base station distance (m)
        l0_linear = 10 ** (g0_dB / 10)
        channel_gain = h * l0_linear * (d0 / distance) ** varpi
        power = 10
        N0_dBm_per_Hz = -174
        N0_W_per_Hz   = 10 ** ((N0_dBm_per_Hz - 30) / 10)
        rate = B * np.log2(1 + power * channel_gain / (N0_W_per_Hz * B))
        rate = rate / 8e6  # 转换为 MBps
        return rate

    def load_from_file(self, config_file):
        with open(config_file, 'r') as f:
          data = json.load(f)
        
        self.num_mds = data["num_mds"]
        self.lambda0 = data["lambda0"]
        self.s = data["s"]
        self.l = data["l"]
        self.D0 = data["D0"]
        self.cn_range = data["cn_range"]
        self.kn_range = data["kn_range"]
        self.Fn_range = data["Fn_range"]
        self.theta = data["theta"]
        self.o = data["o"]
        self.omega_0 = data["omega_0"]
        self.omega_n = data["omega_n"]


class ParamSweeper:
    def __init__(self, sweep_config_path):
        with open(sweep_config_path, 'r') as f:
            sweep_settings = json.load(f)
        
        self.vary_param = sweep_settings["vary_param"]
        self.values = sweep_settings["values"]
        self.fixed_config_file = sweep_settings["fixed_config_file"]

        with open(self.fixed_config_file, 'r') as f:
            self.fixed_config_data = json.load(f)

    def sweep_configs(self):
        configs = []
        for val in self.values:
            config_data = copy.deepcopy(self.fixed_config_data)
            if self.vary_param in ["f_max", "B_max", "c_E", "c_N"]:
                config_data["param_groups"][0][self.vary_param] = val
            elif self.vary_param == "num_users":
                config_data["num_users"] = val
            else:
                raise ValueError(f"暂不支持的 sweep 参数: {self.vary_param}")
            
            # 构造 Config 实例
            cfg = Config()
            cfg.__dict__.update(config_data)
            configs.append(cfg)
        return configs

    def run_param_sweep(self, solver_fn, result_dir="./results/"):
        os.makedirs(result_dir, exist_ok=True)
        results = []

        for i, cfg in enumerate(self.sweep_configs()):
            users = create_users(cfg)
            provider = create_provider(cfg)
            
            try:
                X_star, U_best, _ = solver_fn(users, provider)
                results.append({
                    "param_value": self.values[i],
                    "U_best": U_best,
                    "offload_num": len(X_star),
                    "f_max": provider.f_max,
                    "B_max": provider.B_max,
                    "c_E": provider.c_E,
                    "c_N": provider.c_N,
                    "num_users": len(users)
                })
            except Exception as e:
                print(f"Run {i} failed:", e)
                continue

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(result_dir, f"sweep_{self.vary_param}.csv"), index=False)
        return df
