# src/config.py
import json
import os
import numpy as np

class Config:
    def __init__(self, config_file):
        self.load_from_file(config_file) 

    def generate_user_param(self,rng):
        """
        生成一个随机的用户参数
        """
        user = {
            "cn": rng.uniform(*self.cn_range),  # 任务计算量 (CPU cycles)
            "Fn": rng.uniform(*self.Fn_range),  # 任务数据量 (Bytes)
            "kn": rng.uniform(*self.kn_range),  # 延迟敏感度
            "omega_n": self.omega_n,  # 用户的带宽
            "Rn": self.generate_transmission_rate(rng),  # 传输速率
        }
        return user

    def generate_transmission_rate(self,rng):
        """
        生成一个随机的传输速率
        """
        B = 10 # 10 MHz
        h = rng.exponential(scale=1.0) # small-scale fading channel power gain
        g0_dB=-40 # path-loss constant
        d0=1 # reference distance (m)
        varpi=4 # path-loss exponent
        distance=rng.uniform(100, 150) # user to base station distance (m)
        l0_linear = 10 ** (g0_dB / 10)
        channel_gain = h * l0_linear * (d0 / distance) ** varpi
        power = 10
        Noise = -174  # noise power spectral density in dBm/Hz
        rate = B * np.log2(1 + power * channel_gain / (Noise * B))
        return rate

    def load_from_file(self, config_file):
        with open(config_file, 'r') as f:
          data = json.load(f)
        
        self.num_users = data["num_users"]
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
