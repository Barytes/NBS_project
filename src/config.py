# src/config.py
import json
import os
import numpy as np

class Config:
    def __init__(self, config_file=None):
        if config_file:
            self.load_from_file(config_file)
        else:
            self.set_default()

    def set_default(self):
      """
      设置默认参数（用于测试或调试）
      """
      self.num_users = 10
      self.d_range = [5e1, 2e2]
      self.b_range = [1e1, 5e2]
      self.alpha_range = [0.5, 2.0]
      self.cpu_range = [0.00005, 0.0005]

      # 定义多个实验的资源 + 价格组
      self.param_groups = [
          {"f_max": 1e7, "B_max": 1e4, "c_E": 1e-3, "c_N": 1e-3}
      ]  

    def generate_random_task_param(self,rng):
        """
        生成一个随机的任务参数，包括 d（计算量）、b（数据量）、alpha（延迟敏感度）
        """
        task = {
            "d": rng.uniform(*self.d_range),  # 任务计算量 (CPU cycles)
            "b": rng.uniform(*self.b_range),  # 任务数据量 (Bytes)
            "alpha": rng.uniform(*self.alpha_range),  # 延迟敏感度
        }
        return task

    def generate_random_channel_param(self,rng,real=False):
        """
        生成不同用户的任务参数，每个用户都有不同的 d（计算量）、b（数据量）、alpha（延迟敏感度）
        """
        h = rng.exponential(scale=1.0) # small-scale fading channel power gain
        g0_dB=-40 # path-loss constant
        d0=1 # reference distance (m)
        theta=4 # path-loss exponent
        distance=10 # user to base station distance (m)
        l0_linear = 10 ** (g0_dB / 10)
        Noise_linear_W = 10**(-174/10)
        if real:
          channel = {
              "trans_power": rng.uniform(0.5, 2),  # 传输功率 (W)
              "channel_gain": h * l0_linear * (d0 / distance)**theta,  # 信道增益
              "background_noise": Noise_linear_W,  # 背景噪声 (dBm
          }
        else:
          channel = {
              "trans_power": 1,  # 传输功率 (W)
              "channel_gain": 1,  # 信道增益
              "background_noise": 1,  # 背景噪声 (dBm
          }
        return channel

    def generate_random_local_compute_power(self,rng):
        local_compute = rng.uniform(*self.cpu_range) # 0.5GHz-1.2GHz
        return local_compute  # 每个用户的本地 CPU 计算能力

    def load_from_file(self, config_file):
        with open(config_file, 'r') as f:
          data = json.load(f)

        self.num_users = data["num_users"]
        self.d_range = data["d_range"]
        self.b_range = data["b_range"]
        self.alpha_range = data["alpha_range"]
        self.cpu_range = data["cpu_range"]
        self.param_groups = data["param_groups"]