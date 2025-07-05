# src/model.py
import numpy as np

def social_welfare(ESP,MDs,lam,p):
    L = [md.Ln(pn) for (pn,md) in zip(p,MDs)]
    Dmax = max([md.s/md.Rn + 1/(pn/(md.s*md.l)-lamn) for (md,pn,lamn) in zip(MDs,p,lam)])
    return ESP.Q(Dmax)-sum(L)

def nash_product_log(ESP,MDs,lam,p,r):
    Dmax = max([md.s/md.Rn + 1/(pn/(md.s*md.l)-lamn) for (md,pn,lamn) in zip(MDs,p,lam)])
    n0 = ESP.omega_0 * np.log(ESP.Q(Dmax)-np.sum(r))
    ns = [md.omega_n*np.log(rn-md.Ln(pn)) for (md,rn,pn) in zip(MDs,r,p)]
    return n0+np.sum(ns)

class MD:
    def __init__(self, param):
        self.param = param
        self.cn = param["cn"]  # 能耗系数
        self.Fn = param["Fn"]  # 算力总量 (GHz)
        self.kn = param["kn"]  # 算力敏感度
        self.omega_n = param["omega_n"]  # 议价能力
        self.Rn = param["Rn"]  # 传输速率 (MBps)
        self.s = param["s"]
        self.l = param["l"]
        self.pn = 0

    def transmission_delay(self):
        return self.s/self.Rn
    
    def compute_delay(self,pn,lambdan):
        if pn/(self.s*self.l)<lambdan: return None
        return 1/(pn/(self.s*self.l)-lambdan)
    
    def delay(self,pn,lambdan):
        if pn/(self.s*self.l)<lambdan: return None
        return self.s/self.Rn + 1/(pn/(self.s*self.l)-lambdan)
    
    def energy(self,pn):
        return self.cn*(pn**2)
    
    def bn(self,pn):
        if pn > self.Fn: return None
        return (self.Fn-pn)**self.kn
    
    def utility(self,pn,rn):
        return rn+self.bn(pn)-self.energy(pn)
    
    def Ln(self,pn):
        return self.energy(pn)+self.bn(0)-self.bn(pn)


class ESP:
    def __init__(self,param):
        self.param = param
        self.lambda0 = param["lambda0"]
        self.D0 = param["D0"]
        self.theta = param["theta"]
        self.o = param["o"]
        self.omega_0 = param["omega_0"]

    def Q(self,Dmax):
        if self.D0 < Dmax: return None
        return self.lambda0*self.theta - self.o/(self.D0-Dmax)
    
    def utility(self,Dmax,r):
        return self.Q(Dmax)-sum(r)
