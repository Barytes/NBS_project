# src/baseline.py()
import numpy as np

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