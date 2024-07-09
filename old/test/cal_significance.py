import uncertainties
from uncertainties.umath import * # sin(), etc.
from uncertainties import ufloat
from uncertainties import unumpy, umath
import math
from autograd import hessian
import autograd.numpy as np
import glob
import ROOT
import pandas as pd


def significance(chi2, ndf):
    # Calculate the p-value for the chi-square and ndf
    chi2 = abs(chi2)
    ndf = int(ndf)
    p_value = ROOT.TMath.Prob(chi2, ndf)
    
    # Convert p-value to sigma
    if p_value < 1e-15:  # Handle extremely small p-values
        return float('inf')  # Return infinity if p-value is extremely small
    sigma = ROOT.TMath.Sqrt(2) * ROOT.TMath.ErfInverse(1 - 2 * p_value)
    
    return sigma

foundCKM = False
idx_CKM = 0
cov_string = ""
param_string = ""
log_strings = []
xp = []
name = []
res_fitted={}
baseline_fitted={}

#The fild put your baseline model
ori_file = "/software/pc24403/tfpcbpggsz/test/Fitter.log"
with open(ori_file) as f:
    name_i =[]
    ndf = []
    nll = []
    for l in f:
        if "FitQuality" in l:
            a=l.split()
            name = a[0]
            ndf = float(a[3])
            nll = float(a[4])


baseline_fitted = {'baseline': ufloat(nll, ndf)}


solutions = ['K892']

log_path = "/software/pc24403/tfpcbpggsz/test/"

idx = 0
idxs = []
dx = []
x = []
cov = []
name = []

for res_rm in solutions:
    with open(log_path+res_rm+'.log') as f: 
        for l in f:
            if "FitQuality" in l:
                a=l.split()
                name_i = res_rm
                name += [name_i]
                ndf = float(a[3])
                dx += [ndf]
                nll = float(a[4])
                x += [nll]
                idxs += [idx]
                idx +=1
        log_strings += [l.replace("\n", "")]


res_fitted = {name[i]: ufloat(x[i], dx[i]) for i in range(len(name))}
    

for res_rm in solutions:
    print(res_rm)
    S = {res_rm: significance(baseline_fitted['baseline'].n - res_fitted[res_rm].n, baseline_fitted['baseline'].s - res_fitted[res_rm].s)}

#Make a table from FF
df = pd.DataFrame(S.items(), columns=['Resonance', 'Significance (sigma)'])
#print(df)
#print(df.to_latex(index=False))
#save to file
df.to_csv("significance.csv", index=False)

