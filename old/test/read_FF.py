import uncertainties
from uncertainties.umath import * # sin(), etc.
from uncertainties import ufloat
from uncertainties import unumpy, umath
import math
from scipy.optimize import minimize
from scipy import stats
import math
from autograd import hessian
import autograd.numpy as np
import glob
from matplotlib import pyplot as plt
import pandas as pd


result_files = []
for i in range(1):
    result_file = "/software/pc24403/tfpcbpggsz/test/Fitter.log"
    with open(result_file) as f:
        content = f.read()
    
        # Check if 'INVALID' is in the file content
        if not 'INVALID' in content:
            result_files.append(result_file)
        else:
            print(result_file)

print(f"Found {len(result_files)} valid results")



foundCKM = False
idx_CKM = 0
cov_string = ""
param_string = ""
log_strings = []
xp = []
name = []
FF={}



for result_file in result_files:
    do_next = True
    idx = 0
    idxs = []
    x = []
    dx = []
    cov = []
    name = []

    with open(result_file) as f: 
        for l in f:
            if "FitFraction" in l:
                a=l.split()
                name_i = a[1]
                name += [name_i]
                value = float(a[2])
                x += [value]
                error = float(a[3])
                dx += [error]
                idxs += [idx]
                idx +=1
        log_strings += [l.replace("\n", "")]

    FF = {name[i]: ufloat(x[i], dx[i]) for i in range(len(name))}
    
#Make a table from FF
df = pd.DataFrame(FF.items(), columns=['Parameter', 'Value'])
print(df)

#save to file
df.to_csv("FF.csv", index=False)
