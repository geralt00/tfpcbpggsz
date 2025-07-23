import numpy as np
import json
from tfpcbpggsz.ulti import get_xy_xi, deg_to_rad
from tfpcbpggsz.Includes.common_constants import *
import mplhep
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
mplhep.style.use("LHCb2")
from scipy.stats import norm
from scipy import stats
bounds = [(-1, 3), (0, 5)]


means_pulls  = {}
errors_pulls = {}
means_means  = {}
errors_means = {}
Ntoys  = []
date = "2025_06_24"

var_list = ["xplus", "xminus", "yplus", "yminus", "xxi", "yxi", "gamma", "dB", "rb"]

for var in var_list:
    means_pulls[var]  = []
    errors_pulls[var] = []
    means_means[var]  = []
    errors_means[var] = []
    pass



short_path = f"{date}/01_No_Efficiency_516MisID"
pull_dir = f"/shared/scratch/rj23972/safety_net/tfpcbpggsz/canorman_B2DPI_misID/{short_path}"
ntoy = 5000
plot_dir = short_path
os.makedirs(plot_dir, exist_ok = True)
os.makedirs(f"{plot_dir}/means" , exist_ok=True)
os.makedirs(f"{plot_dir}/pulls" , exist_ok=True)
os.makedirs(f"{plot_dir}/errors", exist_ok=True)

toy_results = {}
for var in var_list:
    toy_results[f"means_{var}"]  = []
    toy_results[f"errors_{var}"] = []
    pass

for i in range(ntoy):
    try:
        with open(f"{pull_dir}/study_{i}/means_results.json") as f:
            tmp = json.load(f)
            for key in var_list:
                toy_results[f"means_{key}"].append(tmp[key])
                pass
            pass
    except FileNotFoundError:
        continue
    with open(f"{pull_dir}/study_{i}/errors_results.json") as f:
        tmp = json.load(f)
        for key in var_list:
            toy_results[f"errors_{key}"].append(tmp[key])
            pass
        pass
    pass

toy_results  = pd.DataFrame.from_dict(toy_results)
if len(toy_results) == 0:
    print(" ERROR ----------- NO TOYS WERE FOUND ")
    exit()
Ntoys.append(len(toy_results))


inputs = {
    "gamma" : GAMMA, "rb" : RB_DK, "dB" : DELTAB_DK,
    "rb_dpi": RB_DPI, "dB_dpi": deg_to_rad(DELTAB_DPI)
}
inputs["xplus"], inputs["yplus"], inputs["xminus"], inputs["yminus"], inputs["xxi"], inputs["yxi"] = get_xy_xi(
    (deg_to_rad(inputs["gamma"]), inputs["rb"], deg_to_rad(inputs["dB"]), inputs["rb_dpi"], inputs["dB_dpi"])
)
# if (Efficiency_shape == "Legendre_2_2"):
#     inputs["c01_DK_Kspipi_DD_Bplus"] = 0.
#     pass

pulls = {}
for var in var_list:
    toy_results[f"pulls_{var}"] = (toy_results[f"means_{var}"] - inputs[var]) / toy_results[f"errors_{var}"]
    pass

toy_results.dropna()
toy_results = toy_results.query("(means_gamma > 50) & (means_gamma < 100)")
toy_results = toy_results.query("(errors_gamma > 2.5) & (errors_gamma < 10)")
toy_results = toy_results.query("(means_dB > 0) & (means_dB < 180)")
print("        Ntoys after dropna:       ", len(toy_results))
    
fig, ax = plt.subplots(3,3, figsize=(20,16))
row = 0
col = 0
for var in var_list:
    # print(var, row, col)
    tmp_ax = ax[row][col]
    ####### means
    mplhep.histplot(
        np.histogram(toy_results[f"means_{var}"], bins=50),
        ax=tmp_ax
    )
    tmp_ax.set_title(f"{var}")
    tmp_ax.vlines(inputs[var], 0, max(np.histogram(toy_results[f"means_{var}"], bins=50)[0]), color="black")
    tmp_ax.set_xlim([inputs[var]-5*np.mean(toy_results[f"errors_{var}"]),inputs[var]+5*np.mean(toy_results[f"errors_{var}"])])
    tmp_ax.set_yticks([])
    row = (row+int(col/2))%3
    col = (col+1)%3
    pass

plt.tight_layout()
plt.savefig(f"{plot_dir}/means/means.png")
plt.savefig(f"{plot_dir}/means/means.pdf")
plt.close("all")


fig, ax = plt.subplots(3,3, figsize=(20,16))
row = 0
col = 0
for var in var_list:
    # print(var, row, col)
    tmp_bounds = [(inputs[var]-5*np.mean(toy_results[f"errors_{var}"]),inputs[var]+5*np.mean(toy_results[f"errors_{var}"])), (0,np.abs(inputs[var]+5*np.mean(toy_results[f"errors_{var}"])))]
    tmp_ax = ax[row][col]
    res = stats.fit(norm, toy_results[f"means_{var}"], tmp_bounds)
    res.plot(ax=tmp_ax)
    tmp_ax.set_xlabel("")
    tmp_ax.set_xlim([inputs[var]-5*np.mean(toy_results[f"errors_{var}"]),inputs[var]+5*np.mean(toy_results[f"errors_{var}"])])
    tmp_ax.set_title(f"{var}")
    tmp_ax.set_ylabel("")
    tmp_ax.annotate(f"mean = {round(res.params[0],3)} \n sigma = {round(res.params[1],3)}", xy=(0.05, 0.85), xycoords='axes fraction')
    tmp_ax.legend([])#,[f"mean = {round(res.params[0],3)}", f"sigma = {round(res.params[1],3)}"],
                  # fontsize=10)
    tmp_ax.set_yticks([])
    # tmp_ax.tick_params(
    #     axis='y',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     labelbottom=False) # labels along the bottom edge are off
    row = (row+int(col/2))%3
    col = (col+1)%3
    means_means[var].append(res.params[0])
    errors_means[var].append(res.params[1])
    pass

plt.tight_layout()
plt.savefig(f"{plot_dir}/means/means_fit.png")
plt.savefig(f"{plot_dir}/means/means_fit.pdf")
plt.close("all")


fig, ax = plt.subplots(3,3, figsize=(20,16))
row = 0
col = 0
for var in var_list:
    # print(var, row, col)
    tmp_ax = ax[row][col]
    ####### errors
    mplhep.histplot(
        np.histogram(toy_results[f"errors_{var}"], bins=50),
        ax=tmp_ax
    )
    tmp_ax.set_title(f"{var}")
    tmp_ax.set_yticks([])
    row = (row+int(col/2))%3
    col = (col+1)%3
    pass

plt.tight_layout()
plt.savefig(f"{plot_dir}/errors/errors.png")
plt.savefig(f"{plot_dir}/errors/errors.pdf")
plt.close("all")

fig, ax = plt.subplots(3,3, figsize=(20,16))
row = 0
col = 0
for var in var_list:
    # print(var, row, col)
    tmp_ax = ax[row][col]
    ####### pulls
    mplhep.histplot(
        np.histogram(toy_results[f"pulls_{var}"],bins=50,range=[-4.,4.]),
        ax=tmp_ax
    )
    tmp_ax.set_xlim([-3.,3.])
    tmp_ax.set_title(f"{var}")
    tmp_ax.set_ylabel("")
    tmp_ax.set_yticks([])
    row = (row+int(col/2))%3
    col = (col+1)%3
    pass

plt.tight_layout()
plt.savefig(f"{plot_dir}/pulls/pulls.png")
plt.savefig(f"{plot_dir}/pulls/pulls.pdf")
plt.close("all")


fig, ax = plt.subplots(3,3, figsize=(20,16))
row = 0
col = 0
for var in var_list:
    # print(var, row, col)
    tmp_ax = ax[row][col]
    res = stats.fit(norm, toy_results[f"pulls_{var}"], bounds)
    res.plot(ax=tmp_ax)
    tmp_ax.set_xlabel("")
    tmp_ax.set_xlim([-5.,5.])
    tmp_ax.set_title(f"{var}")
    tmp_ax.set_ylabel("")
    tmp_ax.annotate(f"mean = {round(res.params[0],3)} \n sigma = {round(res.params[1],3)}", xy=(0.05, 0.85), xycoords='axes fraction')
    tmp_ax.legend([])#,[f"mean = {round(res.params[0],3)}", f"sigma = {round(res.params[1],3)}"],
                  # fontsize=10)
    tmp_ax.set_yticks([])
    # tmp_ax.tick_params(
    #     axis='y',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     labelbottom=False) # labels along the bottom edge are off
    row = (row+int(col/2))%3
    col = (col+1)%3
    means_pulls[var].append(res.params[0])
    errors_pulls[var].append(res.params[1])
    pass

plt.tight_layout()
plt.savefig(f"{plot_dir}/pulls/pulls_fit.png")
plt.savefig(f"{plot_dir}/pulls/pulls_fit.pdf")
plt.close("all")


