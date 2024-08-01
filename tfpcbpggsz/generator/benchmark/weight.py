import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import numpy as np
import tensorflow as tf
import uproot as up

from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd
from plothist import make_hist
from matplotlib import pyplot as plt
import time
import random   


time1 = time.time()
#Load momenta from cache
PHSP = np.load("DalitzEvent.npy",allow_pickle=True).item()
time2 = time.time()
print("Load Time: ",time2-time1)    
time3 = time.time()
print("Calculate Mass Time: ",time3-time2)



def accept_reject(arr, weights):
    n = len(arr)
    max_weight = np.max(weights)
    print(max_weight)
    x = []
    random.seed(int(time.time()) % 10)
    np_seed = random.randint(0, 1e6 - 10)
    np.random.seed(np_seed)

    for i in range(n):
        u2 = np.random.uniform(0, 1)
        if u2 * max_weight <= weights[i]:
            x.append(arr[i])
    return np.array(x)

def accept_reject_single(chunk, weights, np_seed):
    """Performs accept-reject sampling for a single chunk."""
    np.random.seed(np_seed)
    max_weight = np.max(weights)
    x = []
    for i in range(len(chunk)):
        u2 = np.random.uniform(0, 1)
        if u2 * max_weight <= weights[i]:
            x.append(chunk[i])
    return np.array(x)


def check_plot(arr, arr_ar, decay="b2dpi_DD"):

    projs = ['mkp', 'mkm']
    x_range = (0.3, 3.2)

    h_sample = {}
    h_total = {}
    for proj in projs:
        h_sample[proj] = make_hist(arr_ar[proj], bins=50, range=x_range)
        h_total[proj] = make_hist(arr[proj], bins=50, range=x_range)

    from plothist import plot_two_hist_comparison


    fig1, ax_main1, ax_comparison1 = plot_two_hist_comparison(h_sample['mkp'], h_total['mkp'], xlabel='mkp',ylabel="Entries",
    h1_label="$\mathit{H}_{Sample}$",
    h2_label="$\mathit{H}_{Total}$",
    comparison="efficiency"
        # <--)
    )
    fig1.savefig(f'{decay}_mkp.png')

    

    fig2, ax_main2, ax_comparison2 = plot_two_hist_comparison(h_sample['mkm'], h_total['mkm'],  xlabel='mkm',ylabel="Entries",
    h1_label="$\mathit{H}_{Sample}$",
    h2_label="$\mathit{H}_{Total}$",
    comparison="efficiency",  # <--)
    )

    fig2.savefig(f'{decay}_mkm.png')





def eff_fun(x, charge='p', decay='dk_LL'):
    # in GeV !!
    res = 0
    zp_p = x[:,0]  # $z_{+}^{\prime}
    zm_pp = x[:,1]  # $z_{-}^{\prime\prime}
    Legendre_zp_1 = zp_p
    Legendre_zp_2 = (3 * np.power(zp_p, 2) - 1) / 2.
    Legendre_zp_3 = (5 * np.power(zp_p, 3) - 3 * zp_p) / 2.
    Legendre_zp_4 = (35 * np.power(zp_p, 4) - 30 * np.power(zp_p, 2) + 3) / 8.
    Legendre_zp_5 = (36 * np.power(zp_p, 5) - 70 * np.power(zp_p, 3) + 15 * zp_p) / 8.
    Legendre_zp_6 = (231 * np.power(zp_p, 6) - 315 * np.power(zp_p, 4) + 105 * np.power(zp_p, 2) - 5) / 16.
    Legendre_zm_2 = (3 * np.power(zm_pp, 2) - 1) / 2.
    Legendre_zm_4 = (35 * np.power(zm_pp, 4) - 30 * np.power(zm_pp, 2) + 3) / 8.
    Legendre_zm_6 = (231 * np.power(zm_pp, 6) - 315 * np.power(zm_pp, 4) + 105 * np.power(zm_pp, 2) - 5) / 16.
    params = {}
    offset = {}
    mean = {}
    params['bp_b2dk_LL'] = [-9.53264, 30.6442, -70.1242, -171.659, 20.997, 111.455, -25.7043, -208.538, -131.828, 5.30961, 38.3281, 52.9287, -45.1955, -92.5608, -53.7144, -6.41475]
    params['bm_b2dk_LL'] = [-11.6916, 36.6738, -75.5949, -177.263, 24.6625, 128.041, -29.1374, -222.949, -133.375, 5.42292, 50.5985, 63.6018, -44.7697, -94.6231, -65.164, -7.16556]
    params['bp_b2dpi_LL'] = [-11.726, 36.3723, -79.9267, -178.024, 26.746, 125.069, -31.2218, -233.282, -133.757, 6.55028, 48.6604, 59.0497, -43.2816, -99.2002, -65.8039, -6.45939]
    params['bm_b2dpi_LL'] = [-11.2904, 33.8601, -76.9433, -178.023, 25.2621, 118.649, -30.6079, -226.689, -135.131, 5.66233, 46.1495, 57.1009, -45.8845, -98.3241, -65.0337, -6.8956]
    params['bp_b2dk_DD'] = [-30.8217, 98.3102, -201.671, -450.003, 67.5668, 336.798, -77.9681, -580.061, -337.139, 17.1748, 135.283, 166.737, -108.95, -248.926, -160.071, -16.7914]
    params['bm_b2dk_DD'] = [-28.5793, 92.1537, -192.655, -445.199, 66.4849, 315.014, -76.3822, -547.47, -336.686, 17.0215, 130.277, 144.332, -112.366, -224.548, -150.207, -19.5028]
    params['bp_b2dpi_DD'] = [-25.1893, 83.6515, -191.42, -437.186, 59.6098, 286.787, -79.8558, -540.406, -331.55, 13.7906, 114.149, 135.039, -114.09, -219.42, -167.023, -22.0707 ]
    params['bm_b2dpi_DD'] = [-27.3281, 77.6543, -181.93, -451.338, 51.6441, 278.849, -66.5276, -522.833, -353.474, 11.865, 98.5612, 138.88, -121.508, -228.435, -139.317, -15.4476 ]
    offset['bp_b2dk_LL'] = 57.060094380400386
    offset['bm_b2dk_LL'] = 58.17587468590285
    offset['bp_b2dk_DD'] = 144.4818375757835
    offset['bm_b2dk_DD'] = 149.4970954503175
    offset['bp_b2dpi_LL'] = 58.02066608312892
    offset['bm_b2dpi_LL'] = 57.2679149533671
    offset['bp_b2dpi_DD'] = 143.85275330726384
    offset['bm_b2dpi_DD'] = 145.64578485313336
    mean['bp_b2dk_LL'] = 94.71806024672841
    mean['bm_b2dk_LL'] = 95.67192393339243
    mean['bp_b2dk_DD'] = 236.7559605125362
    mean['bm_b2dk_DD'] = 243.96305838750337
    mean['bp_b2dpi_LL'] = 95.5957545077163
    mean['bm_b2dpi_LL'] = 95.47377527333916
    mean['bp_b2dpi_DD'] = 238.0281142413688
    mean['bm_b2dpi_DD'] = 241.6462132232453

    decay = 'b'+charge+'_'+decay

    res = (
        params[decay][0]
        + params[decay][1] * Legendre_zp_1
        + params[decay][2] * Legendre_zp_2
        + params[decay][3] * Legendre_zm_2
        + params[decay][4] * Legendre_zp_3
        + params[decay][5] * Legendre_zp_1 * Legendre_zm_2
        + params[decay][6] * Legendre_zp_4
        + params[decay][7] * Legendre_zp_2 * Legendre_zm_2
        + params[decay][8] * Legendre_zm_4
        + params[decay][9] * Legendre_zp_5
        + params[decay][10] * Legendre_zm_2 * Legendre_zp_3
        + params[decay][11] * Legendre_zm_4 * Legendre_zp_1
        + params[decay][12] * Legendre_zm_6
        + params[decay][13] * Legendre_zm_4 * Legendre_zp_2
        + params[decay][14] * Legendre_zm_2 * Legendre_zp_4
        + params[decay][15] * Legendre_zp_6
    )

    return( res+offset[decay])/mean[decay]



def apply_efficiency_mask(arr, weights):
    """Applies a boolean mask while preserving the original array shape."""
    max_weight = np.max(weights)
    random.seed(int(time.time()) % 10)
    np_seed = random.randint(0, 100000)
    np.random.seed(np_seed)
    threshold = np.random.uniform(0, max_weight, len(weights))

    mask = weights >= threshold
    
    print(f'Efficiency: {mask.sum() / len(mask)}')

    filtered_data = {name: arr[mask] for name, arr in arr.items()}
    return filtered_data




time_start = time.time()
for decay in ['b2dk_LL']:
    for charge in ['m']:
        print(f'Processing {decay} {charge}')
        weights = eff_fun(phsp_to_srd(PHSP['mkp'], PHSP['mkm']), charge, decay)
        time4 = time.time()
        print(f'Efficiency calculation time: {time4 - time3}')
        PHSP_eff = apply_efficiency_mask(PHSP, weights)
        time5 = time.time()
        print(f'Accept-reject time: {time5 - time4}')
        # Save the data
        np.save(f'{decay}_{charge}.npy', PHSP_eff)
        time6 = time.time()
        print(f'Save time: {time6 - time5}')
        # Check the results
        check_plot(PHSP, PHSP_eff, decay=decay)
        time7 = time.time()
        print(f'Plot time: {time7 - time6}')

time_end = time.time()
print(f'Total time: {time_end - time_start}')

