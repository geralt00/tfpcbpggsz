import argparse
parser = argparse.ArgumentParser(description='Signal only Fit')
parser.add_argument('--index', type=int, default=1, help='Index of the toy MC')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

parser.print_help()
args = parser.parse_args()
index=args.index

import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

from importlib.machinery import SourceFileLoader
import uproot as up
import numpy as np
import time
import iminuit
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
tf.get_logger().setLevel('INFO')

time1 = time.time()

sys.path.append('/software/pc24403/tfpcbpggsz/func')
sys.path.append('/software/pc24403/tfpcbpggsz/amp_ampgen')
sys.path.append('/software/pc24403/tfpcbpggsz/core')
from amp import *
from core import *
from tfmassshape import *


mc_path = '/shared/scratch/pc24403/amp_ampgen'

def get_p4(decay="b2dpi", cut='', index=2):

    file_name = ''
    branch_names = []
    if cut == 'int':
        file_name = f'/software/pc24403/PCBPGGSZ/Int/weighted_{decay}.root:DalitzEventList'
        branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
    
    elif decay.split('_')[0] == 'b2dk' or decay.split('_')[0] == 'b2dpi':
        branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
        if cut == 'p':
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/add_sw/new_frac_sw_pg/lhcb_toy_{decay}_{index}_CPrange.root:Bplus_DalitzEventList'

        else:
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/add_sw/new_frac_sw_pg/lhcb_toy_{decay}_{index}_CPrange.root:Bminus_DalitzEventList'

    tree = up.open(file_name)
  # Load the branches as arrays
    
    array = tree.arrays(branch_names)
       

    _p1 = np.asarray([array["_1_K0S0_E"], array["_1_K0S0_Px"], array["_1_K0S0_Py"], array["_1_K0S0_Pz"]])
    _p2 = np.asarray([array["_2_pi#_E"], array["_2_pi#_Px"], array["_2_pi#_Py"], array["_2_pi#_Pz"]])
    _p3 = np.asarray([array["_3_pi~_E"], array["_3_pi~_Px"], array["_3_pi~_Py"], array["_3_pi~_Pz"]])
    
    # convert 4*1000 into a vectot<double>
    p1 = np.transpose(_p1)
    p2 = np.transpose(_p2)
    p3 = np.transpose(_p3)

    p1bar = np.hstack((p1[:, :1], np.negative(p1[:, 1:])))
    p2bar = np.hstack((p2[:, :1], np.negative(p2[:, 1:])))
    p3bar = np.hstack((p3[:, :1], np.negative(p3[:, 1:])))



    return p1, p2, p3, p1bar, p2bar, p3bar


def getAmp(decay='b2dpi', cut='int',index=2):

    start_time = time.time()
    p1, p2, p3, p1bar, p2bar, p3bar = get_p4(decay=decay, cut=cut, index=index)
    amplitude = []
    amplitudeBar = []

    p1_np = np.array(p1)
    p2_np = np.array(p2)
    p3_np = np.array(p3)
    p1bar_np = np.array(p1bar)
    p2bar_np = np.array(p2bar)
    p3bar_np = np.array(p3bar)

    data = [(p1_np[i], p2_np[i], p3_np[i]) for i in range(len(p1_np))]
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        amplitude.append(pool.map(load_int_amp, data))
    data_bar = [(p1bar_np[i], p3bar_np[i], p2bar_np[i]) for i in range(len(p1bar_np))]
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        amplitudeBar.append(pool.map(load_int_amp, data_bar))
    
    end_time = time.time()
    print(f'Amplitude for {decay} loaded in {end_time-start_time} seconds')
    amplitude = np.array(amplitude)
    amplitudeBar = np.array(amplitudeBar)

    return amplitude, amplitudeBar
    
def get_p4_v2(decay="b2dpi", cut='', index=2, comp='sig'):

    file_name = ''
    branch_names = []
    if cut == 'int':
        file_name = f'/software/pc24403/PCBPGGSZ/Int/weighted_{decay}.root:DalitzEventList'
        branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
    
    elif decay.split('_')[0] == 'b2dk' or decay.split('_')[0] == 'b2dpi':
        branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz", "B_M"]
        if cut == 'p':
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/add_sw/new_frac_sw_pg/{decay}_{index}_CPrange.root:DalitzEventList'
#            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/swap/{decay}_{comp}_{index}.root:Bplus_DalitzEventList'

        else:
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/add_sw/new_frac_sw_pg/{decay}_{index}_CPrange.root:DalitzEventList'
#            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/swap/{decay}_{comp}_{index}.root:Bminus_DalitzEventList'

    tree = up.open(file_name)
  # Load the branches as arrays
    charge = '(Bac_ID>0)'
    if cut == 'm':
        charge = '(Bac_ID<0)'
    
    array = tree.arrays(branch_names, charge)
       

    _p1 = np.asarray([array["_1_K0S0_E"], array["_1_K0S0_Px"], array["_1_K0S0_Py"], array["_1_K0S0_Pz"]])
    _p2 = np.asarray([array["_2_pi#_E"], array["_2_pi#_Px"], array["_2_pi#_Py"], array["_2_pi#_Pz"]])
    _p3 = np.asarray([array["_3_pi~_E"], array["_3_pi~_Px"], array["_3_pi~_Py"], array["_3_pi~_Pz"]])
    
    # convert 4*1000 into a vectot<double>
    p1 = np.transpose(_p1)
    p2 = np.transpose(_p2)
    p3 = np.transpose(_p3)

    p1bar = np.hstack((p1[:, :1], np.negative(p1[:, 1:])))
    p2bar = np.hstack((p2[:, :1], np.negative(p2[:, 1:])))
    p3bar = np.hstack((p3[:, :1], np.negative(p3[:, 1:])))

    B_M = np.zeros(len(p1))
    if cut != 'int':
        
        B_M = np.asarray([array["B_M"]])


    return p1, p2, p3, p1bar, p2bar, p3bar, B_M

def getMass_v2(decay='b2dpi', cut='int', index = 2, comp='sig'):

    start_time = time.time()
    p1, p2, p3, p1bar, p2bar, p3bar, B_M = get_p4_v2(decay=decay, cut=cut, comp=comp, index=index)
    amplitude = []
    amplitudeBar = []

    p1_np = np.array(p1)
    p2_np = np.array(p2)
    p3_np = np.array(p3)
    p1bar_np = np.array(p1bar)
    p2bar_np = np.array(p2bar)
    p3bar_np = np.array(p3bar)

    s12 = get_mass(p1_np, p2_np)
    s13 = get_mass(p1_np, p3_np)

    return s12, s13, B_M





config_mass_shape_output = SourceFileLoader('config_mass_shape_output', '/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/config/lhcb/%s'%(f'config_cpfit_output_{index}.py')).load_module()
varDict = config_mass_shape_output.getconfig()

sig_yield = {}
Bdecays = ['b2dk', 'b2dpi']
Types = ['LL', 'DD']
for bdecay in Bdecays:
    for Type in Types:
        decay = ''

        if bdecay == 'b2dk':
            decay = 'DK_KsPiPi_%s'%Type
        elif bdecay == 'b2dpi':
            decay = 'DPi_KsPiPi_%s'%Type

        #print yields
        print('Yields:')
        print('Sig: %.2f'%varDict['n_sig_%s'%decay])
        print('MisID: %.2f'%varDict['n_misid_%s'%decay])
        print('Low: %.2f'%varDict['n_low_%s'%decay])
        print('Comb: %.2f'%varDict['n_comb_%s'%decay])
        if bdecay == 'b2dk':
            print('Low MisID: %.2f'%varDict['n_low_misID_%s'%decay])
            print('Low Bs2DKPi: %.2f'%varDict['n_low_Bs2DKPi_%s'%decay])
            sum_yields = varDict['n_sig_%s'%decay] + varDict['n_misid_%s'%decay] + varDict['n_low_%s'%decay] + varDict['n_comb_%s'%decay] + varDict['n_low_misID_%s'%decay] + varDict['n_low_Bs2DKPi_%s'%decay]
        else:
            sum_yields = varDict['n_sig_%s'%decay] + varDict['n_misid_%s'%decay] + varDict['n_low_%s'%decay] + varDict['n_comb_%s'%decay]
        print('Sum: %.2f'%sum_yields)



logpath = '/dice/users/pc24403/BPGGSZ/sim_fit'
if os.path.exists(logpath) == False:
    os.mkdir(logpath)


print('INFO: Initialize the amplitudes...')
amp_Data_dk_dd_m = []
ampbar_Data_dk_dd_m = []
amp_Data_dk_dd_p = []
ampbar_Data_dk_dd_p = []
amp_Data_dk_ll_m = []
ampbar_Data_dk_ll_m = []
amp_Data_dk_ll_p = []
ampbar_Data_dk_ll_p = []
amp_dk_ll_p = []
amp_dk_ll_m = []
amp_dk_dd_p = []
amp_dk_dd_m = []
ampbar_dk_ll_p = []
ampbar_dk_ll_m = []
ampbar_dk_dd_p = []
ampbar_dk_dd_m = []

amp_Data_dpi_dd_m = []
ampbar_Data_dpi_dd_m = []
amp_Data_dpi_dd_p = []
ampbar_Data_dpi_dd_p = []
amp_Data_dpi_ll_m = []
ampbar_Data_dpi_ll_m = []
amp_Data_dpi_ll_p = []
ampbar_Data_dpi_ll_p = []
amp_dpi_ll_p = []
amp_dpi_ll_m = []
amp_dpi_dd_p = []
amp_dpi_dd_m = []
ampbar_dpi_ll_p = []
ampbar_dpi_ll_m = []
ampbar_dpi_dd_p = []
ampbar_dpi_dd_m = []


print('INFO: Loading amplitudes...')

amp_Data_dk_dd_p, ampbar_Data_dk_dd_p = getAmp('b2dk_DD', 'p', index=index)
amp_Data_dk_dd_m, ampbar_Data_dk_dd_m = getAmp('b2dk_DD', 'm', index=index)
amp_Data_dk_ll_p, ampbar_Data_dk_ll_p = getAmp('b2dk_LL', 'p', index=index)
amp_Data_dk_ll_m, ampbar_Data_dk_ll_m = getAmp('b2dk_LL', 'm', index=index)
amp_dk_dd_p = np.load(mc_path + '/Int_b2dk_DD_p_amp.npy')
ampbar_dk_dd_p = np.load(mc_path + '/Int_b2dk_DD_p_ampbar.npy')
amp_dk_dd_m = np.load(mc_path + '/Int_b2dk_DD_m_amp.npy')
ampbar_dk_dd_m = np.load(mc_path + '/Int_b2dk_DD_m_ampbar.npy')
amp_dk_ll_p = np.load(mc_path + '/Int_b2dk_LL_p_amp.npy')
ampbar_dk_ll_p = np.load(mc_path + '/Int_b2dk_LL_p_ampbar.npy')
amp_dk_ll_m = np.load(mc_path + '/Int_b2dk_LL_m_amp.npy')
ampbar_dk_ll_m = np.load(mc_path + '/Int_b2dk_LL_m_ampbar.npy')

amp_Data_dpi_dd_p, ampbar_Data_dpi_dd_p = getAmp('b2dpi_DD', 'p', index=index)
amp_Data_dpi_dd_m, ampbar_Data_dpi_dd_m = getAmp('b2dpi_DD', 'm', index=index)
amp_Data_dpi_ll_p, ampbar_Data_dpi_ll_p = getAmp('b2dpi_LL', 'p', index=index)
amp_Data_dpi_ll_m, ampbar_Data_dpi_ll_m = getAmp('b2dpi_LL', 'm', index=index)
amp_dpi_dd_p = np.load(mc_path + '/Int_b2dpi_DD_p_amp.npy')
ampbar_dpi_dd_p = np.load(mc_path + '/Int_b2dpi_DD_p_ampbar.npy')
amp_dpi_dd_m = np.load(mc_path + '/Int_b2dpi_DD_m_amp.npy')
ampbar_dpi_dd_m = np.load(mc_path + '/Int_b2dpi_DD_m_ampbar.npy')
amp_dpi_ll_p = np.load(mc_path + '/Int_b2dpi_LL_p_amp.npy')
ampbar_dpi_ll_p = np.load(mc_path + '/Int_b2dpi_LL_p_ampbar.npy')
amp_dpi_ll_m = np.load(mc_path + '/Int_b2dpi_LL_m_amp.npy')
ampbar_dpi_ll_m = np.load(mc_path + '/Int_b2dpi_LL_m_ampbar.npy')



pdfs_data = {}
s12_data = {}
s13_data = {}
Bu_M = {}
for decay in ['b2dk_LL', 'b2dk_DD', 'b2dpi_LL', 'b2dpi_DD']:
    for charge in ['p', 'm']:
        new_decay = decay + '_'+ charge
        print('--- INFO: Preparing pdfs for %s...'%new_decay)
        s12_data[new_decay], s13_data[new_decay], Bu_M[new_decay] = getMass_v2(decay, charge, index)
        pdfs_data[new_decay] = preparePdf_data(Bu_M[new_decay], varDict, decay)
        Bu_M[new_decay] = tf.constant(Bu_M[new_decay], tf.float64)

@tf.function
@tf.autograph.experimental.do_not_convert
def prod_nll_dk_dd(x):

    decay = 'b2dk_DD'
    fracDD = 0.33
    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_dd_p, ampbar_dk_dd_p)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_dd_m, ampbar_dk_dd_m)
    misid_normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_dd_p, ampbar_dpi_dd_p)
    misid_normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_dd_m, ampbar_dpi_dd_m)

    normA_p = tf.math.reduce_mean(tf.math.abs(amp_dk_dd_p)**2)
    normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dk_dd_p)**2)
    normA_m = tf.math.reduce_mean(tf.math.abs(amp_dk_dd_m)**2)
    normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dk_dd_m)**2)

    misid_normA_p = tf.math.reduce_mean(tf.math.abs(amp_dpi_dd_p)**2)
    misid_normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_dd_p)**2)
    misid_normA_m = tf.math.reduce_mean(tf.math.abs(amp_dpi_dd_m)**2)
    misid_normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_dd_m)**2)

    sig_prob_p = prod_totalAmplitudeSquared_XY(1, amp_Data_dk_dd_p, ampbar_Data_dk_dd_p, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    sig_prob_m = prod_totalAmplitudeSquared_XY(-1, amp_Data_dk_dd_m, ampbar_Data_dk_dd_m, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    misid_prob_p = prod_totalAmplitudeSquared_DPi_XY(1, amp_Data_dk_dd_p, ampbar_Data_dk_dd_p, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    misid_prob_m = prod_totalAmplitudeSquared_DPi_XY(-1, amp_Data_dk_dd_m, ampbar_Data_dk_dd_m, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    comb_prob_p = prod_comb(amp_Data_dk_dd_p, ampbar_Data_dk_dd_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], fracDD=fracDD)
    comb_prob_m = prod_comb(amp_Data_dk_dd_m, ampbar_Data_dk_dd_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], fracDD=fracDD)
    low_prob_p = prod_low(1, amp_Data_dk_dd_p, ampbar_Data_dk_dd_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low')
    low_prob_m = prod_low(-1, amp_Data_dk_dd_m, ampbar_Data_dk_dd_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low')
    low_misID_prob_p = prod_low(1, amp_Data_dk_dd_p, ampbar_Data_dk_dd_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low_misID')
    low_misID_prob_m = prod_low(-1, amp_Data_dk_dd_m, ampbar_Data_dk_dd_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low_misID')
    low_Bs2DKPi_prob_p = prod_low(1, amp_Data_dk_dd_p, ampbar_Data_dk_dd_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low_Bs2DKPi')
    low_Bs2DKPi_prob_m = prod_low(-1, amp_Data_dk_dd_m, ampbar_Data_dk_dd_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low_Bs2DKPi')


    total_yield = 1#tf.cast((x[6] + x[7] + x[8] + x[9] + x[10] + x[11])/2, tf.float64)
    
    normalisation_Bplus = totalAmplitudeSquared_Integrated(1, normA_p, normAbar_p, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, normA_m, normAbar_m, normalisationCrossTerms_m, x)
    misid_normalisation_Bplus = totalAmplitudeSquared_DPi_Integrated(1, misid_normA_p, misid_normAbar_p, misid_normalisationCrossTerms_p, x)
    misid_normalisation_Bminus = totalAmplitudeSquared_DPi_Integrated(-1, misid_normA_m, misid_normAbar_m, misid_normalisationCrossTerms_m, x) 

    comb_normalisation_Bplus = tf.cast((normA_p + normAbar_p)*0.5*fracDD + (1-fracDD), tf.float64)
    comb_normalisation_Bminus = tf.cast((normA_m + normAbar_m)*0.5*fracDD + (1-fracDD), tf.float64)
    low_normalisation_Bplus = tf.cast(normAbar_p, tf.float64)
    low_normalisation_Bminus = tf.cast(normA_m, tf.float64)
    low_Bs2DKPi_normalisation_Bplus = tf.cast(normA_p, tf.float64)
    low_Bs2DKPi_normalisation_Bminus = tf.cast(normAbar_m, tf.float64)

    ll_data_p = tf.math.log(1/total_yield*((sig_prob_p/normalisation_Bplus) + (misid_prob_p/misid_normalisation_Bplus) + (comb_prob_p/comb_normalisation_Bplus) + (low_prob_p/low_normalisation_Bplus) + (low_misID_prob_p/low_normalisation_Bplus) + (low_Bs2DKPi_prob_p/low_Bs2DKPi_normalisation_Bplus)))
    ll_data_m = tf.math.log(1/total_yield*((sig_prob_m/normalisation_Bminus) + (misid_prob_m/misid_normalisation_Bminus) + (comb_prob_m/comb_normalisation_Bminus) + (low_prob_m/low_normalisation_Bminus) + (low_misID_prob_m/low_normalisation_Bminus) + (low_Bs2DKPi_prob_m/low_Bs2DKPi_normalisation_Bminus)))


    return (tf.reduce_sum( -2* ll_data_p) + tf.reduce_sum( -2*ll_data_m))

@tf.function
@tf.autograph.experimental.do_not_convert
def prod_nll_dk_ll(x):

    decay = 'b2dk_LL'
    fracDD = 0.30
    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_ll_p, ampbar_dk_ll_p)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_ll_m, ampbar_dk_ll_m)
    misid_normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_ll_p, ampbar_dpi_ll_p)
    misid_normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_ll_m, ampbar_dpi_ll_m)
    normA_p = tf.math.reduce_mean(tf.math.abs(amp_dk_ll_p)**2)
    normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dk_ll_p)**2)
    normA_m = tf.math.reduce_mean(tf.math.abs(amp_dk_ll_m)**2)
    normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dk_ll_m)**2)

    misid_normA_p = tf.math.reduce_mean(tf.math.abs(amp_dpi_ll_p)**2)
    misid_normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_ll_p)**2)
    misid_normA_m = tf.math.reduce_mean(tf.math.abs(amp_dpi_ll_m)**2)
    misid_normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_ll_m)**2)

    sig_prob_p = prod_totalAmplitudeSquared_XY(1, amp_Data_dk_ll_p, ampbar_Data_dk_ll_p, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    sig_prob_m = prod_totalAmplitudeSquared_XY(-1, amp_Data_dk_ll_m, ampbar_Data_dk_ll_m, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])


    misid_prob_p = prod_totalAmplitudeSquared_DPi_XY(1, amp_Data_dk_ll_p, ampbar_Data_dk_ll_p, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    misid_prob_m = prod_totalAmplitudeSquared_DPi_XY(-1, amp_Data_dk_ll_m, ampbar_Data_dk_ll_m, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    comb_prob_p = prod_comb(amp_Data_dk_ll_p, ampbar_Data_dk_ll_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], fracDD=fracDD)
    comb_prob_m = prod_comb(amp_Data_dk_ll_m, ampbar_Data_dk_ll_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], fracDD=fracDD)
    low_prob_p = prod_low(1, amp_Data_dk_ll_p, ampbar_Data_dk_ll_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low')
    low_prob_m = prod_low(-1, amp_Data_dk_ll_m, ampbar_Data_dk_ll_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low')
    low_misID_prob_p = prod_low(1, amp_Data_dk_ll_p, ampbar_Data_dk_ll_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low_misID')
    low_misID_prob_m = prod_low(-1, amp_Data_dk_ll_m, ampbar_Data_dk_ll_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low_misID')
    low_Bs2DKPi_prob_p = prod_low(1, amp_Data_dk_ll_p, ampbar_Data_dk_ll_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low_Bs2DKPi')
    low_Bs2DKPi_prob_m = prod_low(-1, amp_Data_dk_ll_m, ampbar_Data_dk_ll_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low_Bs2DKPi')


    total_yield = 1# tf.cast((x[6] + x[7] + x[8] + x[9] + x[10] + x[11])/2, tf.float64)
    
    normalisation_Bplus = totalAmplitudeSquared_Integrated(1, normA_p, normAbar_p, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, normA_m, normAbar_m, normalisationCrossTerms_m, x)
    misid_normalisation_Bplus = totalAmplitudeSquared_DPi_Integrated(1, misid_normA_p, misid_normAbar_p, misid_normalisationCrossTerms_p, x)
    misid_normalisation_Bminus = totalAmplitudeSquared_DPi_Integrated(-1, misid_normA_m, misid_normAbar_m, misid_normalisationCrossTerms_m, x) 


    comb_normalisation_Bplus = tf.cast((normA_p + normAbar_p)*0.5*fracDD + (1-fracDD), tf.float64)
    comb_normalisation_Bminus = tf.cast((normA_m + normAbar_m)*0.5*fracDD + (1-fracDD), tf.float64)
    low_normalisation_Bplus = tf.cast(normAbar_p, tf.float64)
    low_normalisation_Bminus = tf.cast(normA_m, tf.float64)
    low_Bs2DKPi_normalisation_Bplus = tf.cast(normA_p, tf.float64)
    low_Bs2DKPi_normalisation_Bminus = tf.cast(normAbar_m, tf.float64)

    ll_data_p = tf.math.log(1/total_yield*((sig_prob_p/normalisation_Bplus) + (misid_prob_p/misid_normalisation_Bplus) + (comb_prob_p/comb_normalisation_Bplus) + (low_prob_p/low_normalisation_Bplus) + (low_misID_prob_p/low_normalisation_Bplus) + (low_Bs2DKPi_prob_p/low_Bs2DKPi_normalisation_Bplus)))
    ll_data_m = tf.math.log(1/total_yield*((sig_prob_m/normalisation_Bminus) + (misid_prob_m/misid_normalisation_Bminus) + (comb_prob_m/comb_normalisation_Bminus) + (low_prob_m/low_normalisation_Bminus) + (low_misID_prob_m/low_normalisation_Bminus) + (low_Bs2DKPi_prob_m/low_Bs2DKPi_normalisation_Bminus)))

    return (tf.reduce_sum( -2* ll_data_p) + tf.reduce_sum( -2*ll_data_m))

@tf.function
@tf.autograph.experimental.do_not_convert
def prod_nll_dpi_dd(x):

    decay = 'b2dpi_DD'
    fracDD = 0.27

    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_dd_p, ampbar_dpi_dd_p)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_dd_m, ampbar_dpi_dd_m)
    misid_normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_dd_p, ampbar_dk_dd_p)
    misid_normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_dd_m, ampbar_dk_dd_m)

    normA_p = tf.math.reduce_mean(tf.math.abs(amp_dpi_dd_p)**2)
    normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_dd_p)**2)
    normA_m = tf.math.reduce_mean(tf.math.abs(amp_dpi_dd_m)**2)
    normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_dd_m)**2)

    misid_normA_p = tf.math.reduce_mean(tf.math.abs(amp_dk_dd_p)**2)
    misid_normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dk_dd_p)**2)
    misid_normA_m = tf.math.reduce_mean(tf.math.abs(amp_dk_dd_m)**2)
    misid_normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dk_dd_m)**2)


    sig_prob_p = prod_totalAmplitudeSquared_DPi_XY(1, amp_Data_dpi_dd_p, ampbar_Data_dpi_dd_p, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    sig_prob_m = prod_totalAmplitudeSquared_DPi_XY(-1, amp_Data_dpi_dd_m, ampbar_Data_dpi_dd_m, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])

    misid_prob_p = prod_totalAmplitudeSquared_XY(1, amp_Data_dpi_dd_p, ampbar_Data_dpi_dd_p, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    misid_prob_m = prod_totalAmplitudeSquared_XY(-1, amp_Data_dpi_dd_m, ampbar_Data_dpi_dd_m, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    comb_prob_p = prod_comb(amp_Data_dpi_dd_p, ampbar_Data_dpi_dd_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], fracDD=fracDD)
    comb_prob_m = prod_comb(amp_Data_dpi_dd_m, ampbar_Data_dpi_dd_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], fracDD=fracDD)
    low_prob_p = prod_low(1, amp_Data_dpi_dd_p, ampbar_Data_dpi_dd_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low')
    low_prob_m = prod_low(-1, amp_Data_dpi_dd_m, ampbar_Data_dpi_dd_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low')



    total_yield = 1#tf.cast((x[6] + x[7] + x[8] + x[9])/2, tf.float64)
    
    normalisation_Bplus = totalAmplitudeSquared_DPi_Integrated(1, normA_p, normAbar_p, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_DPi_Integrated(-1, normA_m, normAbar_m, normalisationCrossTerms_m, x)
    misid_normalisation_Bplus = totalAmplitudeSquared_Integrated(1, misid_normA_p, misid_normAbar_p, misid_normalisationCrossTerms_p, x)
    misid_normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, misid_normA_m, misid_normAbar_m, misid_normalisationCrossTerms_m, x) 

    comb_normalisation_Bplus = tf.cast((normA_p + normAbar_p)*0.5*fracDD + (1-fracDD), tf.float64)
    comb_normalisation_Bminus = tf.cast((normA_m + normAbar_m)*0.5*fracDD + (1-fracDD), tf.float64)
    low_normalisation_Bplus = tf.cast(normAbar_p, tf.float64)
    low_normalisation_Bminus = tf.cast(normA_m, tf.float64)


    ll_data_p = tf.math.log(1/total_yield*((sig_prob_p/normalisation_Bplus) + (misid_prob_p/misid_normalisation_Bplus) + (comb_prob_p/comb_normalisation_Bplus) + (low_prob_p/low_normalisation_Bplus) ))
    ll_data_m = tf.math.log(1/total_yield*((sig_prob_m/normalisation_Bminus) + (misid_prob_m/misid_normalisation_Bminus) + (comb_prob_m/comb_normalisation_Bminus) + (low_prob_m/low_normalisation_Bminus)))

    return (tf.reduce_sum( -2* ll_data_p) + tf.reduce_sum( -2*ll_data_m))

@tf.function
@tf.autograph.experimental.do_not_convert
def prod_nll_dpi_ll(x):

    decay = 'b2dpi_LL'
    fracDD = 0.48
    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_ll_p, ampbar_dpi_ll_p)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dpi_ll_m, ampbar_dpi_ll_m)
    misid_normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_ll_p, ampbar_dk_ll_p)
    misid_normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp_dk_ll_m, ampbar_dk_ll_m)

    normA_p = tf.math.reduce_mean(tf.math.abs(amp_dpi_ll_p)**2)
    normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_ll_p)**2)
    normA_m = tf.math.reduce_mean(tf.math.abs(amp_dpi_ll_m)**2)
    normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dpi_ll_m)**2)

    misid_normA_p = tf.math.reduce_mean(tf.math.abs(amp_dk_ll_p)**2)
    misid_normAbar_p = tf.math.reduce_mean(tf.math.abs(ampbar_dk_ll_p)**2)
    misid_normA_m = tf.math.reduce_mean(tf.math.abs(amp_dk_ll_m)**2)
    misid_normAbar_m = tf.math.reduce_mean(tf.math.abs(ampbar_dk_ll_m)**2)


    sig_prob_p = prod_totalAmplitudeSquared_DPi_XY(1, amp_Data_dpi_ll_p, ampbar_Data_dpi_ll_p, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    sig_prob_m = prod_totalAmplitudeSquared_DPi_XY(-1, amp_Data_dpi_ll_m, ampbar_Data_dpi_ll_m, x, pdfs_data[decay+'_p']['sig'], pdfs_data[decay+'_m']['sig'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])


    misid_prob_p = prod_totalAmplitudeSquared_XY(1, amp_Data_dpi_ll_p, ampbar_Data_dpi_ll_p, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    misid_prob_m = prod_totalAmplitudeSquared_XY(-1, amp_Data_dpi_ll_m, ampbar_Data_dpi_ll_m, x, pdfs_data[decay+'_p']['misid'], pdfs_data[decay+'_m']['misid'], Bu_M[decay+'_p'], Bu_M[decay+'_m'])
    comb_prob_p = prod_comb(amp_Data_dpi_ll_p, ampbar_Data_dpi_ll_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], fracDD=fracDD)
    comb_prob_m = prod_comb(amp_Data_dpi_ll_m, ampbar_Data_dpi_ll_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], fracDD=fracDD)
    low_prob_p = prod_low(1, amp_Data_dpi_ll_p, ampbar_Data_dpi_ll_p, x, pdfs_data[decay+'_p'], Bu_M[decay+'_p'], type='low')
    low_prob_m = prod_low(-1, amp_Data_dpi_ll_m, ampbar_Data_dpi_ll_m, x, pdfs_data[decay+'_m'], Bu_M[decay+'_m'], type='low')



    total_yield = 1#tf.cast((x[6] + x[7] + x[8] + x[9])/2, tf.float64)
    
    normalisation_Bplus = totalAmplitudeSquared_DPi_Integrated(1, normA_p, normAbar_p, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_DPi_Integrated(-1, normA_m, normAbar_m, normalisationCrossTerms_m, x)
    misid_normalisation_Bplus = totalAmplitudeSquared_Integrated(1, misid_normA_p, misid_normAbar_p, misid_normalisationCrossTerms_p, x)
    misid_normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, misid_normA_m, misid_normAbar_m, misid_normalisationCrossTerms_m, x) 
    comb_normalisation_Bplus = tf.cast((normA_p + normAbar_p)*0.5*fracDD + (1-fracDD), tf.float64)
    comb_normalisation_Bminus = tf.cast((normA_m + normAbar_m)*0.5*fracDD + (1-fracDD), tf.float64)
    low_normalisation_Bplus = tf.cast(normAbar_p, tf.float64)
    low_normalisation_Bminus = tf.cast(normA_m, tf.float64)



    ll_data_p = tf.math.log(1/total_yield*((sig_prob_p/normalisation_Bplus) + (misid_prob_p/misid_normalisation_Bplus) + (comb_prob_p/comb_normalisation_Bplus) + (low_prob_p/low_normalisation_Bplus) ))
    ll_data_m = tf.math.log(1/total_yield*((sig_prob_m/normalisation_Bminus) + (misid_prob_m/misid_normalisation_Bminus) + (comb_prob_m/comb_normalisation_Bminus) + (low_prob_m/low_normalisation_Bminus) ))

    return (tf.reduce_sum( -2* ll_data_p) + tf.reduce_sum( -2*ll_data_m))

def nll_dk_dd(x):
    x1 = x[:12]
    print(prod_nll_dk_dd(x1))
    return prod_nll_dk_dd(x1)

def nll_dk_ll(x):
    x2 = np.append(x[:6], x[12:])
    print(prod_nll_dk_ll(x2))
    return prod_nll_dk_ll(x2)

def nll_dk(x):
    x1 = x[:12]
    x2 = np.append(x[:6], x[12:])
    print(prod_nll_dk_dd(x1) + prod_nll_dk_ll(x2))
    return prod_nll_dk_dd(x1) + prod_nll_dk_ll(x2)

def nll_dpi_dd(x):
    x3 = x[:12]
    print(prod_nll_dpi_dd(x3))
    return prod_nll_dpi_dd(x3)

def nll_dpi_ll(x):
    x4 = np.append(x[:6], x[12:])
    print(prod_nll_dpi_ll(x4))
    return prod_nll_dpi_ll(x4)

def nll_dpi(x):
    x3 = np.append(x[:6], x[18:22])
    x4 = np.append(x[:6], x[22:26])
    print(prod_nll_dpi_dd(x3) + prod_nll_dpi_ll(x4))
    return prod_nll_dpi_dd(x3) + prod_nll_dpi_ll(x4)

def nll(x):
    x1 = np.array(x[:12])
    x2 = np.append(x[:6], x[12:18])
    x3 = np.append(x[:6], x[18:22])
    x4 = np.append(x[:6], x[22:26])

    print(prod_nll_dpi_dd(x3) + prod_nll_dpi_ll(x4) + prod_nll_dk_dd(x1) + prod_nll_dk_ll(x2))
    return prod_nll_dpi_dd(x3) + prod_nll_dpi_ll(x4) + prod_nll_dk_dd(x1) + prod_nll_dk_ll(x2)




    
xp = tf.Variable(0, tf.float64)
yp = tf.Variable(0, tf.float64)
xm = tf.Variable(0, tf.float64)
ym = tf.Variable(0, tf.float64)
xxi = tf.Variable(0, tf.float64)
yxi = tf.Variable(0, tf.float64)
x = [xp, yp, xm, ym,  -0.05, 0.007, varDict['n_sig_DK_KsPiPi_DD'], varDict['n_misid_DK_KsPiPi_DD'], varDict['n_comb_DK_KsPiPi_DD'], varDict['n_low_DK_KsPiPi_DD'], varDict['n_low_misID_DK_KsPiPi_DD'], varDict['n_low_Bs2DKPi_DK_KsPiPi_DD'], varDict['n_sig_DK_KsPiPi_LL'], varDict['n_misid_DK_KsPiPi_LL'], varDict['n_comb_DK_KsPiPi_LL'], varDict['n_low_DK_KsPiPi_LL'], varDict['n_low_misID_DK_KsPiPi_LL'], varDict['n_low_Bs2DKPi_DK_KsPiPi_LL'], varDict['n_misid_DPi_KsPiPi_DD'],  varDict['n_sig_DPi_KsPiPi_DD'], varDict['n_comb_DPi_KsPiPi_DD'], varDict['n_low_DPi_KsPiPi_DD'],  varDict['n_misid_DPi_KsPiPi_LL'], varDict['n_sig_DPi_KsPiPi_LL'], varDict['n_comb_DPi_KsPiPi_LL'], varDict['n_low_DPi_KsPiPi_LL']]

m = iminuit.Minuit(nll_dk, x)
m.fixed = [False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
mg = m.scipy()

with open(f'{logpath}/simfit_output_{index}.txt', 'w') as f:
    print(mg, file=f)
    print(m.values, file=f)
    print(m.errors, file=f)
    print(m.fval, file=f)
    print(m.nfcn, file=f)
    print(m.covariance, file=f)

time2 = time.time()
print(f'Fitting finished in {time2-time1} seconds')
