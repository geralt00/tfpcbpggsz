#import tensorflow as tf
import uproot as up
import cupy as cp
import numpy as np
import time
from multiprocessing import Pool
import iminuit
import matplotlib.pyplot as plt
import sys
cp.cuda.Device(4).use()
sys.path.append('/software/pc24403/tfpcbpggsz/func')
sys.path.append('/software/pc24403/tfpcbpggsz/amp')
from PyD0ToKSpipi2018 import *
Kspipi = PyD0ToKSpipi2018()
Kspipi.init()

mc_path = '/shared/scratch/pc24403/amp_test'

def random_sample(arr: cp.array, size: int = 1) -> cp.array:
    return arr[cp.random.choice(len(arr), size=size, replace=False)]


def get_mass(p1,p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)


def get_p4(decay="b2dpi", cut='', index=1):

    file_name = ''
    if cut == 'int':
        file_name = f'/software/pc24403/PCBPGGSZ/Int/weighted_{decay}.root:DalitzEventList'
    
    elif decay.split('_')[0] == 'b2dk' or decay.split('_')[0] == 'b2dpi':
        if cut == 'p':
            #file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/mass_fit/add_sw/lhcb_toy_{decay}_{index}_CPrange.root:BplusDalitzEventList'
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/1x_test/{decay}_sig_{index}.root:Bplus_DalitzEventList'

        else:
            #file_name = f'/software/pc24403/PCBPGGSZ/outputs/oy/mass_fit/add_sw/lhcb_toy_{decay}_{index}.root:BminusDalitzEventList'
            file_name = f'/software/pc24403/PCBPGGSZ/outputs/toy/1x_test/{decay}_sig_{index}.root:Bminus_DalitzEventList'

    tree = up.open(file_name)
  # Load the branches as arrays
    branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
    array = tree.arrays(branch_names)
       

    p1 = cp.asarray([array["_1_K0S0_E"], array["_1_K0S0_Px"], array["_1_K0S0_Py"], array["_1_K0S0_Pz"]])
    p2 = cp.asarray([array["_2_pi#_E"], array["_2_pi#_Px"], array["_2_pi#_Py"], array["_2_pi#_Pz"]])
    p3 = cp.asarray([array["_3_pi~_E"], array["_3_pi~_Px"], array["_3_pi~_Py"], array["_3_pi~_Pz"]])
    
    # convert 4*1000 into a vectot<double>
    p1 = cp.transpose(p1)
    p2 = cp.transpose(p2)
    p3 = cp.transpose(p3)
    p1bar = cp.hstack((p1[:, :1], cp.negative(p1[:, 1:])))
    p2bar = cp.hstack((p3[:, :1], cp.negative(p3[:, 1:])))
    p3bar = cp.hstack((p2[:, :1], cp.negative(p2[:, 1:])))


    return p1, p2, p3, p1bar, p2bar, p3bar


def load_int_amp(args):
    p1, p2, p3 = args

    return Kspipi.Amp_PFT(p1.tolist(), p2.tolist(), p3.tolist())

def DeltadeltaD(A, Abar):
    temp_var = cp.angle(A*cp.conj(Abar))
    var = (temp_var + cp.pi) % (2 * cp.pi) - cp.pi    
    return var

def totalAmplitudeSquared_Integrated_crossTerm(A, Abar):
    '''
    This function calculates the total amplitude squared for the integrated decay, v0.1 only for MD fitted, no correction yet
    |A||Abar|cos(deltaD)
    |A||Abar|sin(deltaD)
    '''
    phase = DeltadeltaD(A, Abar)
    AAbar = cp.abs(A)*cp.abs(Abar)
    real_part = cp.sum(AAbar*cp.cos(phase))
    imag_part = cp.sum(AAbar*cp.sin(phase))

    return (real_part/phase.shape[1], imag_part/phase.shape[1])

def getAmp(decay='b2dpi', cut='int'):

    start_time = time.time()
    p1, p2, p3, p1bar, p2bar, p3bar = get_p4(decay=decay, cut=cut)
    amplitude = []
    amplitudeBar = []

    p1_np = cp.asnumpy(p1)
    p2_np = cp.asnumpy(p2)
    p3_np = cp.asnumpy(p3)
    p1bar_np = cp.asnumpy(p1bar)
    p2bar_np = cp.asnumpy(p2bar)
    p3bar_np = cp.asnumpy(p3bar)

    data = [(p1_np[i], p2_np[i], p3_np[i]) for i in range(len(p1_np))]
    with Pool(processes=12) as pool:
        amplitude.append(pool.map(load_int_amp, data))
    data_bar = [(p1bar_np[i], p2bar_np[i], p3bar_np[i]) for i in range(len(p1bar_np))]
    with Pool(processes=12) as pool:
        amplitudeBar.append(pool.map(load_int_amp, data_bar))
    
    end_time = time.time()
    #print(f'Amplitude for {decay} loaded in {end_time-start_time} seconds')
    amplitude = cp.asarray(amplitude)
    amplitudeBar = cp.asarray(amplitudeBar)

    return amplitude, amplitudeBar

def getMass(decay='b2dpi', cut='int'):

    start_time = time.time()
    p1, p2, p3, p1bar, p2bar, p3bar = get_p4(decay=decay, cut=cut)
    amplitude = []
    amplitudeBar = []

    p1_np = cp.asnumpy(p1)
    p2_np = cp.asnumpy(p2)
    p3_np = cp.asnumpy(p3)
    p1bar_np = cp.asnumpy(p1bar)
    p2bar_np = cp.asnumpy(p2bar)
    p3bar_np = cp.asnumpy(p3bar)

    s12 = get_mass(p1_np, p2_np)
    s13 = get_mass(p1_np, p3_np)

    return s12, s13

def totalAmplitudeSquared_Integrated(Bsign=1, normA=1.1, normAbar=1.1, crossTerm=(0, 0), x=(0,0,0,0)):
    '''
    A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)

    A^2 + Abar^2 * rb^2 + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)
    '''


    if Bsign == 1:
        xPlus = x[0]
        yPlus = x[1]
        rB2 = xPlus**2 + yPlus**2

        return (normA * rB2 + normAbar + 2*(xPlus *crossTerm[0] - yPlus * crossTerm[1]))
    
    else:
        xMinus = x[2]
        yMinus = x[3]
        rB2 = xMinus**2 + yMinus**2

        return (normA + normAbar  * rB2 + 2*(xMinus *crossTerm[0] + yMinus * crossTerm[1]))
    
def totalAmplitudeSquared_XY(Bsign=1, amp=[], ampbar=[], x=(0,0,0,0)):

    phase = DeltadeltaD(amp, ampbar)
    absA = cp.abs(amp)**2
    absAbar = cp.abs(ampbar)**2


    if Bsign == 1:
        xPlus = x[0]
        yPlus = x[1]
        rB2 = xPlus**2 + yPlus**2
        return (absA * rB2 + absAbar + 2 * cp.sqrt(absA * absAbar) * (xPlus * cp.cos(phase) - yPlus * cp.sin(phase)))
    
    else:
        xMinus = x[2]
        yMinus = x[3]
        rB2 = xMinus**2 + yMinus**2

        return (absA  + absAbar * rB2 + 2 * cp.sqrt(absA * absAbar) * (xMinus * cp.cos(phase) + yMinus * cp.sin(phase)))

def nll_dk_ll(x):
    amp = cp.load(mc_path + '/Int_b2dk_LL_p_amp.npy')
    ampbar = cp.load(mc_path + '/Int_b2dk_LL_p_ampbar.npy')

    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp, ampbar)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp, ampbar)

    normA = cp.sum(cp.abs(amp)**2)/amp.shape[1]
    normAbar = cp.sum(cp.abs(ampbar)**2)/ampbar.shape[1]


    amp_Data_p, ampbar_Data_p = getAmp('b2dk_LL', 'p')
    amp_Data_m, ampbar_Data_m = getAmp('b2dk_LL', 'm')



    normalisation_Bplus = totalAmplitudeSquared_Integrated(1, normA, normAbar, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, normA, normAbar, normalisationCrossTerms_m, x)

    prob_p = totalAmplitudeSquared_XY(1, amp_Data_p, ampbar_Data_p, x)
    prob_m = totalAmplitudeSquared_XY(-1, amp_Data_m, ampbar_Data_m, x)



    ll_data_p = cp.log(prob_p/normalisation_Bplus)
    ll_data_m = cp.log(prob_m/normalisation_Bminus)


    return (cp.sum( -2* ll_data_p) + cp.sum( -2*ll_data_m))

def nll_dk_dd(x):
    amp = cp.load(mc_path + '/Int_b2dk_DD_p_amp.npy')
    ampbar = cp.load(mc_path + '/Int_b2dk_DD_p_ampbar.npy')

    normalisationCrossTerms_p = totalAmplitudeSquared_Integrated_crossTerm(amp, ampbar)
    normalisationCrossTerms_m = totalAmplitudeSquared_Integrated_crossTerm(amp, ampbar)
    normA = cp.sum(cp.abs(amp)**2)/amp.shape[1]
    normAbar = cp.sum(cp.abs(ampbar)**2)/ampbar.shape[1]
    amp_Data_p, ampbar_Data_p = getAmp('b2dk_DD', 'p')
    amp_Data_m, ampbar_Data_m = getAmp('b2dk_DD', 'm')


    normalisation_Bplus = totalAmplitudeSquared_Integrated(1, normA, normAbar, normalisationCrossTerms_p, x)
    normalisation_Bminus = totalAmplitudeSquared_Integrated(-1, normA, normAbar, normalisationCrossTerms_m, x)

    prob_p = totalAmplitudeSquared_XY(1, amp_Data_p, ampbar_Data_p, x)
    prob_m = totalAmplitudeSquared_XY(-1, amp_Data_m, ampbar_Data_m, x)

    ll_data_p = cp.log(prob_p/normalisation_Bplus)
    ll_data_m = cp.log(prob_m/normalisation_Bminus)

    return (cp.sum(-2*ll_data_p) + cp.sum(-2*ll_data_m))

def nll_dk(x):
    print('Calculating nll:',nll_dk_ll(x)+nll_dk_dd(x))
    return nll_dk_ll(x)+nll_dk_dd(x)



m = iminuit.Minuit(nll_dk, (-0.075893, 0.052607, -0.008281, 0.049907))
mg = m.migrad()
print(mg)
