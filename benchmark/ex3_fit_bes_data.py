from tfpcbpggsz.tensorflow_wrapper import tf
import numpy as np
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd, read_minos_errors
from matplotlib import pyplot as plt
from tfpcbpggsz.bes.config_loader import ConfigLoader
import os
import time
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yml")
parser.add_argument("--order", type=int, default=1)
parser.add_argument('--minos', dest='minos', action='store_true', help="Run MINOS to calculate asymmetric errors.")
parser.add_argument('--plot', dest='plot', action='store_true', help="Plot the results after fitting.")
parser.add_argument('--plot-all', dest='plot_all', action='store_true', help="Do not plot all categories combined.")
parser.add_argument('--plot-each', dest='plot_each', action='store_true', help="Do not plot each category separately.")
parser.add_argument('--fit_result', type=str, default=None)

args = parser.parse_args()

order = args.order
minos = args.minos

pc_coefficients = None
print(args.plot_all, args.plot_each)
if args.fit_result is not None:
    with open(args.fit_result, 'r') as f:
        fit_data = json.load(f)
    pc_coefficients = fit_data['Means']
    print(pc_coefficients)
time1 = time.time()

#CP odd Kspipi
config = ConfigLoader(args.config)
config.get_all_data()
config.vm
print("Signal generated")

time2 = time.time()


time3 = time.time()
print("D decay amplitudes generated")
from tfpcbpggsz.bes.model import BaseModel

Model = BaseModel(config)
Model.pc.correctionType = "antiSym_legendre"
Model.pc.order = order
Model.pc.PhaseCorrection()
Model.pc.DEBUG = False
Model.load_bkg_frac()

time4 = time.time()
print("NLL function defined")
def fit(minos=args.minos):
    from iminuit import Minuit

    var_args = {}
    var_names = Model.vm.trainable_vars
    x0 = []
    for i in var_names:
        x0.append(tf.ones_like(Model.vm.get(i)).shape)
        var_args[i] = Model.vm.get(i)

    m = Minuit(
        Model.fun,
        np.array(x0),
        name=var_names
    )
    m.errordef = 1.0
    m.strategy = 2
    mg = m.migrad()
    minos_r = None
    if args.minos is True:
        minos_r = m.minos()
    fit_result = minos_r if args.minos else mg
    print(fit_result)
    fit_result_dir = config._config_data.get("fit_result_dir")
    if not os.path.exists(fit_result_dir):
        os.makedirs(fit_result_dir)

    fit_params(mg, fit_result_dir)
    return fit_result.values, fit_result.errors

def fit_params(mg, dir):
    fit_result = mg
    fcn_minima = fit_result.fval
    correlation = fit_result.covariance.correlation()
    convariance = fit_result.covariance
    status = fit_result.valid and fit_result.accurate
    means = fit_result.values
    errors = fit_result.errors
    minos_errors = None
    minos_errors_dict = None
    if args.minos:
        minos_errors = fit_result.merrors#read_minos_errors(fit_result)
    means_dict = {k: v for k, v in zip(fit_result.parameters, means)}
    errors_dict = {k: v for k, v in zip(fit_result.parameters, errors)}
    if minos_errors is not None:
        minos_errors_dict = {}
        for key in minos_errors.keys():
            low = minos_errors[key].lower
            high = minos_errors[key].upper
            minos_errors_dict[key] = [low, high]

    else:
        minos_errors_dict = None
    fit_result_dict = {
        'Status': status,
        'FCN_minima': fcn_minima,
        'Means': means_dict,
    }
    fit_errors_dict = {
        'Errors': errors_dict,
        'Minos_errors': minos_errors_dict
    }
    json.dump(fit_result_dict, open(f"{dir}/fit_result_order{Model.pc.order}.json", 'w'), indent=4)
    json.dump(fit_errors_dict, open(f"{dir}/fit_errors_order{Model.pc.order}.json", 'w'), indent=4)
    # Save covariance and correlation matrices into csv files
    np.savetxt(f"{dir}/covariance_order{Model.pc.order}.csv", convariance, delimiter=',')
    np.savetxt(f"{dir}/correlation_order{Model.pc.order}.csv", correlation, delimiter=',')


        
    

    return fit_result.values, fit_result.errors


def plot(plot_all=False, plot_each=False, coefficient=None):
    from tfpcbpggsz.bes.plotter import Plotter
    Model.pc.set_coefficients(coefficients=list(coefficient))

    Plotter = Plotter(Model)
    if plot_all:
        Plotter.plot_cato('cp_odd')
        Plotter.plot_cato('cp_even')
        Plotter.plot_cato('dks')
    if plot_each:
        Plotter.plot_each('dks')
        Plotter.plot_each('cp_odd')
        Plotter.plot_each('cp_even')




    #time5 = time.time()

    #PHSP
    phsp = PhaseSpaceGenerator().generate
    plot_dir="./plots_data"
    os.makedirs(plot_dir,exist_ok=True)
    #plot the phase correction in phase space
    plot_phsp = phsp(500000)
    p1_noeff,p2_noeff,p3_noeff = plot_phsp
    m12_noeff = get_mass(p1_noeff,p2_noeff)
    m13_noeff = get_mass(p1_noeff,p3_noeff)
    srd_noeff = phsp_to_srd(m12_noeff,m13_noeff)
    phase_correction_noeff = Model.pc.eval_corr(srd_noeff)

    plt.clf()
    plt.scatter(srd_noeff[0],srd_noeff[1],c=phase_correction_noeff)
    plt.colorbar()
    plt.savefig(f"data_PhaseCorrection_srd_order{Model.pc.order}.png")
    plt.clf()
    plt.scatter(m12_noeff,m13_noeff,c=phase_correction_noeff)
    plt.colorbar()
    plt.savefig(f"data_PhaseCorrection_mass_order{Model.pc.order}.png")

    #Save all the generated data
    time6 = time.time()

    #print("All data saved")
    #print("Save data: ",time6-time1)

pc_coefficients = None
if args.fit_result is None:
    #Fit the data
    pc_coefficients, pc_errors = fit(minos=args.minos)
else:
    with open(args.fit_result, 'r') as f:
        fit_data = json.load(f)
    pc_coefficients = list(fit_data['Means'].values())

if args.plot:

    plot(plot_all=args.plot_all, plot_each=args.plot_each, coefficient=pc_coefficients)
