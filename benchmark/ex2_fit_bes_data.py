
from tfpcbpggsz.tensorflow_wrapper import tf
import numpy as np
from tfpcbpggsz.generator.phasespace import PhaseSpaceGenerator
from tfpcbpggsz.ulti import get_mass, phsp_to_srd, read_minos_errors
from matplotlib import pyplot as plt
from tfpcbpggsz.bes.config_loader import ConfigLoader
from tfpcbpggsz.bes.model import BaseModel

import os
import time
import argparse
import importlib.resources
path = importlib.resources.files('tfpcbpggsz').joinpath('../benchmark')
parser = argparse.ArgumentParser()
parser.add_argument("--order", type=int, default=4)
parser.add_argument('--minos', type=bool, default=False)
parser.add_argument('--plot_all', type=bool, default=True)
parser.add_argument('--plot_each', type=bool, default=True)
parser.add_argument('--plot', type=bool, default=False)
parser.add_argument('--fit_result', type=str, default=None)

args = parser.parse_args()
order = args.order
minos = args.minos
pc_coefficients = None
if args.fit_result is not None:
    fit_data = np.load(args.fit_result, allow_pickle=True)
    pc_coefficients = fit_data['fitted_params']
    print(pc_coefficients)

#Load the configuration file
config = ConfigLoader("config.yml")
config.get_all_data()
config.vm

print("Signal generated")



print("D decay amplitudes generated")

Model = BaseModel(config)
Model.pc.correctionType = "antiSym_legendre"
Model.pc.order = order
Model.pc.PhaseCorrection()
Model.pc.DEBUG = False


print("NLL function defined")
def fit(minos=False):
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
    if minos:
        minos_r = m.minos()
    fit_result = minos_r if minos else mg
    print(fit_result)
    fcn_minima = fit_result.fval
    correlation = fit_result.covariance.correlation()
    if minos:
        minos_errors = None
        minos_errors = read_minos_errors(fit_result)
        np.savez(f"{path}/results/full_data_fit_order{Model.pc.order}.npz", fitted_params=fit_result.values, fitted_params_error=fit_result.errors, merrors=minos_errors, cov_matrix=fit_result.covariance,corr_matrix=correlation, accurate=fit_result.accurate, valid=fit_result.valid, fcn_minima=fcn_minima)
    else:
        np.savez(f"{path}/results/full_data_fit_order{Model.pc.order}.npz", fitted_params=fit_result.values, fitted_params_error=fit_result.errors, cov_matrix=fit_result.covariance, corr_matrix=correlation, accurate=fit_result.accurate, valid=fit_result.valid, fcn_minima=fcn_minima)

    return fit_result.values, fit_result.errors


def plot(plot_all=False, plot_each=False, coefficient=None):
    from tfpcbpggsz.bes.plotter import Plotter
    Model.pc.set_coefficients(coefficients=coefficient)
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
    plot_dir= os.path.join(path, 'plots')
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
    plt.savefig(f"{plot_dir}/PhaseCorrection_srd_order{Model.pc.order}.png")
    plt.clf()
    plt.scatter(m12_noeff,m13_noeff,c=phase_correction_noeff)
    plt.colorbar()
    plt.savefig(f"{plot_dir}/PhaseCorrection_mass_order{Model.pc.order}.png")

#pc_coefficients, pc_errors = fit(minos=minos)
fit_result = np.load(f"full_data_fit_order{order}.npz", allow_pickle=True)
pc_coefficients = fit_result['fitted_params']
if args.plot:
    
    plot(plot_all=args.plot_all, plot_each=args.plot_each, coefficient=pc_coefficients)
