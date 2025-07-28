import numpy as np
import time
import iminuit
import matplotlib.pyplot as plt
import mplhep
import pandas as pd
import json
import os
import yaml
from datetime import datetime

# Import your custom libraries
from tfpcbpggsz.amp.amplitude import Amplitude
from tfpcbpggsz.core import *
from tfpcbpggsz.ulti import *
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.lhcb.common_classes import *
from tfpcbpggsz.lhcb.selections import *
from tfpcbpggsz.lhcb.ntuples import *
from tfpcbpggsz.lhcb.variables import *
from tfpcbpggsz.lhcb.common_constants import *
from tfpcbpggsz.lhcb.functions import *
from tfpcbpggsz.lhcb.Measurement import Measurement
from tfpcbpggsz.lhcb.VARDICT_DALITZ import VARDICT, varDict

from tfpcbpggsz.tensorflow_wrapper import tf
tf.get_logger().setLevel('INFO')

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    print("_INFO_ Loading configuration from:", config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """Creates output directories using paths from the config."""
    date = datetime.now().strftime("%Y_%m_%d")
    # Format directory paths with the current date
    fit_dir = config['paths']['fit_dir'].format(date=date)
    plot_dir = config['paths']['plot_dir'].format(date=date)
    
    print(f"_INFO_ Fit results will be saved to: {fit_dir}")
    print(f"_INFO_ Plots will be saved to: {plot_dir}")
    
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(f"{fit_dir}/preFit", exist_ok=True)
    os.makedirs(f"{fit_dir}/results", exist_ok=True)
    return fit_dir, plot_dir

def prepare_analysis_data(config):
    """Loads ntuples, calculates efficiencies, and prepares all data structures for the fit."""
    print("_INFO_ Preparing analysis data...")
    
    # Extract settings from config
    settings = config['settings']
    components = config['components']
    list_channels = settings['list_channels']
    list_sources = settings['list_sources']
    bdt_cut = settings['bdt_cut']
    min_mass, max_mass = settings['mass_range']
    
    # Initialize data structures
    ntuples = {}
    total_eff = {}
    
    Kspipi_up = Amplitude()
    Kspipi_up.init()

    # Define variables needed for reading ntuples
    basic_list_var = ["Bu_ID", "zp_p", "zm_pp", "m_Kspip", "m_Kspim"]
    for particle in ["Ks", "h1", "h2"]:
        for mom in ["PE", "PX", "PY", "PZ"]:
            basic_list_var.append(f"{particle}_{mom}")
            
    # --- Loop over MC sources to get efficiencies ---
    for source in list_sources:
        ntuples[source] = {}
        total_eff[source] = {}
        pre_cuts_eff = {}
        fin_cuts_eff = {}
        bdt_cut_efficiency = {}
        
        for channel in list_channels:
            ntuples[source][channel] = Ntuple(f"{source}_TightCut_LooserCuts_fixArrow", channel, "YRUN2", "MagAll")
            
            pre_cuts_eff = ntuples[source][channel].get_merged_cuts_eff("preliminary")
            fin_cuts_eff = ntuples[source][channel].get_merged_cuts_eff("final")
            
            cut_str = f"(BDT_output > {bdt_cut}) & {ntuples[source][channel].dict_final_cuts['Bach_PID']} & ({ntuples[source][channel].variable_to_fit} < {max_mass}) & ({ntuples[source][channel].variable_to_fit} > {min_mass})"
            list_var = [ntuples[source][channel].variable_to_fit] + basic_list_var
            
            ntuples[source][channel].store_events(
                ntuples[source][channel].final_cuts_paths,
                list_var,
                cut_str,
                Kspipi_up
            )
            
            bdt_cut_efficiency = len(ntuples[source][channel].uproot_data[list_var[0]]) / fin_cuts_eff["YRUN2"]["MagAll"]["selected_events"]
            total_eff[source][channel] = pre_cuts_eff["YRUN2"]["MagAll"]["efficiency"] * fin_cuts_eff["YRUN2"]["MagAll"]["efficiency"] * bdt_cut_efficiency
    
    # --- Calculate efficiency ratios needed for constraints ---
    # NOTE: These must be calculated here, as they depend on the `total_eff` values computed above.
    ratio_map = {
        "ratio_DK_to_Dpi": total_eff["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"] / total_eff["MC_Bu_D0pi_KSpipi"]["CB2DPI_D2KSPIPI_DD"],
        "ratio_Dpi_misID_to_Dpi": total_eff["MC_Bu_D0pi_KSpipi"]["CB2DK_D2KSPIPI_DD"] / total_eff["MC_Bu_D0pi_KSpipi"]["CB2DPI_D2KSPIPI_DD"],
        "ratio_DK_misID_to_DK": total_eff["MC_Bu_D0K_KSpipi"]["CB2DPI_D2KSPIPI_DD"] / total_eff["MC_Bu_D0K_KSpipi"]["CB2DK_D2KSPIPI_DD"]
    }
    print("_INFO_ Calculated efficiency ratios for constraints:", ratio_map)
    
    # --- Build the final constraint dictionaries ---
    # Here we substitute the string placeholders from the YAML with the calculated ratio values
    dict_constrained_parameters = []
    for const_list in config['constrained_parameters']:
        new_const_list = []
        for item in const_list:
            new_item = item.copy()
            # The second part of the constraint is the one with the factor
            constraint_def = new_item[1]
            for i, factor in enumerate(constraint_def):
                #print(f"_DEBUG_ Processing factor: {factor}")
                if isinstance(factor, str):
                    # If it's a string and exists in the ratio_map, replace it
                    if factor in ratio_map:
                        constraint_def[i] = ratio_map[factor]  # Replace string with calculated value
                elif isinstance(factor, list):
                    for j, sub_factor in enumerate(factor):
                        if isinstance(sub_factor, str) and sub_factor in ratio_map:
                            constraint_def[i][j] = ratio_map[sub_factor]  # Replace string with calculated value
            # print(f"_DEBUG_ Final constraint definition: {constraint_def}")
            new_item[1] = constraint_def
            # Append the modified item to the new list
            # print(f"_INFO_ Adding constraint: {new_item}")
            new_const_list.append(new_item)
        dict_constrained_parameters.extend(new_const_list)

    # --- Load and prepare SDATA ---
    input_variables = {}
    for channel in list_channels:
        # Prepare input_variables dictionary needed by NLLComputation
        input_variables[channel] = {}
        for comp in components[channel]:
            input_variables[channel][comp[0]] = VARDICT["SDATA"][channel][comp[0]]
            pass
        pass
    input_variables["SHARED_THROUGH_CHANNELS"] = VARDICT["SDATA"]["SHARED_THROUGH_CHANNELS"]
    #
    ntuples["SDATA"] = {}
    for channel in list_channels:
        ntuples["SDATA"][channel] = Ntuple("SDATA", channel, "YRUN2", "MagAll")
        index_channel = list(input_variables.keys()).index(channel)
        ntuples["SDATA"][channel].initialise_fit(components[channel], index_channel)
        ntuples["SDATA"][channel].define_mass_pdfs()
        
        cut_str = f"(BDT_output > {bdt_cut}) & {ntuples['SDATA'][channel].dict_final_cuts['Bach_PID']} & ({ntuples['SDATA'][channel].variable_to_fit} < {max_mass}) & ({ntuples['SDATA'][channel].variable_to_fit} > {min_mass})"
        list_var = [ntuples["SDATA"][channel].variable_to_fit] + basic_list_var
        
        ntuples["SDATA"][channel].store_events(
            ntuples["SDATA"][channel].final_cuts_paths,
            list_var,
            cut_str,
            Kspipi_up
        )

    return ntuples, input_variables, dict_constrained_parameters


def run_fit(nll_computer, config, ntuples, fit_dir):
    """Configures and runs the iminuit fit, then saves the results."""
    print("\n_INFO_ Starting fit...")
    
    start_values = config['fit_parameters']['start_values']
    limit_values = config['fit_parameters']['limit_values']
    parameters_to_fit = list(start_values.keys())

    @tf.function
    def nll_func(x):
        return nll_computer.get_total_mass_nll(x)

    # Test the NLL function with starting values
    initial_nll = nll_func(tf.cast(list(start_values.values()), tf.float64))
    print(f"_INFO_ Initial -log(L) = {initial_nll.numpy():.2f}")

    # Configure Minuit
    m = iminuit.Minuit(nll_func, list(start_values.values()), name=parameters_to_fit)
    m.limits = list(limit_values.values())
    m.errordef = iminuit.Minuit.LEAST_SQUARES # For a -2*log(L) fit

    # Run Migrad!
    mg = m.migrad()
    print(mg)

    if not mg.valid:
        print("_WARNING_ Fit did not converge or is not valid!")
        return None, None

    # --- Save results ---
    print("_INFO_ Saving fit results...")
    means_results = {p: v for p, v in zip(parameters_to_fit, mg.values)}
    errors_results = {p: e for p, e in zip(parameters_to_fit, mg.errors)}

    with open(f"{fit_dir}/results/means_results.json", "w") as f:
        json.dump(means_results, f, indent=4)
    with open(f"{fit_dir}/results/errors_results.json", "w") as f:
        json.dump(errors_results, f, indent=4)

    # Save correlation and covariance matrices
    pd_corr = pd.DataFrame(mg.covariance.correlation(), index=parameters_to_fit, columns=parameters_to_fit)
    pd_cov = pd.DataFrame(mg.covariance, index=parameters_to_fit, columns=parameters_to_fit)
    
    pd_corr.to_csv(f"{fit_dir}/results/correlation_matrix.csv")
    pd_cov.to_csv(f"{fit_dir}/results/covariance_matrix.csv")

    return mg.values, mg.errors


def generate_plots(means, nll_computer, ntuples, config, plot_dir):
    """Generates and saves the post-fit plots."""
    if means is None:
        print("_WARNING_ Skipping plotting because fit results are not available.")
        return
        
    print("_INFO_ Generating post-fit plots...")
    
    # Extract settings
    settings = config['settings']
    components = config['components']
    components_tex = config['components_tex']
    min_mass, max_mass = settings['mass_range']
    nbins = settings['nbins']
    mass_scaling = (max_mass - min_mass) / float(nbins)
    Bmass_vec = np.arange(min_mass, max_mass, 1)
    tf_Bmass_vec = tf.cast(Bmass_vec, tf.float64)

    # Get the dictionary of all variables with their post-fit values
    postFit_list_variables = ntuples["SDATA"]["CB2DK_D2KSPIPI_DD"].get_list_variables(
        nll_computer.fixed_variables,
        params=means,
        shared_parameters=nll_computer.shared_parameters,
        constrained_parameters=nll_computer.constrained_parameters
    )

    # Plot each channel
    for channel in settings['list_channels']:
        plt.figure(figsize=(12, 8))
        mplhep.style.use("LHCb2")
        
        # Get the PDF values
        mass_pdfs_values = ntuples["SDATA"][channel].draw_combined_mass_pdfs(
            tf_Bmass_vec,
            postFit_list_variables
        )
        
        # Plot data histogram
        mplhep.histplot(
            np.histogram(ntuples["SDATA"][channel].Bu_M["both"], bins=nbins, range=[min_mass, max_mass]),
            label=ntuples["SDATA"][channel].channel.tex + " Data",
            **config['kwargs_data']
        )
        
        # Plot total PDF
        plt.plot(
            Bmass_vec,
            mass_scaling * mass_pdfs_values["both"]["total_mass_pdf"],
            label="Total Fit PDF",
            color='red',
            linewidth=2
        )
        
        # Plot individual components
        for comp in components[channel]:
            plt.plot(
                Bmass_vec,
                mass_pdfs_values["both"][comp[0]] * mass_scaling,
                linestyle="--",
                label=components_tex[comp[0]]
            )
            
        plt.xlabel(f"Constrained {ntuples['SDATA'][channel].variable_to_fit.replace('_', ' ')} [MeV]")
        plt.ylabel(f"Candidates / ({round(mass_scaling)} MeV)")
        plt.title(rf"Invariant Mass Fit for {ntuples['SDATA'][channel].channel.tex}")
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        plot_path_png = f"{plot_dir}/{channel}_mass_fit.png"
        plot_path_pdf = f"{plot_dir}/{channel}_mass_fit.pdf"
        plt.savefig(plot_path_png)
        plt.savefig(plot_path_pdf)
        plt.close("all")
        print(f"_INFO_ Saved plot to {plot_path_png}")


## ------------------------------------------------------------------
## Main Execution Block
## ------------------------------------------------------------------

if __name__ == '__main__':
    time_start = time.time()
    
    # 1. Load configuration from YAML file
    config = load_config('mass_fit.yml')

    # 2. Create output directories
    fit_dir, plot_dir = setup_directories(config)

    # 3. Load data, calculate efficiencies, and prepare constraints
    ntuples, input_vars, constrained_params = prepare_analysis_data(config)
    
    # 4. Initialize the NLL Computation object
    # The structure of gaussian constraints from YAML is slightly different, so we re-map it
    gaussian_constraints = [
        [item['from'], item['to']] for item in config['gaussian_constraints']
    ]
    
    NLL = NLLComputation(
        start_values=config['fit_parameters']['start_values'],
        limit_values=config['fit_parameters']['limit_values'],
        dict_shared_parameters=config['shared_parameters'],
        dict_constrained_parameters=constrained_params,
        dict_gaussian_constraints=gaussian_constraints,
        list_channels=config['settings']['list_channels'],
        dict_fixed_variables=input_vars,
        ntuples=ntuples["SDATA"]
    )
    
    # 5. Run the fit
    means, errors = run_fit(NLL, config, ntuples, fit_dir)
    
    # 6. Generate post-fit plots
    generate_plots(means, NLL, ntuples, config, plot_dir)

    time_end = time.time()
    print(f"\n✅ Analysis finished in {time_end - time_start:.2f} seconds.")