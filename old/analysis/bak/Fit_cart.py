import numpy as np
import re
import glob
import matplotlib.pyplot as plt
from scipy.stats import norm
import ROOT as r
import os
# Load the style file
import lhcbStyle
from importlib.machinery import SourceFileLoader
import ROOT
def readConfigDict(configDict):
    print('--- INFO: Importing configurations from configDict...')
    varDict = {}
    print('--- INFO: Fixed variables:')
    for key, value in configDict.items():
        if key[-4:] == '_err': continue
        if not isinstance(value, list): 
            varDict[key] = value
            print(f'--- INFO: {key} = {value}')

def extract_fit_result(file_path):
    with open(file_path, 'r') as file:
        config_data = file.read()

    pattern_fitquality = r'FitQuality 0 0 (-?\d+) (-?\d+(?:\.\d+)?) (-?\d+)'
    match_fitquality = re.findall(pattern_fitquality, config_data)

    fit_values = []  # Change to a list to store multiple fit values
    status_values = []
    nll_values = []

    for match in match_fitquality:
        ndf, nll, status = match
        status_values.append(float(status))
        nll_values.append(float(nll))
        fit_values.append({ 
            'ndf': float(ndf),
            'nll': float(nll),
            'status': float(status)
        })

    return fit_values

def fit_result(file_path, is_polar=False):
    with open(file_path, 'r') as file:
        config_data = file.read()

    # Define variable names to search for
    variables_to_search = ['xPlus', 'xMinus', 'yPlus', 'yMinus']
    fit_results = []

    for var in variables_to_search:
        # Create a pattern specific to each variable
        if is_polar:
            pattern = r'{} Free ([-+]?\d+\.\d+) ([-+]?\d+\.\d+)'.format(var)
        else:
            pattern = r"{}\s+Free\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)".format(var)


        matches = re.findall(pattern, config_data)
        if matches:
            for match in matches:
                mean, uncertainty = match
                fit_results.append({
                    'var': var,
                    'mean': float(mean),
                    'uncertainty': float(uncertainty)
                })
    return fit_results

def run_fit(order=1,type="_"):

    
    input_ckm = {'xPlus': -0.0897, 'xMinus': 0.0586, 'yPlus': -0.0110, 'yMinus': 0.0688}
    title_latex = {'xPlus': 'x^{DK}_{+}', 'xMinus': 'x^{DK}_{-}', 'yPlus': 'y^{DK}_{+}', 'yMinus': 'y^{DK}_{-}'}
    # Define variable names and their corresponding RooRealVar objects
    variable_names = ['xPlus', 'xMinus', 'yPlus', 'yMinus']
    variables = {}
    mean_var = {}
    sigma_var = {}
    for name in variable_names:
        variables[name] = r.RooRealVar(f"pull_{name}", f"pull_{name}", -8, 8)
        mean_var[name] = r.RooRealVar(f"mean_{name}", f"mean_{name}", 0, -8, 8)
        sigma_var[name] = r.RooRealVar(f"sigma_{name}", f"sigma_{name}", 1, 0.01, 3)
    
    # Define Gaussian PDFs for all four variables
    gaussians = {}
    for name in variable_names:
        gaussians[name] = r.RooGaussian(f"gaussian_{name}", f"gaussian_{name}", variables[name], mean_var[name], sigma_var[name])
    
    # Define RooFit data sets for all four variables
    pull_data = {}
    for name in variable_names:
        pull_data[name] = r.RooDataSet(f"pull_data_{name}", f"pull_data_{name}", r.RooArgSet(variables[name]))


    nsets = 100 
    looping = 0
    looping_massfit = 0
    file_paths = []
    # Loop through the variables and perform fitting
    for index in range(1, nsets+1):
        full_str=""
        file_paths.append(f"/dice/users/pc24403/BPGGSZ/sig_only_fit/b2dk_output_{index}.txt")
        config = SourceFileLoader('config', '%s/toy/mass_fit/config/%s'%(os.environ['BPGGSZ_OUTPUT_PATH'], f'config_mass_shape_output_{index}.py')).load_module()
        varDict = config.getconfig()
        if varDict['Status'] == 0:
            looping_massfit+=1


    print('INFO: Totally {} files Loaded'.format(len(file_paths)))
    for file_path in file_paths:
        is_polor = False
        results = fit_result(file_path, is_polor)
        wrong = False
        for result in results:
            uncertainty = result['uncertainty']
            if uncertainty == 0:
                wrong = True
                break

        pull_values = []
        status = extract_fit_result(file_path)
        if status[0]['status'] == 0.0 and not wrong:
            looping += 1
            differences = {}  # Create a new differences dictionary for each file
            pull = {}
            for result in results:
                var = result['var']
                mean = result['mean']
                uncertainty = result['uncertainty']
                var_str = ''.join(var)
    
                if var_str in input_ckm:
                    input_value = input_ckm[var_str]
                    difference = mean - input_value
                    pull[var_str] = difference / uncertainty
    
                # Calculate the average pull for this file and add it to the pulls list
            if pull:  # Check if the pull dictionary is not empty
                   # Add the pull value to the RooFit data set
                for name in variable_names:
                    variables[name].setVal(pull[name])
                    pull_data[name].add(r.RooArgSet(variables[name]))
        else: 
            print("Error: Fit failed for file {}".format(file_path))
    
    
    # Create a Canvas and plot the results
    canvas = r.TCanvas("canvas", "Pull Data", 800*2, 600*2)
    canvas.Divide(2, 2)
    label={}
    for i, name in enumerate(variable_names):
    
        gaussians[name].fitTo(pull_data[name])
        canvas.cd(i+1).SetTicks(1, 1)
        pull_plot = variables[name].frame()
        pull_data[name].plotOn(pull_plot, r.RooFit.Binning(30), r.RooFit.MarkerColor(r.kBlue), r.RooFit.LineColor(r.kPink+7))
        gaussians[name].plotOn(pull_plot, r.RooFit.LineColor(r.kAzure-1))
        pull_plot.SetXTitle(title_latex[name])
        pull_plot.Draw()
        # Create a TPaveText object
        label[name] = r.TPaveText(0.65, 0.7, 0.9, 0.85, "NDC")
        label[name].SetFillColor(0)
        label[name].SetBorderSize(0)
        label[name].AddText("Mean: {:.2f}#pm{:.2f}".format(mean_var[name].getVal(), mean_var[name].getError()))
        label[name].AddText("Sigma: {:.2f}#pm{:.2f}".format(sigma_var[name].getVal(), sigma_var[name].getError()))
        label[name].Draw()
        pull_plot.Draw("Same")
    
    
    canvas.SaveAs(f"pull_plots{type}p{order}.pdf")
    #save the mean value into txt file
    with open(f"pull_study/pull_plots{type}p{order}.txt", "w") as f:
        for name in variable_names:
            f.write(f"{name} {mean_var[name].getVal()} {mean_var[name].getError()}\n")
    #finished and clear all

    # Delete variables and free up memory
    for name in variable_names:
        del variables[name]
        del mean_var[name]
        del sigma_var[name]
        del pull_data[name]
        del gaussians[name]
        del label[name]

    # Close ROOT resources
    r.gROOT.GetListOfFunctions().Delete()
    r.gROOT.GetListOfSpecials().Delete()

    canvas.Clear()
    canvas.Close()
    print("Successful: Mass fit {:.3f}%".format(looping_massfit/nsets*100))
    print("Successful: CP fit {:.3f}%".format(looping/nsets*100))


def run_res_fit(order=1,type="_"):

    
    input_ckm = {'xPlus': -0.0897, 'xMinus': 0.0586, 'yPlus': -0.0110, 'yMinus': 0.0688}
    title_latex = {'xPlus': 'x^{DK}_{+}', 'xMinus': 'x^{DK}_{-}', 'yPlus': 'y^{DK}_{+}', 'yMinus': 'y^{DK}_{-}'}
    # Define variable names and their corresponding RooRealVar objects
    variable_names = ['xPlus', 'xMinus', 'yPlus', 'yMinus']
    variables = {}
    mean_var = {}
    sigma_var = {}
    for name in variable_names:
        variables[name] = r.RooRealVar(f"pull_{name}", f"pull_{name}", -0.1, 0.1)
        mean_var[name] = r.RooRealVar(f"mean_{name}", f"mean_{name}", 0, -1, 1)
        sigma_var[name] = r.RooRealVar(f"sigma_{name}", f"sigma_{name}", 1, 0.001, 1.1)
    
    # Define Gaussian PDFs for all four variables
    gaussians = {}
    for name in variable_names:
        gaussians[name] = r.RooGaussian(f"gaussian_{name}", f"gaussian_{name}", variables[name], mean_var[name], sigma_var[name])
    
    # Define RooFit data sets for all four variables
    pull_data = {}
    for name in variable_names:
        pull_data[name] = r.RooDataSet(f"pull_data_{name}", f"pull_data_{name}", r.RooArgSet(variables[name]))


    nsets = 100 
    looping = 0
    looping_massfit = 0
    file_paths = []
    # Loop through the variables and perform fitting
    for index in range(1, nsets+1):
        full_str=""
        file_paths.append(f"/dice/users/pc24403/BPGGSZ/sig_only_fit/b2dk_output_{index}.txt")
        config = SourceFileLoader('config', '%s/toy/mass_fit/config/%s'%(os.environ['BPGGSZ_OUTPUT_PATH'], f'config_mass_shape_output_{index}.py')).load_module()
        varDict = config.getconfig()
        if varDict['Status'] == 0:
            looping_massfit+=1


    print('INFO: Totally {} files Loaded'.format(len(file_paths)))
    for file_path in file_paths:
        is_polor = False
        results = fit_result(file_path, is_polor)
        wrong = False
        for result in results:
            uncertainty = result['uncertainty']
            if uncertainty == 0:
                wrong = True
                break

        pull_values = []
        status = extract_fit_result(file_path)
        if status[0]['status'] == 0.0 and not wrong:
            looping += 1
            differences = {}  # Create a new differences dictionary for each file
            pull = {}
            for result in results:
                var = result['var']
                mean = result['mean']
                uncertainty = result['uncertainty']
                var_str = ''.join(var)
    
                if var_str in input_ckm:
                    input_value = input_ckm[var_str]
                    difference =  input_value - mean
                    pull[var_str] = difference 
    
                # Calculate the average pull for this file and add it to the pulls list
            if pull:  # Check if the pull dictionary is not empty
                   # Add the pull value to the RooFit data set
                for name in variable_names:
                    variables[name].setVal(pull[name])
                    pull_data[name].add(r.RooArgSet(variables[name]))
    
    
    # Create a Canvas and plot the results
    canvas = r.TCanvas("canvas", "Pull Data", 800*2, 600*2)
    canvas.Divide(2, 2)
    label={}
    for i, name in enumerate(variable_names):
    
        gaussians[name].fitTo(pull_data[name])
        canvas.cd(i+1).SetTicks(1, 1)
        pull_plot = variables[name].frame()
        pull_data[name].plotOn(pull_plot, r.RooFit.Binning(30), r.RooFit.MarkerColor(r.kBlue), r.RooFit.LineColor(r.kPink+7))
        gaussians[name].plotOn(pull_plot, r.RooFit.LineColor(r.kAzure-1))
        pull_plot.SetXTitle(title_latex[name])
        pull_plot.Draw()
        # Create a TPaveText object
        label[name] = r.TPaveText(0.65, 0.7, 0.9, 0.85, "NDC")
        label[name].SetFillColor(0)
        label[name].SetBorderSize(0)
        mean_center = mean_var[name].getVal()*100
        sigma_center = sigma_var[name].getVal()*100
        mean_error = mean_var[name].getError()*100
        sigma_error = sigma_var[name].getError()*100
        label[name].AddText("100*Mean: ({:.2f}#pm{:.2f})".format(mean_var[name].getVal()*100, mean_var[name].getError()*100))
        label[name].AddText("100*Sigma: ({:.2f}#pm{:.2f})".format(sigma_var[name].getVal()*100, sigma_var[name].getError()*100))
        label[name].Draw()
        pull_plot.Draw("Same")
    
    
    canvas.SaveAs(f"residual_plots{type}p{order}.pdf")
    #save the mean value into txt file
    with open(f"pull_study/residual_plots{type}p{order}.txt", "w") as f:
        for name in variable_names:
            f.write(f"{name} {mean_var[name].getVal()} {mean_var[name].getError()}\n")
    #finished and clear all

    # Delete variables and free up memory
    for name in variable_names:
        del variables[name]
        del mean_var[name]
        del sigma_var[name]
        del pull_data[name]
        del gaussians[name]
        del label[name]

    # Close ROOT resources
    r.gROOT.GetListOfFunctions().Delete()
    r.gROOT.GetListOfSpecials().Delete()

    canvas.Clear()
    canvas.Close()
    print("Successful: Mass fit {:.3f}%".format(looping_massfit/nsets*100))
    print("Successful: CP fit {:.3f}%".format(looping/nsets*100))

def run_var_fit(order=1,type="_"):

    
    input_ckm = {'xPlus': -0.0897, 'xMinus': 0.0586, 'yPlus': -0.0110, 'yMinus': 0.0688}
    exp_unc = {'xPlus': 0.009, 'xMinus': 0.011, 'yPlus': 0.010, 'yMinus': 0.010}
    title_latex = {'xPlus': 'x^{DK}_{+}', 'xMinus': 'x^{DK}_{-}', 'yPlus': 'y^{DK}_{+}', 'yMinus': 'y^{DK}_{-}'}

    # Define variable names and their corresponding RooRealVar objects
    variable_names = ['xPlus', 'xMinus', 'yPlus', 'yMinus']
    variables = {}
    mean_var = {}
    sigma_var = {}
    for name in variable_names:
        variables[name] = r.RooRealVar(f"pull_{name}", f"pull_{name}", input_ckm[name]-10*abs(exp_unc[name]), input_ckm[name]+10*abs(exp_unc[name]))
        mean_var[name] = r.RooRealVar(f"mean_{name}", f"mean_{name}", 0, -1, 1)
        sigma_var[name] = r.RooRealVar(f"sigma_{name}", f"sigma_{name}", 1, 0.00001, 1.1)
    
    # Define Gaussian PDFs for all four variables
    gaussians = {}
    for name in variable_names:
        gaussians[name] = r.RooGaussian(f"gaussian_{name}", f"gaussian_{name}", variables[name], mean_var[name], sigma_var[name])
    
    # Define RooFit data sets for all four variables
    pull_data = {}
    for name in variable_names:
        pull_data[name] = r.RooDataSet(f"pull_data_{name}", f"pull_data_{name}", r.RooArgSet(variables[name]))


    nsets = 100 
    looping = 0
    looping_massfit = 0
    file_paths = []
    # Loop through the variables and perform fitting
    for index in range(1, nsets+1):
        full_str=""
        file_paths.append(f"/dice/users/pc24403/BPGGSZ/sig_only_fit/b2dk_output_{index}.txt")
        config = SourceFileLoader('config', '%s/toy/mass_fit/config/%s'%(os.environ['BPGGSZ_OUTPUT_PATH'], f'config_mass_shape_output_{index}.py')).load_module()
        varDict = config.getconfig()
        if varDict['Status'] == 0:
            looping_massfit+=1


    print('INFO: Totally {} files Loaded'.format(len(file_paths)))
    for file_path in file_paths:
        is_polor = False
        results = fit_result(file_path, is_polor)
        wrong = False
        for result in results:
            uncertainty = result['uncertainty']
            if uncertainty == 0:
                wrong = True
                break

        pull_values = []
        status = extract_fit_result(file_path)
        if status[0]['status'] == 0.0 and not wrong:
            looping += 1
            differences = {}  # Create a new differences dictionary for each file
            pull = {}
            for result in results:
                var = result['var']
                mean = result['mean']
                uncertainty = result['uncertainty']
                var_str = ''.join(var)
    
                if var_str in input_ckm:
                    input_value = input_ckm[var_str]
                    difference =  mean
                    pull[var_str] = difference 
    
                # Calculate the average pull for this file and add it to the pulls list
            if pull:  # Check if the pull dictionary is not empty
                   # Add the pull value to the RooFit data set
                for name in variable_names:
                    variables[name].setVal(pull[name])
                    pull_data[name].add(r.RooArgSet(variables[name]))
    
    
    # Create a Canvas and plot the results
    canvas = r.TCanvas("canvas", "Pull Data", 800*2, 600*2)
    canvas.Divide(2, 2)
    label={}
    for i, name in enumerate(variable_names):
    
        gaussians[name].fitTo(pull_data[name])
        canvas.cd(i+1).SetTicks(1, 1)
        pull_plot = variables[name].frame()
        pull_data[name].plotOn(pull_plot, r.RooFit.Binning(30), r.RooFit.MarkerColor(r.kBlue), r.RooFit.LineColor(r.kPink+7))
        gaussians[name].plotOn(pull_plot, r.RooFit.LineColor(r.kAzure-1))
        pull_plot.SetXTitle(title_latex[name])
        pull_plot.Draw()
        # Create a TPaveText object
        label[name] = r.TPaveText(0.65, 0.7, 0.9, 0.85, "NDC")
        label[name].SetFillColor(0)
        label[name].SetBorderSize(0)
        mean_center = mean_var[name].getVal()*100
        sigma_center = sigma_var[name].getVal()*100
        mean_error = mean_var[name].getError()*100
        sigma_error = sigma_var[name].getError()*100
        label[name].AddText("100*Mean: ({:.2f}#pm{:.2f})".format(mean_var[name].getVal()*100, mean_var[name].getError()*100))
        label[name].AddText("100*Sigma: ({:.2f}#pm{:.2f})".format(sigma_var[name].getVal()*100, sigma_var[name].getError()*100))
        label[name].Draw()
        pull_plot.Draw("Same")
    
    
    canvas.SaveAs(f"var_plots{type}p{order}.pdf")
    #save the mean value into txt file
    with open(f"pull_study/var_plots{type}p{order}.txt", "w") as f:
        for name in variable_names:
            f.write(f"{name} {mean_var[name].getVal()} {mean_var[name].getError()}\n")
    #finished and clear all

    # Delete variables and free up memory
    for name in variable_names:
        del variables[name]
        del mean_var[name]
        del sigma_var[name]
        del pull_data[name]
        del gaussians[name]
        del label[name]

    # Close ROOT resources
    r.gROOT.GetListOfFunctions().Delete()
    r.gROOT.GetListOfSpecials().Delete()

    canvas.Clear()
    canvas.Close()
    print("Successful: Mass fit {:.3f}%".format(looping_massfit/nsets*100))
    print("Successful: CP fit {:.3f}%".format(looping/nsets*100))
def run_unc_fit(order=1,type="_"):

    
    input_ckm = {'xPlus': -0.0897, 'xMinus': 0.0586, 'yPlus': -0.0110, 'yMinus': 0.0688}
    title_latex = {'xPlus': '#sigma(x^{DK}_{+})', 'xMinus': '#sigma(x^{DK}_{-})', 'yPlus': '#sigma(y^{DK}_{+})', 'yMinus': '#sigma(y^{DK}_{-})'}
    exp_unc = {'xPlus': 0.009, 'xMinus': 0.011, 'yPlus': 0.010, 'yMinus': 0.010}

    # Define variable names and their corresponding RooRealVar objects
    variable_names = ['xPlus', 'xMinus', 'yPlus', 'yMinus']
    variables = {}
    mean_var = {}
    sigma_var = {}
    for name in variable_names:
        variables[name] = r.RooRealVar(f"pull_{name}", f"pull_{name}", exp_unc[name]-2*abs(exp_unc[name]), exp_unc[name]+2*abs(exp_unc[name]))
        mean_var[name] = r.RooRealVar(f"mean_{name}", f"mean_{name}", 0, -1, 1)
        sigma_var[name] = r.RooRealVar(f"sigma_{name}", f"sigma_{name}", 1, 0.001, 1.1)
    
    # Define Gaussian PDFs for all four variables
    gaussians = {}
    for name in variable_names:
        gaussians[name] = r.RooGaussian(f"gaussian_{name}", f"gaussian_{name}", variables[name], mean_var[name], sigma_var[name])
    
    # Define RooFit data sets for all four variables
    pull_data = {}
    for name in variable_names:
        pull_data[name] = r.RooDataSet(f"pull_data_{name}", f"pull_data_{name}", r.RooArgSet(variables[name]))


    nsets = 100 
    looping = 0
    looping_massfit = 0
    file_paths = []
    # Loop through the variables and perform fitting
    for index in range(1, nsets+1):
        full_str=""
        file_paths.append(f"/dice/users/pc24403/BPGGSZ/sig_only_fit/b2dk_output_{index}.txt")
        config = SourceFileLoader('config', '%s/toy/mass_fit/config/%s'%(os.environ['BPGGSZ_OUTPUT_PATH'], f'config_mass_shape_output_{index}.py')).load_module()
        varDict = config.getconfig()
        if varDict['Status'] == 0:
            looping_massfit+=1


    print('INFO: Totally {} files Loaded'.format(len(file_paths)))
    for file_path in file_paths:
        is_polor = False
        results = fit_result(file_path, is_polor)
        wrong = False
        for result in results:
            uncertainty = result['uncertainty']
            if uncertainty == 0:
                wrong = True
                break

        pull_values = []
        status = extract_fit_result(file_path)
        if status[0]['status'] == 0.0 and not wrong:
            looping += 1
            differences = {}  # Create a new differences dictionary for each file
            pull = {}
            for result in results:
                var = result['var']
                mean = result['mean']
                uncertainty = result['uncertainty']
                var_str = ''.join(var)
    
                if var_str in input_ckm:
                    input_value = input_ckm[var_str]
                    difference = input_value - mean
                    pull[var_str] = uncertainty 
    
                # Calculate the average pull for this file and add it to the pulls list
            if pull:  # Check if the pull dictionary is not empty
                   # Add the pull value to the RooFit data set
                for name in variable_names:
                    variables[name].setVal(pull[name])
                    pull_data[name].add(r.RooArgSet(variables[name]))
    
    
    # Create a Canvas and plot the results
    canvas = r.TCanvas("canvas", "Pull Data", 800*2, 600*2)
    canvas.Divide(2, 2)
    label={}
    for i, name in enumerate(variable_names):
    
        gaussians[name].fitTo(pull_data[name])
        canvas.cd(i+1).SetTicks(1, 1)
        pull_plot = variables[name].frame()
        pull_data[name].plotOn(pull_plot, r.RooFit.Binning(50), r.RooFit.MarkerColor(r.kBlue), r.RooFit.LineColor(r.kPink+7))
        gaussians[name].plotOn(pull_plot, r.RooFit.LineColor(r.kAzure-1))
        pull_plot.SetXTitle(title_latex[name])
        pull_plot.Draw()
        # Create a TPaveText object
        label[name] = r.TPaveText(0.65, 0.7, 0.9, 0.85, "NDC")
        label[name].SetFillColor(0)
        label[name].SetBorderSize(0)
        mean_center = mean_var[name].getVal()*100
        sigma_center = sigma_var[name].getVal()*100
        mean_error = mean_var[name].getError()*100
        sigma_error = sigma_var[name].getError()*100
        label[name].AddText("100*Mean: ({:.2f}#pm{:.2f})".format(mean_var[name].getVal()*100, mean_var[name].getError()*100))
        label[name].AddText("100*Sigma: ({:.2f}#pm{:.2f})".format(sigma_var[name].getVal()*100, sigma_var[name].getError()*100))
        label[name].Draw()
        pull_plot.Draw("Same")
    
    
    canvas.SaveAs(f"uncertain_plots{type}p{order}.pdf")
    #save the mean value into txt file
    with open(f"pull_study/uncertain_plots{type}p{order}.txt", "w") as f:
        for name in variable_names:
            f.write(f"{name} {mean_var[name].getVal()} {mean_var[name].getError()}\n")
    #finished and clear all

    # Delete variables and free up memory
    for name in variable_names:
        del variables[name]
        del mean_var[name]
        del sigma_var[name]
        del pull_data[name]
        del gaussians[name]
        del label[name]

    # Close ROOT resources
    r.gROOT.GetListOfFunctions().Delete()
    r.gROOT.GetListOfSpecials().Delete()

    canvas.Clear()
    canvas.Close()
    print("Successful: Mass fit {:.3f}%".format(looping_massfit/nsets*100))
    print("Successful: CP fit {:.3f}%".format(looping/nsets*100))

for i in range(1):
    #run_fit(i)
    #run_res_fit(i)
    run_var_fit(i)
    #run_unc_fit(i)
