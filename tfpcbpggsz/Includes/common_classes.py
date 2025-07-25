from tfpcbpggsz.Includes.selections import *
from tfpcbpggsz.Includes.ntuples import *
from tfpcbpggsz.Includes.variables import *
from tfpcbpggsz.Includes.common_constants import *
from tfpcbpggsz.Includes.functions import *
from tfpcbpggsz.Includes.VARDICT import VARDICT
from tfpcbpggsz.core import *
from tfpcbpggsz.masspdfs import *
from tfpcbpggsz.amp import *
import json
import os
from scipy.stats import poisson
import pandas as pd
import numpy as np
import uproot
import tensorflow as tf
import sys
import tensorflow_probability as tfp


from tfpcbpggsz.masspdfs    import MassPDF, norm_pdf, norm_distribution
from tfpcbpggsz.dalitz_pdfs import DalitzPDF


class MagPol:
    def __init__(self, name, list_magpol=None):
        self.name = name
        self.list_magpol = list_magpol

MagDown = MagPol("MagDown")
MagDown.list_magpol = [MagDown]
MagUp   = MagPol("MagUp" )
MagUp.list_magpol = [MagUp]
MagAll  = MagPol("MagAll", [MagUp, MagDown])
DICT_MAGPOL = {
    "MagDown": MagDown,
    "MagUp"  : MagUp,
    "MagAll" : MagAll
}
    
TREE_NAMES = { # const channel_dictionary 
    "CALLEVENTSTUPLE" :               "AllEventsTuple/EventTuple" ,        
    "CB2DPI_D2KSPIPI_DD" :            "B2DPi_D2KsPiPi_DD/DecayTree" ,     
    "CB2DK_D2KSPIPI_DD" :             "B2DK_D2KsPiPi_DD/DecayTree" ,      
    "CB2DPI_D2KSPIPI_LL" :            "B2DPi_D2KsPiPi_LL/DecayTree" ,     
    "CB2DK_D2KSPIPI_LL" :             "B2DK_D2KsPiPi_LL/DecayTree" ,      
    "CB2DPI_D2KSPIPI_LL_WS" :         "B2DPi_D2KsPiPi_LL_WS/DecayTree" ,  
    "CB2DK_D2KSPIPI_LL_WS" :          "B2DK_D2KsPiPi_LL_WS/DecayTree" ,   
    "CB2DK_D2KSPIPI_DD_WS" :          "B2DK_D2KsPiPi_DD_WS/DecayTree" ,   
    "CB2DPI_D2KSPIPI_DD_WS" :         "B2DPi_D2KsPiPi_DD_WS/DecayTree" ,  
    "CB2DPI_D2KSKK_DD_WS" :           "B2DPi_D2KsKK_DD_WS/DecayTree" ,    
    "CB2DK_D2KSKK_DD_WS" :            "B2DK_D2KsKK_DD_WS/DecayTree" ,     
    "CB2DPI_D2KSKK_LL" :              "B2DPi_D2KsKK_LL/DecayTree" ,	      
    "CB2DK_D2KSKK_LL" :               "B2DK_D2KsKK_LL/DecayTree" ,	      
    "CB2DPI_D2KSKK_LL_WS" :           "B2DPi_D2KsKK_LL_WS/DecayTree" ,    
    "CB2DK_D2KSKK_LL_WS" :            "B2DK_D2KsKK_LL_WS/DecayTree" ,     
    "CB2DPI_D2KSKK_DD" :              "B2DPi_D2KsKK_DD/DecayTree" ,	      
    "CB2DK_D2KSKK_DD" :               "B2DK_D2KsKK_DD/DecayTree" ,        
}

class Source:
    def __init__(self, name, isMC, tex, color):
        self.name = name
        self.isMC = isMC
        self.tex  = tex
        self._path = None
        self.color = color

#### SOURCES
SDATA = Source("SDATA", False,"LHCb Data", "black")
MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow  = Source("MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow" , True,r"MC $B^{\pm} \rightarrow [K_S\pi^+\pi^-]_D K^{\pm}$", "teal")
MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow = Source("MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow", True,r"MC $B^{\pm} \rightarrow [K_S\pi^+\pi^-]_D \pi^{\pm}$", "deeppink")
MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow    = Source("MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow"   , True,r"MC $B^{\pm} \rightarrow [K_SK^+K^-]_D K^{\pm}$", "limegreen")
MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow   = Source("MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow"  , True,r"MC $B^{\pm} \rightarrow [K_SK^+K^-]_D \pi^{\pm}$", "mediumorchid")
DICT_SOURCES = {
    "SDATA": SDATA,
    "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"  : MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow ,
    "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow" : MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow,
    "MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow"    : MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow   ,
    "MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow"   : MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow  ,
}
DICT_CHANNELS_TO_SOURCES = {
    "CB2DK_D2KSPIPI_DD" : "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow" ,
    "CB2DPI_D2KSPIPI_DD": "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow",
    "CB2DK_D2KSPIPI_LL" : "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow" ,
    "CB2DPI_D2KSPIPI_LL": "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow",
    "CB2DK_D2KSKK_DD"   : "MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow"   ,
    "CB2DPI_D2KSKK_DD"  : "MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow"  ,
    "CB2DK_D2KSKK_LL"   : "MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow"   ,
    "CB2DPI_D2KSKK_LL"  : "MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow"  ,
}

class Channel:
    def __init__(self, name, tex, bach, Ddecay,KsTrack):
        self.name = name
        self.tree_name = TREE_NAMES[name]
        self._PROBNN_BACH_CUTS = None
        self._PROBNN_D_CHILDREN_CUTS = None
        self.preliminary_cuts = None
        self.final_cuts = None
        self.tex = tex
        self.bach = bach
        self.Ddecay = Ddecay
        self.KsTrack = KsTrack
        # self.Bach_ProbNN_cuts = None
    def get_preliminary_cuts(self, _year): # TCut 
        res = {}
        ## trigger
        res["Trigger"] = MAP_TRIGGER[_year]
        self.preliminary_cuts = res["Trigger"]
        ## cleaning
        res["DTF_Status"] = CUT_CLEANING_DATA
        self.preliminary_cuts += " & " + res["DTF_Status"]
        ## ProbNN D CHILDREN
        res["ProbNN_D_CHILDREN"] = MAP_PROBNN_DCHILDREN[_year][self.name]
        self.preliminary_cuts += " & " + res["ProbNN_D_CHILDREN"]
        # ## ProbNN Bach
        # res["ProbNN_Bach"] = MAP_PROBNN_BACH[_year][self.name]
        # self.preliminary_cuts += " & " + res["ProbNN_Bach"]
        ## DTF cuts
        res["DTF_Masses"] = MAP_DTF_CUTS[self.name]
        self.preliminary_cuts += " & " + res["DTF_Masses"]
        ## ENDVERTEX  ## encorporated into kinematics 02/09/2024
        # res["EndVertex_BDZSIG"] = CUTS_ENDVERTEX
        # self.preliminary_cuts += " & " + res["EndVertex_BDZSIG"]
        ## KINEMATICS
        res["Kinematics"] = MAP_KINEMATIC_CUTS[self.name]
        self.preliminary_cuts += " & " + res["Kinematics"]
        ## Lepton_veto
        res["Bach_PID"] = MAP_LEPTON_VETO[self.name]
        self.preliminary_cuts += " & " + res["Bach_PID"]
        ## Bachelor PID
        res["Lepton_veto"] = MAP_BACH_PID_PRELIM[self.name]
        self.preliminary_cuts += " & " + res["Lepton_veto"]
        self.dict_preliminary_cuts = res
        return self.preliminary_cuts
    def get_final_cuts(self, _year): # TCut 
        res = {}
        ## is_in_D_PhaseSpace
        res["is_in_D_PhaseSpace"] = "(is_in_D_PhaseSpace == True)"
        self.final_cuts = res["is_in_D_PhaseSpace"]
        res["Bach_PID"] = MAP_BACH_PID_FINAL[self.name]
        self.final_cuts += " & " + res["Bach_PID"]
        self.dict_final_cuts = res
        return self.final_cuts
    # def get_Bach_ProbNN_cuts(self, _year): # TCut 
    #     ## ProbNN Bach
    #     self.Bach_ProbNN_cuts = MAP_PROBNN_BACH[_year][self.name]
    #     return self.Bach_ProbNN_cuts


### CHANNELS
CB2DK_D2KSPIPI_DD  = Channel("CB2DK_D2KSPIPI_DD" ,r"$[K_S|_{DD}\pi^+\pi^-]_{D} K$"  ,"K","Kspipi", "DD")
CB2DPI_D2KSPIPI_DD = Channel("CB2DPI_D2KSPIPI_DD",r"$[K_S|_{DD}\pi^+\pi^-]_{D} \pi$","pi","Kspipi", "DD")
CB2DK_D2KSPIPI_LL  = Channel("CB2DK_D2KSPIPI_LL" ,r"$[K_S|_{LL}\pi^+\pi^-]_{D} K$"  ,"K","Kspipi", "LL")
CB2DPI_D2KSPIPI_LL = Channel("CB2DPI_D2KSPIPI_LL",r"$[K_S|_{LL}\pi^+\pi^-]_{D} \pi$","pi","Kspipi", "LL")
CB2DK_D2KSKK_DD    = Channel("CB2DK_D2KSKK_DD" ,r"$[K_S|_{DD}K^+K^-]_{D} K$"  ,"K","KsKK", "DD")
CB2DPI_D2KSKK_DD   = Channel("CB2DPI_D2KSKK_DD",r"$[K_S|_{DD}K^+K^-]_{D} \pi$","pi","KsKK", "DD")
CB2DK_D2KSKK_LL    = Channel("CB2DK_D2KSKK_LL" ,r"$[K_S|_{LL}K^+K^-]_{D} K$"  ,"K","KsKK", "LL")
CB2DPI_D2KSKK_LL   = Channel("CB2DPI_D2KSKK_LL",r"$[K_S|_{LL}K^+K^-]_{D} \pi$","pi","KsKK", "LL")
DICT_CHANNELS = {
    "CB2DK_D2KSPIPI_DD" : CB2DK_D2KSPIPI_DD ,
    "CB2DPI_D2KSPIPI_DD": CB2DPI_D2KSPIPI_DD,
    "CB2DK_D2KSPIPI_LL" : CB2DK_D2KSPIPI_LL ,
    "CB2DPI_D2KSPIPI_LL": CB2DPI_D2KSPIPI_LL,
    "CB2DK_D2KSKK_DD"   : CB2DK_D2KSKK_DD   ,
    "CB2DPI_D2KSKK_DD"  : CB2DPI_D2KSKK_DD  ,
    "CB2DK_D2KSKK_LL"   : CB2DK_D2KSKK_LL   ,
    "CB2DPI_D2KSKK_LL"  : CB2DPI_D2KSKK_LL  ,
}


class Year:
    def __init__(self, name, tex, run_name, luminosity=-1):
        self.name = name
        self.tex  = tex
        self._trigger_cuts = None
        self.list_years = None
        self.run_name = run_name
        self.luminosity = luminosity
    def __str__(self):
        return self.name

Y2011 = Year("Y2011", "2011","YRUN1", 1.0)
Y2011.list_years = [Y2011]
Y2012 = Year("Y2012", "2012","YRUN1", 2.0)
Y2012.list_years = [Y2012]
Y2015 = Year("Y2015", "2015","YRUN2", 0.30)
Y2015.list_years = [Y2015]
Y2016 = Year("Y2016", "2016","YRUN2", 1.6)
Y2016.list_years = [Y2016]
Y2017 = Year("Y2017", "2017","YRUN2", 1.7)
Y2017.list_years = [Y2017]
Y2018 = Year("Y2018", "2018","YRUN2", 2.1)
Y2018.list_years = [Y2018]
###RUN1
YRUN1 = Year("YRUN1", "Run 1", "YRUN1")
YRUN1.list_years = [Y2011, Y2012]
###RUN2
YRUN2 = Year("YRUN2", "Run 2", "YRUN2")
YRUN2.list_years = [Y2015, Y2016, Y2017, Y2018]
###YALL
YALL  = Year("YALL", "Run 1&2", "YALL")
YALL.list_years  = [Y2011, Y2012, Y2015, Y2016, Y2017, Y2018]
### 
DICT_YEARS = {
    "Y2011" : Y2011,
    "Y2012" : Y2012,
    "Y2015" : Y2015,
    "Y2016" : Y2016,
    "Y2017" : Y2017,
    "Y2018" : Y2018,
    "YRUN1" : YRUN1,
    "YRUN2" : YRUN2,
    "YALL"  : YALL ,
}

## getting the correct ntuple for a given source and a given channel year and magnet
def get_ntuple(_source, _channel, _year, _magpol, selec = "before_selection"):
    res = []
    tree_name = _channel.tree_name
    if (selec == "before_selection"):
        ## Which TTree do you need: 
        data_dict = SOURCE_LOCATIONS[_source.name]
        for year_item in _year.list_years:
            for magpol_item in _magpol.list_magpol:
                try:
                    paths = data_dict[year_item.name][magpol_item.name]
                    # print(paths)
                    for _path in paths:
                        res.append(_path+":"+tree_name)
                        # print("path:", _path)
                        pass # loop paths
                except KeyError:
                    print(" WARNING ================================ ntuples before selection")
                    print("No ntuples for ",_source.name,_channel.name,year_item.name,magpol_item.name)
                    pass # try except
                pass # loop mag
            pass # loop years
        return res
    elif (selec == "preliminary_cuts") :
        analysis_step = "00_Preliminary_cuts"
    elif (selec == "BDT_training"):
        analysis_step = "01_Prepare_training_datasets"
    elif (selec == "With_BDT"):
        analysis_step = "04_Apply_BDT"
    elif (selec == "final_cuts"):
        analysis_step = "05_Final_cuts"
    else:
        print(" ERROR ================================")
        print(" problem in ntuples.py --- check")
        return ""
    for year_item in _year.list_years:
        for magpol_item in _magpol.list_magpol:
            _path = pathname_storage+analysis_step+"/"+_channel.name+"/"+_source.name+"/"+year_item.name+"/"+magpol_item.name+"/outfile.root"
            if (os.path.isfile(_path)) :
                res.append(_path+":"+tree_name)
            else:
                print(" WARNING ================================ ntuples Pre-c or BDT")
                print("No ntuples for ",_source.name,_channel.name,year_item.name,magpol_item.name)
                print(_path)
            pass # loop mag
        pass # loop years
    return res

class Ntuple:
    def __init__(self, source, channel, year, magpol):
        if (type(source) == str):
            self.source = DICT_SOURCES[source]
            pass
        else:
            self.source = source
            pass
        if (type(channel) == str):
            self.channel = DICT_CHANNELS[channel]
            pass
        else:
            self.channel = channel
            pass
        if (type(year) == str):
            self.year = DICT_YEARS[year]
            pass
        else :
            self.year = year
            pass
        if (type(magpol) == str):
            self.magpol = DICT_MAGPOL[magpol]
            pass
        else :
            self.magpol = magpol
            pass
        self.original_paths = get_ntuple(self.source, self.channel, self.year, self.magpol, selec = "before_selection")
        self.preliminary_cuts_paths = get_ntuple(self.source, self.channel, self.year, self.magpol, selec = "preliminary_cuts")
        self.final_cuts_paths = get_ntuple(self.source, self.channel, self.year, self.magpol, selec = "final_cuts")
        # self.folder_path = self.channel.name +"/"+ self.source.name+"/"+self.year.name+"/"+self.magpol.name+"/"
        self._trigger_cuts = self.year._trigger_cuts
        self.preliminary_cuts = self.channel.get_preliminary_cuts(self.year.run_name)
        self.dict_preliminary_cuts = self.channel.dict_preliminary_cuts
        self.preliminary_cuts_eff  = self.get_preliminary_cuts_eff()
        self.final_cuts = self.channel.get_final_cuts(self.year.run_name)
        self.dict_final_cuts = self.channel.dict_final_cuts
        self.final_cuts_eff  = self.get_final_cuts_eff()
        # self.Bach_ProbNN_cuts = self.channel.get_Bach_ProbNN_cuts(self.year.run_name)
        self.variables = list(VARIABLES[self.year.run_name].keys())
        if (self.source.isMC) :
            self.variables = self.variables + list(TRUTH_VARIABLES[self.year.run_name].keys())
            pass
        self.new_variables = list(NEW_VARIABLES[self.year.run_name].keys())
        if (self.source.isMC) :
            self.new_variables = self.new_variables + list(NEW_TRUTH_VARIABLES[self.year.run_name].keys())
            pass
        self.tex = self.year.tex + " " + self.source.tex + " selected " + self.channel.tex
        self.BDT_training_paths = get_ntuple(self.source, self.channel, self.year, self.magpol, selec = "BDT_training")
        self.With_BDT_paths = get_ntuple(self.source, self.channel, self.year, self.magpol, selec = "With_BDT")
        self.truth_matching_cuts = self.get_truth_matching_cuts()
        self.variable_to_fit = VARIABLE_TO_FIT[self.channel.name]
        pass

    
    def __str__(self):
        pattern = '''
        Source         : {}\n
        Channel        : {}\n
        Year           : {}\n
        MagPol         : {}\n
        Paths OG       : {}\n
        Paths Pre-c    : {}\n
        Paths BDT      : {}\n
        Paths post-BDT : {}\n
        Paths Fin-c    : {}\n
        '''
        return pattern.format(self.source.name, self.channel.name, self.year.name, self.magpol.name, self.original_paths, self.preliminary_cuts_paths, self.BDT_training_paths, self.With_BDT_paths, self.final_cuts_paths)
    
    def get_preliminary_cuts_eff(self):
        res = {}
        for year_item in self.year.list_years:
            res[year_item.name] = {}
            for magpol_item in self.magpol.list_magpol:
                try:
                    path = pathname_local+'Includes/efficiency/'+self.channel.name+"/"+self.source.name+"/"+year_item.name+"/"+magpol_item.name+'/preliminary_cuts_info.json'
                    with open(path) as f:
                        print(" ================= GET PRELIMINARY CUTS EFFICIENCY INFO AT")
                        print(path)
                        res[year_item.name][magpol_item.name] = json.load(f)
                except FileNotFoundError:
                    print("Preliminary cuts efficiency not computed for this ntuple")
                    # res[year_item.name][magpol_item.name] = None
                    pass
                pass
            pass
        return res
    
    def get_final_cuts_eff(self):
        res = {}
        for year_item in self.year.list_years:
            res[year_item.name] = {}
            for magpol_item in self.magpol.list_magpol:
                try:
                    path = pathname_local+'Includes/efficiency/'+self.channel.name+"/"+self.source.name+"/"+year_item.name+"/"+magpol_item.name+'/final_cuts_info.json'
                    with open(path) as f:
                        print(" ================= GET FINAL CUTS EFFICIENCY INFO AT")
                        print(path)
                        res[year_item.name][magpol_item.name] = json.load(f)
                except FileNotFoundError:
                        print("Final cuts efficiency not computed for this ntuple")
                        # res[year_item.name][magpol_item.name] = None
        return res

    def get_merged_cuts_eff(self,level):
        ### this should be ran only when we have
        # ran the preliminary selections for all samples
        if (level == "preliminary"):
            tmp_eff = self.preliminary_cuts_eff
        elif (level == "final"):
            tmp_eff = self.final_cuts_eff
        else:
            print("ERROR --------- in get_merged_cuts_eff, wrong level for cuts")
            exit()
        num_lumi   = 0
        denom_lumi = 0
        res = {} # self.preliminary_cuts_eff
        res[self.year.name] = {}
        res[self.year.name][self.magpol.name] = {}
        total_input_events = 0
        total_selected_events = 0
        for year_item in self.year.list_years:
            try:
                ### necessary since some MC don't have samples for
                # certain years
                test = tmp_eff[year_item.name]["MagUp"]
            except KeyError:
                continue
            selected_events   = 0
            input_events      = 0
            for magpol_item in self.magpol.list_magpol:
                ### weighted average on magpol"
                selected_events += tmp_eff[year_item.name][magpol_item.name]["selected_events"]
                input_events    += tmp_eff[year_item.name][magpol_item.name]["input_events"]
                pass
            num_lumi   += year_item.luminosity * selected_events / input_events
            denom_lumi += year_item.luminosity
            total_input_events    += input_events    
            total_selected_events += selected_events 
            pass
        res[self.year.name][self.magpol.name]["efficiency"]      = num_lumi / denom_lumi
        res[self.year.name][self.magpol.name]["input_events"]    = total_input_events
        res[self.year.name][self.magpol.name]["selected_events"] = total_selected_events
        return res

    
    def get_truth_matching_cuts(self):
        if (self.source.isMC == False):
            return "(Bu_M > 0)"
        # print("MAP_TRUTH_MATCHING[self.source.name]", MAP_TRUTH_MATCHING[self.source.name])
        return MAP_TRUTH_MATCHING[self.source.name]


    ################### Fit stuff
    def initialise_fit(self,components, index_channel):
        ##### invariant mass fit objects
        self.components      = components # [self.source.name][self.channel.name]
        self.variable_to_fit = VARIABLE_TO_FIT[self.channel.name]
        #### for mass fit
        self.i_c = index_channel
        pass


    def initialise_fixed_pdfs(self, _fixed_variables):
        #### first initialise the mass pdfs for the sum of Bplus and Bminus
        # print("Initialising PDFS")
        self.comp_yields           = {}
        self.comp_yields["Bplus"]  = []
        self.comp_yields["Bminus"] = []
        self.comp_yields["both"]   = []
        for i_comp in range(len(self.mass_pdfs["both"])):
            # print("   mass pdf for ",self.mass_pdfs["both"][i_comp].component)
            # tf.print("i_comp: ",i_comp)
            self.mass_pdfs["both"][i_comp].get_mass_pdf(
                _fixed_variables[self.i_c][i_comp][4], # last index is space
                Bsign=None
            )
            self.comp_yields["both"].append(_fixed_variables[self.i_c][i_comp][4][INDEX_YIELDS["Bplus"]] + _fixed_variables[self.i_c][i_comp][4][INDEX_YIELDS["Bminus"]])
            pass
        # and then initialise both dalitz and mass separately for Bplus and Bminus
        for Bsign in BSIGNS.keys():
            # print(Bsign)
            for i_comp in range(len(self.mass_pdfs[Bsign])):
                # tf.print("i_comp: ",i_comp)
                # print("   mass and dalitz pdf for ",self.mass_pdfs[Bsign][i_comp].component)
                self.comp_yields[Bsign].append(_fixed_variables[self.i_c][i_comp][4][INDEX_YIELDS[Bsign]])
                self.mass_pdfs[Bsign][i_comp].get_mass_pdf(
                    _fixed_variables[self.i_c][i_comp][4],
                    Bsign=Bsign
                )
                # print(" in get_dalitz_pdfs")
                # print("    i_comp ", i_comp)
                # print(" component ", self.mass_pdfs[Bsign][i_comp].component)
                # print("  function ", self.dalitz_pdfs[Bsign][i_comp].name)
                # print(" variables ", fixed_variables[self.i_c][i_comp])
                # print(" shraed_variables ", fixed_variables[INDEX_SHARED_THROUGH_CHANNELS][0])
                # print(" fixed var  : ",fixed_variables[self.i_c][i_comp][INDEX_SPACE[Bsign]])
                # print(" fixed var+1: ",fixed_variables[self.i_c][i_comp][INDEX_SPACE[Bsign]+1])
                # print(i_comp)
                self.dalitz_pdfs[Bsign][i_comp].get_dalitz_pdf(
                    self.norm_ampD0[Bsign]   ,
                    self.norm_ampD0bar[Bsign],
                    self.norm_zp_p[Bsign]    ,
                    self.norm_zm_pp[Bsign]   ,
                    variables_eff    = _fixed_variables[self.i_c][i_comp][INDEX_SPACE[Bsign]],
                    variables_model  = _fixed_variables[self.i_c][i_comp][INDEX_SPACE[Bsign]+1],
                    shared_variables = _fixed_variables[INDEX_SHARED_THROUGH_CHANNELS][0]
                    ### the 0 here doesn't mean anything, it refers to the
                    # "component" column of the dictionary, which is irrelevant
                    # for the parameters that are shared through channels.
                    # In the origin dictionary VARDICT, it corresponds to the "parameters"
                    # column.
                    # INDEX_SPACE is defined by VARDICT, where we first have Bplus_eff
                    # and then Bplus_model for the order of the parameters
                )
                pass
            pass
        return
    
    # @tf.function
    def get_list_variables(self, fixed_variables, params=None, shared_parameters=None, constrained_parameters=None):
        ##### self.list_variables will contain all variables
        #  (all channels etc, similarly to list_vardict)
        list_variables = 1*fixed_variables
        for i_channel in range(len(list_variables)): # loop channels
            for i_comp in range(len(list_variables[i_channel])): # loop components
                for i_space in range(len(list_variables[i_channel][i_comp])): # loop params
                    for i_var in range(len(list_variables[i_channel][i_comp][i_space])): # loop space
                        for i_par in range(len(shared_parameters)):
                            # tf.print(" start loop i_par CB2DK Kspipi mean at  [",i_channel,"][",i_comp,"][",i_var,"] :", list_variables[0][0][1])
                            if ([i_channel, i_comp, i_space, i_var] in shared_parameters[i_par]):
                                # print(i_channel)
                                # print(list_variables)
                                list_variables[i_channel][i_comp][i_space][i_var] = params[i_par]
                                # tf.print("params[",i_par,"]                           :", params[i_par])
                                # tf.print("list_variables[",i_channel,"][",i_comp,"][",i_var,"] :", list_variables[i_channel][i_comp][i_var])
                                break
                            else:
                                # list_variables[i_channel][i_comp][i_var] = variables[i_channel][i_comp][i_var]
                                pass
                            # tf.print(" end loop i_par CB2DK Kspipi mean at  [",i_channel,"][",i_comp,"][",i_var,"] :", list_variables[0][0][1])
                            pass
                        pass
                    pass
                pass
            pass
        # tf.print(" BEFORE CONSTRAINED")
        # tf.print("CB2DK Kspipi mean      ",list_variables[0][0][1])
        ### now we look if one of the shared_param is also used for constraining
        for i_const in constrained_parameters:
            i_chan   = i_const[0][0]
            i_comp   = i_const[0][1]
            i_space  = i_const[0][2]
            i_var    = i_const[0][3]
            if (type(i_const[1])==float):
                list_variables[i_chan][i_comp][i_space][i_var] = i_const[1]
                continue
            for i_par in range(len(shared_parameters)):
                if (i_const[1][:-1] in shared_parameters[i_par]):
                    const_i_chan  = i_const[1][0]
                    const_i_comp  = i_const[1][1]
                    const_i_space = i_const[1][2]
                    const_i_var   = i_const[1][3]
                    # list_variables[i_chan][i_comp][i_var] = tf.cast(params[i_par]*i_const[1][-1],tf.float64)
                    factor = list_variables[const_i_chan][const_i_comp][const_i_space][const_i_var]
                    # print("list_variables[const_i_chan][const_i_comp]")
                    # print(list_variables[const_i_chan][const_i_comp])
                    new_const = i_const[1][4]
                    while (type(new_const) == list):
                        # print(factor, " = factor should be equal to Dpi yield " , list_variables[1][0][2][0])
                        const_i_chan  = new_const[0] # contains the indices of the next parameter to multiply
                        const_i_comp  = new_const[1] # contains the indices of the next parameter to multiply
                        const_i_space = new_const[2] # contains the indices of the next parameter to multiply
                        const_i_var   = new_const[3] # contains the indices of the next parameter to multiply
                        # print(new_const  )
                        # print(list_variables[const_i_chan][const_i_comp][const_i_space][const_i_var], " should be equal to BR_ratio ", list_variables[-1][0][2][0])
                        factor *= list_variables[const_i_chan][const_i_comp][const_i_space][const_i_var]
                        new_const = new_const[4]
                        # print(new_const, " should be equal to efficiency ratio 1.2672565348886882")
                        pass
                    list_variables[i_chan][i_comp][i_space][i_var] = new_const*factor
                    pass
                pass
            pass
        # tf.print(" AFTER CONSTRAINED")
        # tf.print("CB2DK Kspipi mean      ",list_variables[0][0][1])
        self.list_variables = list_variables
        return list_variables
        ####
        

    def store_events(self, paths, list_var, cuts, Kspipi_ampModel, aliases = None):
        print("STARTING STORING THE EVENTS FOR ", self.tex)
        self.uproot_data = pd.DataFrame.from_dict(
            uproot.concatenate(paths,
                               list_var,
                               cuts,
                               aliases = aliases,
                               library='np'))
        ##### Momenta
        # for particle in ["KS", "pip", "pim"]:
        #     for mom_ in ["PE", "PX", "PY", "PZ"]:                
        #         pass
        #     pass
        ##### Amplitudes
        if ("KS_PE" not in self.uproot_data):
            self.uproot_data["KS_PE"] = self.uproot_data["Ks_PE"]
            self.uproot_data["KS_PX"] = self.uproot_data["Ks_PX"]
            self.uproot_data["KS_PY"] = self.uproot_data["Ks_PY"]
            self.uproot_data["KS_PZ"] = self.uproot_data["Ks_PZ"]            
            self.uproot_data["pip_PE"] = np.where(self.uproot_data["Bu_ID"]>0, self.uproot_data["h1_PE"], self.uproot_data["h2_PE"])
            self.uproot_data["pip_PX"] = np.where(self.uproot_data["Bu_ID"]>0, self.uproot_data["h1_PX"], self.uproot_data["h2_PX"])
            self.uproot_data["pip_PY"] = np.where(self.uproot_data["Bu_ID"]>0, self.uproot_data["h1_PY"], self.uproot_data["h2_PY"])
            self.uproot_data["pip_PZ"] = np.where(self.uproot_data["Bu_ID"]>0, self.uproot_data["h1_PZ"], self.uproot_data["h2_PZ"])
            self.uproot_data["pim_PE"] = np.where(self.uproot_data["Bu_ID"]<0, self.uproot_data["h1_PE"], self.uproot_data["h2_PE"])
            self.uproot_data["pim_PX"] = np.where(self.uproot_data["Bu_ID"]<0, self.uproot_data["h1_PX"], self.uproot_data["h2_PX"])
            self.uproot_data["pim_PY"] = np.where(self.uproot_data["Bu_ID"]<0, self.uproot_data["h1_PY"], self.uproot_data["h2_PY"])
            self.uproot_data["pim_PZ"] = np.where(self.uproot_data["Bu_ID"]<0, self.uproot_data["h1_PZ"], self.uproot_data["h2_PZ"])
            pass
        p1 = np.asarray([
            self.uproot_data["KS_PE"],
            self.uproot_data["KS_PX"],
            self.uproot_data["KS_PY"],
            self.uproot_data["KS_PZ"],            
        ]).transpose()
        p2 = np.asarray([
            self.uproot_data["pim_PE"],
            self.uproot_data["pim_PX"],
            self.uproot_data["pim_PY"],
            self.uproot_data["pim_PZ"],            
        ]).transpose()
        p3 = np.asarray([
            self.uproot_data["pip_PE"],
            self.uproot_data["pip_PX"],
            self.uproot_data["pip_PY"],
            self.uproot_data["pip_PZ"],            
        ]).transpose()
        tmp_Amp    = Kspipi_ampModel.AMP(
            p1.tolist(),
            p2.tolist(),
            p3.tolist()
        )   # []
        p1bar, p2bar, p3bar = np.concatenate([p1[:, :1], np.negative(p1[:, 1:])], axis=1), np.concatenate([p2[:, :1], np.negative(p2[:, 1:])], axis=1), np.concatenate([p3[:, :1], np.negative(p3[:, 1:])], axis=1)
        tmp_Ampbar = Kspipi_ampModel.AMP(
            p1bar.tolist(),
            p3bar.tolist(),
            p2bar.tolist()
        )   # []
        # indices    = []
        # for index, row in self.uproot_data.iterrows():
        #     indices.append(index)
        #     # print(index)
        #     # print(row["zp_p"])
        #     # print(row["zm_pp"])
        #     tmp_amps = Kspipi_ampModel.get_amp(
        #         row["zp_p"] ,
        #         row["zm_pp"]
        #     )
        #     # print(tmp_amps)
        #     tmp_Amp.append(tmp_amps[0])
        #     tmp_Ampbar.append(tmp_amps[1])
        #     pass
        # tmp_Amp    = np.asarray(tmp_Amp)
        # tmp_Ampbar = np.asarray(tmp_Ampbar)
        # self.uproot_data[f"AmpD0"]    = pd.DataFrame(tmp_Amp,   index=indices)
        # self.uproot_data[f"AmpD0bar"] = pd.DataFrame(tmp_Ampbar,index=indices)
        self.uproot_data[f"AmpD0"]    = tmp_Amp
        self.uproot_data[f"AmpD0bar"] = np.negative(tmp_Ampbar)
        # print("tmp_Ampbar and tmp_Amp")
        # print("tmp_Ampbar and tmp_Amp")
        # print("tmp_Ampbar and tmp_Amp")
        # print("tmp_Ampbar and tmp_Amp")
        # print("tmp_Ampbar and tmp_Amp")
        # print(tmp_Ampbar[0])
        # print(tmp_Amp[0])
        # print(p1bar.tolist())
        # print(p3bar.tolist())
        # print(p2bar.tolist())
        self.uproot_data = self.uproot_data.dropna()
        ############ now some assignments
        self.Bplus_data  = self.uproot_data.query("Bu_ID>0")
        self.Bminus_data = self.uproot_data.query("Bu_ID<0")
        self.Bu_M    = {
            "Bplus" : np.asarray(self.Bplus_data[self.variable_to_fit]),
            "Bminus": np.asarray(self.Bminus_data[self.variable_to_fit]),
            "both"  : np.asarray(self.uproot_data[self.variable_to_fit])
        }
        self.zp_p    = {
            "Bplus" : np.asarray(self.Bplus_data["zp_p"]),
            "Bminus": np.asarray(self.Bminus_data["zp_p"]),
        }
        self.zm_pp    = {
            "Bplus" : np.asarray(self.Bplus_data["zm_pp"]),
            "Bminus": np.asarray(self.Bminus_data["zm_pp"]),
        }
        self.m_Kspip    = {
            "Bplus" : np.asarray(self.Bplus_data["m_Kspip"]),
            "Bminus": np.asarray(self.Bminus_data["m_Kspip"]),
        }
        self.m_Kspim    = {
            "Bplus" : np.asarray(self.Bplus_data["m_Kspim"]),
            "Bminus": np.asarray(self.Bminus_data["m_Kspim"]),
        }
        self.AmpD0    = {
            "Bplus" : np.asarray(self.Bplus_data["AmpD0"]),
            "Bminus": np.asarray(self.Bminus_data["AmpD0"]),
        }
        self.AmpD0bar = {
            "Bplus" : np.asarray(self.Bplus_data["AmpD0bar"]),
            "Bminus": np.asarray(self.Bminus_data["AmpD0bar"]),
        }
        return


    ############## mass PDFs
    def define_mass_pdfs(self):
        self.mass_pdfs = {}
        self.mass_pdfs["both"] = []
        print("Define Dalitz PDFs")
        for comp in self.components:
            self.mass_pdfs["both"].append(MassPDF(comp[1], comp[0], "both"))
            pass
        return

    def initialise_fixed_mass_pdfs(self, _fixed_variables):
        #### first initialise the mass pdfs for the sum of Bplus and Bminus
        # print("Initialising PDFS")
        self.comp_yields           = {}
        self.comp_yields["both"]   = []
        for i_comp in range(len(self.mass_pdfs["both"])):
            # print("   mass pdf for ",self.mass_pdfs["both"][i_comp].component)
            # tf.print("i_comp: ",i_comp)
            self.mass_pdfs["both"][i_comp].get_mass_pdf(
                _fixed_variables[self.i_c][i_comp][4], # last index is space
                Bsign=None
            )
            self.comp_yields["both"].append(_fixed_variables[self.i_c][i_comp][4][INDEX_YIELDS["Bplus"]] + _fixed_variables[self.i_c][i_comp][4][INDEX_YIELDS["Bminus"]])
            pass
        return

    
    # @tf.function
    # def get_mass_pdf_values(self):
    #     Bu_M     = self.Bu_M["both"]
    #     try:
    #         test = self.mass_pdfs
    #     except AttributeError:
    #         print(" ")
    #         print(" ")
    #         print("=================================== ")
    #         print("ERROR IN get_total_pdf_values --------------- ")
    #         print("      DALITZ AND MASS PDFS IN NTUPLE ARE NOT DEFINED ")
    #         print(self.source)
    #         print(self.channel)
    #         print("PLEASE RUN define_mass_pdfs")
    #         print("=================================== ")
    #         print(" ")
    #         print(" ")
    #         return np.zeros(np.shape(Bu_M))
    #     mass_pdf_values = np.zeros(np.shape(Bu_M))
    #     for comp_pdf in self.mass_pdfs["both"]:
    #         mass_pdf_values += comp_pdf.pdf(Bu_M)
    #     return mass_pdf_values # total_mass_pdf_values, total_dalitz_pdf_values

    @tf.function
    def get_mass_pdf_values(self, comp_pdf, Bu_M, comp_yield):
        mass_pdf_values = comp_pdf.pdf(Bu_M)*comp_yield
        return mass_pdf_values # total_mass_pdf_values, total_dalitz_pdf_values

    ##### dalitz * mass pdfs
    # @tf.function
    def get_mass_nll(self, list_variables, tf_sum_yields, gaussian_constraints=[]): # params, fixed_variables, shared_parameters, constrained_parameters
        try:
            total_yield    = {}
            total_yield["both"] = tf.cast(len(self.Bu_M["both"]  ), tf.float64)
        except ValueError:
            print("ERROR -------------- in get_mass_nll for ntuple")
            print(self)
            print("  -- For this to work you need to first store the data in ntuple.uproot_data")
            print("        by running ntuple.store_events()")
            print(" EXIT ")
            print("  ")
            return 0
        #### prepare the data
        Bu_M      = tf.cast(self.Bu_M["both"], tf.float64)
        nevents = {
            "both" : tf.cast(len(self.Bu_M["both"]), tf.float64)
        }
        # list_variables = self.get_list_variables(fixed_variables, params=params, shared_parameters=shared_parameters, constrained_parameters=constrained_parameters)
        # self.initialise_fixed_mass_pdfs(list_variables)
        nll = 0
        ##### getting the yields of Bsign component
        # in the list for the Bplus, second for the Bminus
        # This is defined in VARDICT
        mass_pdf_values = np.zeros(np.shape(Bu_M))
        for comp_pdf, comp_yield in zip(self.mass_pdfs["both"], self.comp_yields["both"]):
            mass_pdf_values += self.get_mass_pdf_values(comp_pdf, Bu_M, comp_yield)
        ######## the first term is the sum of the product of the two pdfs
        # mass_pdf_values = self.get_mass_pdf_values()
        term1 = tf.reduce_sum(-2 * clip_log(mass_pdf_values))
        ### sum the yields of all components and
        # constrain it to the total number of events
        # print(" the yields are:")
        # print([comp[4][index_yields] for comp in list_variables[self.i_c]])
        # sum_yields    = sum([comp[4][index_yields] for comp in list_variables[self.i_c]])
        poisson = tfp.distributions.Poisson(rate=tf_sum_yields)
        log_poisson_constraint = tf.cast(
            poisson.log_prob(total_yield["both"]),
            tf.float64
        )
        term2 = - 2*log_poisson_constraint
        term3 = 2*nevents["both"]*clip_log(tf_sum_yields)
        nll  += term1 + term2 + term3
        # tf.print(" +              gaussian_constraints ", term4)
        # tf.print(" =  nll :                  ", nll)
        # tf.print(" ")
        # tf.print(" ")
        # tf.print(" ")
        return nll
    
    def draw_combined_mass_pdfs(self, np_input, variables):
        self.initialise_fixed_mass_pdfs(variables)
        pdf_values = {}
        pdf_values["both"] = {}
        pdf_values["both"]["total_mass_pdf"] = np.zeros(np.shape(np_input))
        for comp_pdf, comp_yield in zip(self.mass_pdfs["both"], self.comp_yields["both"]):
            pdf_values["both"][comp_pdf.component] = comp_pdf.pdf(np_input)*comp_yield
            pdf_values["both"]["total_mass_pdf"]  += comp_pdf.pdf(np_input)*comp_yield
            pass
        return pdf_values


    def draw_mass_pdfs(self, np_input, variables):
        self.initialise_fixed_pdfs(variables)
        pdf_values = {}
        for Bsign in list(BSIGNS.keys()) + ["both"]:
            pdf_values[Bsign] = {}
            pdf_values[Bsign]["total_mass_pdf"] = np.zeros(np.shape(np_input))
            for comp_pdf, comp_yield in zip(self.mass_pdfs[Bsign], self.comp_yields[Bsign]):
                pdf_values[Bsign][comp_pdf.component] = comp_pdf.pdf(np_input)*comp_yield
                pdf_values[Bsign]["total_mass_pdf"]  += comp_pdf.pdf(np_input)*comp_yield
                pass
            pass
        return pdf_values



    ############## Dalitz PDFs
    def define_dalitz_pdfs(self, norm_ampD0, norm_ampD0bar, norm_zp_p, norm_zm_pp):
        ##### dalitz_components
        self.dalitz_pdfs = {}
        self.dalitz_pdfs["Bplus"]  = []
        self.dalitz_pdfs["Bminus"] = []
        self.mass_pdfs = {}
        self.mass_pdfs["Bplus"]  = []
        self.mass_pdfs["Bminus"] = []
        self.mass_pdfs["both"] = []
        print("Define Dalitz PDFs")
        for comp in self.components:
            self.mass_pdfs["both"].append(MassPDF(comp[1], comp[0], "both"))
            for Bsign in BSIGNS.keys():
                self.mass_pdfs[Bsign].append(MassPDF(comp[1], comp[0], Bsign))
                isSignalDK   = (comp[0] in SIGNAL_COMPONENTS_DK )
                isSignalDPI  = (comp[0] in SIGNAL_COMPONENTS_DPI)
                isSignal     = ((isSignalDK == True) or (isSignalDPI == True))
                # print("")
                # print(comp[0])
                # print("isSignalDK : ", isSignalDK)
                # print("isSignalDPI: ", isSignalDPI)
                # print("isSignal   : ", isSignal)
                self.dalitz_pdfs[Bsign].append(
                    DalitzPDF(
                        comp[2], # function name like "Legendre_2_2" 
                        comp[0], # component name like "DK_Kspipi"
                        Bsign  , # this is the momentum of the kaon no i'm joking
                                 # this is obv the sign of the B
                        isSignal = isSignal # this might be important at some point
                        # I'm not sure yet
                    )
                )
                pass
            pass
        self.norm_ampD0    = norm_ampD0
        self.norm_ampD0bar = norm_ampD0bar
        self.norm_zp_p     = norm_zp_p
        self.norm_zm_pp    = norm_zm_pp
        return

    
    @tf.function
    def get_total_pdf_values(self, mass_pdf, dalitz_pdf, Bu_M, ampD0, ampD0bar, zp_p, zm_pp, comp_yield):
        total_pdf_values = dalitz_pdf.pdf(ampD0, ampD0bar, zp_p, zm_pp)*mass_pdf.pdf(Bu_M)*comp_yield
        # tf.print("total_pdf_values   : ", total_pdf_values)
        # tf.print("dalitz_pdf.pdf(ampD0, ampD0bar, zp_p, zm_pp): ", dalitz_pdf.pdf(ampD0, ampD0bar, zp_p, zm_pp))
        # tf.print("mass_pdf.pdf(Bu_M) : ", mass_pdf.pdf(Bu_M))
        # tf.print("comp_yield         : ", comp_yield)
        # # total_mass_pdf_values["Bplus"]  = np.zeros(np.shape(Bu_M))
        # # total_mass_pdf_values["Bminus"] = np.zeros(np.shape(Bu_M))
        # # for Bsign in BSIGNS.keys():
        # for comp_pdf in self.mass_pdfs[Bsign]:
        #     total_mass_pdf_values += comp_pdf.pdf(Bu_M)
        #     pass
        return total_pdf_values # total_mass_pdf_values, total_dalitz_pdf_values

    ##### dalitz * mass pdfs
    # @tf.function
    def get_total_nll(self, tf_sum_yields, Bsign=""):
        try:
            total_yield    = tf.cast(len(self.Bu_M[Bsign]  ), tf.float64)
        except ValueError:
            print("ERROR -------------- in get_total_nll for ntuple")
            print(self)
            print("  -- For this to work you need to first store the data in ntuple.uproot_data")
            print("        by running ntuple.store_events()")
            print(" EXIT ")
            print("  ")
            return 0
        #### prepare the data
        Bu_M      = tf.cast(self.Bu_M[Bsign]     , tf.float64)
        ampD0     = tf.cast(self.AmpD0[Bsign]    , tf.complex128)
        ampD0bar  = tf.cast(self.AmpD0bar[Bsign] , tf.complex128)
        zp_p      = tf.cast(self.zp_p[Bsign]     , tf.float64)
        zm_pp     = tf.cast(self.zm_pp[Bsign]    , tf.float64)
        nll = 0
        ######## the first term is the sum of the product of the two pdfs
        total_pdf_values = np.zeros(np.shape(ampD0))
        for dalitz_pdf, mass_pdf, comp_yield in zip(self.dalitz_pdfs[Bsign], self.mass_pdfs[Bsign], self.comp_yields[Bsign]):
            # print(dalitz_pdf.name)
            # print(mass_pdf.name)
            # print(comp_yield)
            total_pdf_values += self.get_total_pdf_values(mass_pdf, dalitz_pdf, Bu_M, ampD0, ampD0bar, zp_p, zm_pp, comp_yield)
            pass
        term1 = tf.reduce_sum(-2 * clip_log(total_pdf_values))
        ### sum the yields of all components and
        # constrain it to the total number of events
        # print(" the yields are:")
        # print([comp[4][index_yields] for comp in list_variables[self.i_c]])
        poisson = tfp.distributions.Poisson(rate=tf_sum_yields)
        log_poisson_constraint = tf.cast(
            poisson.log_prob(total_yield),
            tf.float64
        )
        term2 = - 2*log_poisson_constraint
        term3 = 2*total_yield*clip_log(tf_sum_yields)
        nll  += term1 + term2 + term3
        # tf.print(f"sum_yields             {Bsign} ", tf_sum_yields)
        # tf.print(f"total_yield            {Bsign} ", total_yield)
        # tf.print(f"log_poisson_constraint {Bsign} ", log_poisson_constraint)
        # tf.print("                     term sum_events     ", term1)
        # tf.print(" -          2*log_poisson_constraint     ", term2)
        # tf.print(" + 2*total_yield*clip_log(tf_sum_yields) ", term3)
        # tf.print("                               = nll     ", nll)
        # tf.print(" ")
        return nll

    
    # ##### dalitz * mass pdfs
    # # @tf.function
    # def get_total_nll(self, params, fixed_variables, shared_parameters, constrained_parameters, gaussian_constraints=[]):
    #     try:
    #         total_yield    = {}
    #         total_yield["Bplus"]    = tf.cast(len(self.Bu_M["Bplus"]  ), tf.float64)
    #         total_yield["Bminus"]   = tf.cast(len(self.Bu_M["Bminus"] ), tf.float64)
    #     except ValueError:
    #         print("ERROR -------------- in get_total_nll for ntuple")
    #         print(self)
    #         print("  -- For this to work you need to first store the data in ntuple.uproot_data")
    #         print("        by running ntuple.store_events()")
    #         print(" EXIT ")
    #         print("  ")
    #         return 0
    #     #### prepare the data
    #     Bu_M_Bplus      = tf.cast(self.Bu_M["Bplus"]     , tf.float64)
    #     Bu_M_Bminus     = tf.cast(self.Bu_M["Bminus"]    , tf.float64)
    #     ampD0_Bplus     = tf.cast(self.AmpD0["Bplus"]    , tf.complex128)
    #     ampD0_Bminus    = tf.cast(self.AmpD0["Bminus"]   , tf.complex128)
    #     ampD0bar_Bplus  = tf.cast(self.AmpD0bar["Bplus"] , tf.complex128)
    #     ampD0bar_Bminus = tf.cast(self.AmpD0bar["Bminus"], tf.complex128)
    #     nevents = {
    #         "Bplus" : tf.cast(len(self.Bu_M["Bplus"]), tf.float64),
    #         "Bminus": tf.cast(len(self.Bu_M["Bminus"]), tf.float64)
    #     }
    #     list_variables = self.get_list_variables(fixed_variables, params=params, shared_parameters=shared_parameters, constrained_parameters=constrained_parameters)
    #     self.initialise_fixed_pdfs(list_variables)
    #     nll = 0
    #     for Bsign in BSIGNS.keys():
    #         ##### getting the yields of Bsign component
    #         # in the list for the Bplus, second for the Bminus
    #         # This is defined in VARDICT
    #         index_yields  = INDEX_YIELDS[Bsign]
    #         ######## the first term is the sum of the product of the two pdfs
    #         total_pdf_values = self.get_total_pdf_values(
    #             Bsign
    #         )
    #         term1 = tf.reduce_sum(-2 * clip_log(total_pdf_values))
    #         ### sum the yields of all components and
    #         # constrain it to the total number of events
    #         # print(" the yields are:")
    #         # print([comp[4][index_yields] for comp in list_variables[self.i_c]])
    #         sum_yields    = sum([comp[4][index_yields] for comp in list_variables[self.i_c]])
    #         tf_sum_yields    = tf.cast(sum_yields, tf.float64)
    #         poisson = tfp.distributions.Poisson(rate=tf_sum_yields)
    #         log_poisson_constraint = tf.cast(
    #             poisson.log_prob(total_yield[Bsign]),
    #             tf.float64
    #         )
    #         term2 = - 2*log_poisson_constraint
    #         term3 = 2*nevents[Bsign]*clip_log(tf_sum_yields)
    #         nll  += term1 + term2 + term3
    #         # tf.print(" ntuple: ", self.tex)
    #         # tf.print(f"              shared variables ", list_variables[-1][0][4])
    #         # tf.print(f"sum_yields             {Bsign} ", tf_sum_yields)
    #         # tf.print(f"total_yield            {Bsign} ", total_yield)
    #         # tf.print(f"log_poisson_constraint {Bsign} ", log_poisson_constraint)
    #         # tf.print("                     term sum_events ", term1)
    #         # tf.print(" -          2*log_poisson_constraint ", term2)
    #         # tf.print(" + 2*nevents*clip_log(tf_sum_yields) ", term3)
    #         # tf.print("                               = nll ", nll)
    #         # tf.print(" ")
    #         pass
    #     term4 = self.get_gaussian_constraints(gaussian_constraints, list_variables)
    #     nll  += term4
    #     # tf.print(" +              gaussian_constraints ", term4)
    #     # tf.print(" =  nll :                  ", nll)
    #     # tf.print(" ")
    #     # tf.print(" ")
    #     # tf.print(" ")
    #     return nll
        


        
        

        
class Variables:
    def __init__(self, name, tex, scale, range_value):
        self.name  = name
        self.scale = scale
        self.tex   = tex
        self.range_value = range_value

DICT_VARIABLES_TEX = {
    'Bu_constD0KSPV_M':          Variables('Bu_constD0KSPV_M'      , '$m(DK^\pm)$ [MeV/$c^2$] DTF D0-KS-PV const', 'linear',  [4500,7000]),
    'Bu_constD0PV_M':            Variables('Bu_constD0PV_M'        , '$m(DK^\pm)$ [MeV/$c^2$] DTF D0-PV const'  , 'linear',   [4500,7000]),
    'Bu_constKSPV_M':            Variables('Bu_constKSPV_M'        , '$m(DK^\pm)$ [MeV/$c^2$] DTF KS-PV const'  , 'linear',   [4500,7000]),
    'Bu_constD0KSPV_status':     Variables('Bu_constD0KSPV_status' , 'Status DTF D0KSPV', 'linear', [0,1]),
    'Bu_constD0PV_status':       Variables('Bu_constD0PV_status'   , 'Status DTF D0PV'  , 'linear', [0,1]),
    'Bu_constKSPV_status':       Variables('Bu_constKSPV_status'   , 'Status DTF KSPV'  , 'linear', [0,1]),
    'Bu_M':                      Variables('Bu_M'   , '$m(B)$ [MeV/$c^2$]', 'linear', [4500,7000]),
    "m_Kspip_True" : Variables("m_Kspip_True" ,"True $m(K_S\pi^+)$ [GeV$^2$]"  , 'linear', [0,3]),
    "m_Kspim_True" : Variables("m_Kspim_True" ,"True $m(K_S\pi^-)$ [GeV$^2$]"  , 'linear', [0,3]),
    "m_pippim_True": Variables("m_pippim_True","True $m(\pi^+\pi^-)$ [GeV$^2$]", 'linear', [0,3]),
    "m_Kspip" : Variables("m_Kspip" ,"$m(K_S\pi^+)$ [GeV$^2$]"  , 'linear', [0,3]),
    "m_Kspim" : Variables("m_Kspim" ,"$m(K_S\pi^-)$ [GeV$^2$]"  , 'linear', [0,3]),
    "m_pippim": Variables("m_pippim","$m(\pi^+\pi^-)$ [GeV$^2$]", 'linear', [0,3]),
    "zp_p" :Variables("zp_p"  ,r"$z_{+}^{\prime}$", 'linear', [-1,1]),
    "zm_pp":Variables("zm_pp" ,r"$z_{-}^{\prime\prime}$", 'linear', [-1,1]),
    "min_Ksh1h2_IPCHI2_OWNPV": Variables("min_Ksh1h2_IPCHI2_OWNPV",r"min(IPCHI2_OWNPV Ks(h1/h2) )", 'log', [0,4000]),
    "max_Ksh1h2_IPCHI2_OWNPV": Variables("max_Ksh1h2_IPCHI2_OWNPV",r"max(IPCHI2_OWNPV Ks(h1/h2) )", 'log', [0,5000] ),
    "min_h1h2_IPCHI2_OWNPV"  : Variables("min_h1h2_IPCHI2_OWNPV"  ,r"min(IPCHI2_OWNPV h1/h2)", 'log', [0,50000] ),
    "max_h1h2_IPCHI2_OWNPV"  : Variables("max_h1h2_IPCHI2_OWNPV"  ,r"max(IPCHI2_OWNPV h1/h2)", 'log', [0,180000]),
    "BD_ZSIG"                : Variables("BD_ZSIG"                ,r"$B/D$ $z$ significance" , 'log', [-50,200]),
    "Bu_constKSPV_D0_M"      : Variables("Bu_constKSPV_D0_M"      ,r"$m(D^{0})$ with $K_S/PV$ constraint" , 'linear', [1700,2000]),
    "Bu_P"             : Variables("Bu_P"              ,r"$B$ momentum [MeV/$c^2$]" , 'linear', [0,1000000])    ,
    "Bu_PT"            : Variables("Bu_PT"             ,r"$B$ $p_T$ [MeV/$c^2$]"    , 'linear', [0,60000])     ,
    "Bu_TAU"           : Variables("Bu_TAU"            ,r"$B$ lifetime [ps]" , 'linear', [0,0.07])      ,
    "Bu_FD_OWNPV"      : Variables("Bu_FD_OWNPV"       ,r"$B$ Flight distance [mm]" , 'log', [0,400])           ,
    "D0_P"             : Variables("D0_P"              ,r"$D^0$ momentum [MeV/$c^2$]" , 'linear', [0,800000])    ,
    "D0_PT"            : Variables("D0_PT"             ,r"$D^0$ $p_T$ [MeV/$c^2$]"    , 'linear', [0,40000])     ,
    "Ks_P"             : Variables("Ks_P"              ,r"$K_S$ momentum [MeV/$c^2$]" , 'linear', [0,300000])    ,
    "Ks_PT"            : Variables("Ks_PT"             ,r"$K_S$ $p_T$ [MeV/$c^2$]"    , 'linear', [0,30000])     ,
    "h1_P"             : Variables("h1_P"              ,r"$\pi^{(1)}|_D$ momentum [MeV/$c^2$]" , 'linear', [0,500000])    ,
    "h1_PT"            : Variables("h1_PT"             ,r"$\pi^{(1)}|_D$ $p_T$ [MeV/$c^2$]"    , 'linear', [0,17500] )     ,
    "h2_P"             : Variables("h2_P"              ,r"$\pi^{(2)}|_D$ momentum [MeV/$c^2$]" , 'linear', [0,500000])    ,
    "h2_PT"            : Variables("h2_PT"             ,r"$\pi^{(2)}|_D$ $p_T$ [MeV/$c^2$]"    , 'linear', [0,17500] )     ,
    "Ksh1_P"           : Variables("Ksh1_P"            ,r"$\pi^{(1)}|_K$ momentum [MeV/$c^2$]" , 'linear', [0,250000])      ,
    "Ksh1_PT"          : Variables("Ksh1_PT"           ,r"$\pi^{(1)}|_K$ $p_T$ [MeV/$c^2$]"    , 'linear', [0,20000])       ,
    "Ksh2_P"           : Variables("Ksh2_P"            ,r"$\pi^{(2)}|_K$ momentum [MeV/$c^2$]" , 'linear', [0,250000])      ,
    "Ksh2_PT"          : Variables("Ksh2_PT"           ,r"$\pi^{(2)}|_K$ $p_T$ [MeV/$c^2$]"    , 'linear', [0,20000])       ,
    "Bach_P"           : Variables("Bach_P"            ,r"Bachelor momentum [MeV/$c^2$]" , 'linear', [0,1200000])      ,
    "Bach_PT"          : Variables("Bach_PT"           ,r"Bachelor $p_T$ [MeV/$c^2$]"    , 'linear', [0,40000])       ,
    "log_min_h1h2_IPCHI2_OWNPV"  :    Variables("log_min_h1h2_IPCHI2_OWNPV"   , "log_min_h1h2_IPCHI2_OWNPV"  , 'linear', None),
    "log_max_h1h2_IPCHI2_OWNPV"  :    Variables("log_max_h1h2_IPCHI2_OWNPV"   , "log_max_h1h2_IPCHI2_OWNPV"  , 'linear', None),
    "log_Bach_IPCHI2_OWNPV"      :    Variables("log_Bach_IPCHI2_OWNPV"       , "log_Bach_IPCHI2_OWNPV"      , 'linear', None),
    "log_min_Ksh1h2_IPCHI2_OWNPV":    Variables("log_min_Ksh1h2_IPCHI2_OWNPV" , "log_min_Ksh1h2_IPCHI2_OWNPV", 'linear', None),
    "log_max_Ksh1h2_IPCHI2_OWNPV":    Variables("log_max_Ksh1h2_IPCHI2_OWNPV" , "log_max_Ksh1h2_IPCHI2_OWNPV", 'linear', None),
    "log_D0_IPCHI2_OWNPV"        :    Variables("log_D0_IPCHI2_OWNPV"         , "log_D0_IPCHI2_OWNPV"        , 'linear', None),
    "log_Bu_IPCHI2_OWNPV"        :    Variables("log_Bu_IPCHI2_OWNPV"         , "log_Bu_IPCHI2_OWNPV"        , 'linear', None),
    "log_Bu_FDCHI2_OWNPV"        :    Variables("log_Bu_FDCHI2_OWNPV"         , "log_Bu_FDCHI2_OWNPV"        , 'linear', None),
    "log_Bu_P"                   :    Variables("log_Bu_P"                    , "log_Bu_P"                   , 'linear', None),
    "log_Bu_PT"                  :    Variables("log_Bu_PT"                   , "log_Bu_PT"                  , 'linear', None),
    "log_D0_P"                   :    Variables("log_D0_P"                    , "log_D0_P"                   , 'linear', None),
    "log_D0_PT"                  :    Variables("log_D0_PT"                   , "log_D0_PT"                  , 'linear', None),
    "log_Bach_PT"                :    Variables("log_Bach_PT"                 , "log_Bach_PT"                , 'linear', None),
    "log_Bach_P"                 :    Variables("log_Bach_P"                  , "log_Bach_P"                 , 'linear', None),
    "log_Bu_RHO_BPV"             :    Variables("log_Bu_RHO_BPV"              , "log_Bu_RHO_BPV"             , 'linear', None),
    "log_D0_RHO_BPV"             :    Variables("log_D0_RHO_BPV"              , "log_D0_RHO_BPV"             , 'linear', None),
    "Bu_MAXDOCA"                 :    Variables("Bu_MAXDOCA"                  , "Bu_MAXDOCA"                 , 'linear', None),
    "D0_MAXDOCA"                 :    Variables("D0_MAXDOCA"                  , "D0_MAXDOCA"                 , 'linear', None),
    "Bu_PTASY_1_5"               :    Variables("Bu_PTASY_1_5"                , "Bu_PTASY_1_5"               , 'linear', None),
    "log_D0_VTXCHI2DOF"          :    Variables("log_D0_VTXCHI2DOF"           , "log_D0_VTXCHI2DOF"          , 'linear', None),
    "log_Ks_VTXCHI2DOF"          :    Variables("log_Ks_VTXCHI2DOF"           , "log_Ks_VTXCHI2DOF"          , 'linear', None),
    "log_Bu_VTXCHI2DOF"          :    Variables("log_Bu_VTXCHI2DOF"           , "log_Bu_VTXCHI2DOF"          , 'linear', None),
    "log10_1_minus_Bu_DIRA_BPV"  :    Variables("log10_1_minus_Bu_DIRA_BPV"   , "log10_1_minus_Bu_DIRA_BPV"  , 'linear', None),
    "log10_1_minus_D0_DIRA_BPV"  :    Variables("log10_1_minus_D0_DIRA_BPV"   , "log10_1_minus_D0_DIRA_BPV"  , 'linear', None),
    "log10_1_minus_Ks_DIRA_BPV"  :    Variables("log10_1_minus_Ks_DIRA_BPV"   , "log10_1_minus_Ks_DIRA_BPV"  , 'linear', None),
    "log_Bu_constD0KSPV_CHI2NDOF":    Variables("log_Bu_constD0KSPV_CHI2NDOF" , "log_Bu_constD0KSPV_CHI2NDOF", 'linear', None),    
    'Bu_constD0KSPV_swapBachToPi_M':          Variables('Bu_constD0KSPV_swapBachToPi_M'      , r'$m(D\pi^\pm)$ [MeV/$c^2$] DTF D0-KS-PV const', 'linear',  [5050,5800]),
    'Bu_constD0KSPV_swapBachToPi_status':     Variables('Bu_constD0KSPV_swapBachToPi_status' , r'Status DTF D0KSPV ($D\pi$ hyp.)', 'linear', [0,1]),
    'Bu_constD0KSPV_swapBachToK_M':          Variables('Bu_constD0KSPV_swapBachToK_M'        , r'$m(DK^\pm)$ [MeV/$c^2$] DTF D0-KS-PV const', 'linear',  [5050,5800]),
    'Bu_constD0KSPV_swapBachToK_status':     Variables('Bu_constD0KSPV_swapBachToK_status'   , r'Status DTF D0KSPV ($DK$ hyp.)', 'linear', [0,1]),
    'Ks_FDCHI2_ORIVX'            :    Variables("Ks_FDCHI2_ORIVX"             , r"$K_S$ $\chi^{2}_{FD}$"     , 'linear', None) ,
    'h1_hasRich'                 :    Variables("h1_hasRich"                  , r"$h^1_D$ has RICH signal"   , 'linear', [0,1]),
    'h2_hasRich'                 :    Variables("h2_hasRich"                  , r"$h^2_D$ has RICH signal"   , 'linear', [0,1]),
    'h1_PIDK'                    :    Variables("h1_PIDK"                     , r"$h^1_D$ PID($K$)"          , 'linear', None) , 
    'h2_PIDK'                    :    Variables("h2_PIDK"                     , r"$h^2_D$ PID($K$)"          , 'linear', None) ,
}

zp_p_tex  = DICT_VARIABLES_TEX["zp_p"].tex
zm_pp_tex = DICT_VARIABLES_TEX["zm_pp"].tex
