#ifndef NTUPLES_H 
#define NTUPLES_H 1

## Dictionary of ntuples for this analysis
##
## Dictionary of files
## File names are hardcoded so that we can realize if there are missing files
## Do not change this to a dynamic listing

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "TString.h"
#include "TChain.h"
#include "common_types.h"

## Type of data dictionary
## Example data.at(D20120.at(MagDown) is a list (vector) of strings with files names
## Warning: at() is mandatory since we defined things with const's
# typedef std::map < Year, std::map < Magnet, std::vector<TString> > > data_dictionary;
# typedef std::map < Channel,
# 		   std::map < Year,
# 			      std::map < Magnet,
# 					 std::vector<TString> > > > Realdata_dictionary;


## eos access
prefix_mc = "/eos/lhcb/wg/b2oc/GGSZ-MI_Bu2Dh_Run12/mc_post_stripping/"  ;            # TString 
prefix_data_strip = "/eos/lhcb/wg/b2oc/GGSZ-MI_Bu2Dh_Run12/data_post_stripping/"  ;  # TString 
prefix_data_selec = "/eos/lhcb/wg/b2oc/GGSZ-MI_Bu2Dh_Run12/data_final_selection/"  ; # TString 

real_data_post_selection = {
    "CB2DPi_D2KsPiPi_DD":
    {
	"Y2011" :
        {
            "MagAll":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_11_*.root"],
            "MagDown":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_11_Down.root"],
	    "MagUp" :   [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_11_Up.root"],
	},
	"Y2012" :
        {
            "MagAll":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_12_*.root"],
            "MagDown":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_12_Down.root"],
	    "MagUp" :   [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_12_Up.root"],
	},
	"Y2015" :
        {
            "MagAll":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_15_*.root"],
            "MagDown":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_15_Down.root"],
	    "MagUp" :   [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_15_Up.root"],
	},
	"Y2016" :
        {
            "MagAll":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_16_*.root"],
            "MagDown":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_16_Down.root"],
	    "MagUp" :   [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_16_Up.root"],
	},
	"Y2017" :
        {
            "MagAll":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_17_*.root"],
            "MagDown":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_17_Down.root"],
	    "MagUp" :   [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_17_Up.root"],
	},
	"Y2018" :
        {
            "MagAll":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_18_*.root"],
            "MagDown":  [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_18_Down.root"],
	    "MagUp" :   [ prefix_data_selec + "Bu2DPi_D2KsPiPi_DD_18_Up.root"],
	},
    },
}


real_data_pre_selection = { 
    "Y2011":
    {
        "MagAll" :  [ prefix_data_strip + "Bu2Dh_D2Kshh_11_*.root"],
        "MagDown":  [ prefix_data_strip + "Bu2Dh_D2Kshh_11_Down.root"],
        "MagUp"  :  [ prefix_data_strip + "Bu2Dh_D2Kshh_11_Up.root"],
    },
    "Y2012" :
    {
        "MagAll" : [ prefix_data_strip + "Bu2Dh_D2Kshh_12_*.root"],
        "MagDown" : [ prefix_data_strip + "Bu2Dh_D2Kshh_12_Down.root"],
	"MagUp"   : [ prefix_data_strip + "Bu2Dh_D2Kshh_12_Up.root"],
    },
    "Y2015":
    {
        "MagAll":  [ prefix_data_strip + "Bu2Dh_D2Kshh_15_*.root"],
        "MagDown":  [ prefix_data_strip + "Bu2Dh_D2Kshh_15_Down.root"],
        "MagUp"  :  [ prefix_data_strip + "Bu2Dh_D2Kshh_15_Up.root"],
    },
    "Y2016":
    {
        "MagAll":  [ prefix_data_strip + "Bu2Dh_D2Kshh_16_*.root"],
        "MagDown":  [ prefix_data_strip + "Bu2Dh_D2Kshh_16_Down.root"],
        "MagUp"  :  [ prefix_data_strip + "Bu2Dh_D2Kshh_16_Up.root"],
    },
    "Y2017":
    {
        "MagAll":  [ prefix_data_strip + "Bu2Dh_D2Kshh_17_*.root"],
        "MagDown":  [ prefix_data_strip + "Bu2Dh_D2Kshh_17_Down.root"],
        "MagUp"  :  [ prefix_data_strip + "Bu2Dh_D2Kshh_17_Up.root"],
    },
    "Y2018":
    {
        "MagAll":  [ prefix_data_strip + "Bu2Dh_D2Kshh_18_*.root"],
        "MagDown":  [ prefix_data_strip + "Bu2Dh_D2Kshh_18_Down.root"],
        "MagUp"  :  [ prefix_data_strip + "Bu2Dh_D2Kshh_18_Up.root"],
    },
}

mc_data_MC_Bu_D0pi_KSpipi_DecProdCut = { # data_dictionary 
    "Y2015": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165122_15_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165122_15_Down.root"],
	"MagUp":    [ prefix_mc + "Bu2Dh_D2Kshh_12165122_15_Up.root"]  ,
    },
    "Y2016": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165122_16_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165122_16_Down.root"],
	"MagUp":    [ prefix_mc + "Bu2Dh_D2Kshh_12165122_16_Up.root"]  ,
    },
}

mc_data_MC_Bu_D0pi_KSpipi_TightCut = { #data_dictionary 
    "Y2011": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165151_11_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165151_11_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165151_11_Up.root"],
    },
    "Y2012": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165151_12_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165151_12_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165151_12_Up.root"],
    },
}

mc_data_MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow = { # data_dictionary 
    "Y2012": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_12_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_12_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_12_Up.root"]  ,
    },
    "Y2017": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_17_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_17_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_17_Up.root"]  ,
    },
    "Y2018": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_18_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_18_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165146_18_Up.root"]  ,
    }
}

mc_data_MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow = { # data_dictionary 
    "Y2012": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_12_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_12_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_12_Up.root"]  ,
    },
    "Y2017": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_17_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_17_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_17_Up.root"]  ,
    },
    "Y2018": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_18_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_18_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165147_18_Up.root"]  ,
    },
}

mc_data_MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow = {
    "Y2012": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_12_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_12_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_12_Up.root"]  ,
    },
    "Y2017": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_17_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_17_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_17_Up.root"]  ,
    },
    "Y2018": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_18_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_18_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165148_18_Up.root"]  ,
    },    
}

mc_data_MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow = {
    "Y2012": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_12_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_12_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_12_Up.root"]  ,
    },
    "Y2017": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_17_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_17_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_17_Up.root"]  ,
    },
    "Y2018": {
        "MagAll":  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_18_*.root"],
        "MagDown":  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_18_Down.root"],
        "MagUp"  :  [ prefix_mc + "Bu2Dh_D2Kshh_12165149_18_Up.root"]  ,
    },
}


SOURCE_LOCATIONS =  { # std::map < Source, data_dictionary > 
    "SDATA": real_data_pre_selection,
    "MC_Bu_D0pi_KSpipi_DecProdCut":  mc_data_MC_Bu_D0pi_KSpipi_DecProdCut,
    "MC_Bu_D0pi_KSpipi_TightCut":  mc_data_MC_Bu_D0pi_KSpipi_TightCut,
    "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow":  mc_data_MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow,
    "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow":   mc_data_MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow,
    "MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow":  mc_data_MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow,
    "MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow":   mc_data_MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow,
}


#endif 
