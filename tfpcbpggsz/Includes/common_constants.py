### inputs
GAMMA      = 68.7
RB_DK      = 0.0904
DELTAB_DK  = 118.3
RB_DPI     = 0.005
DELTAB_DPI = 291.0

## PDG constants
PDG_m_pi = 0.13957039 # ,0.00018);
PDG_m_Ks = 0.497611   # ,0.013);
PDG_m_Dz = 1.86484   # ,0.05);

## QMI method
QMI_smax_Kspi = (PDG_m_Dz-PDG_m_pi)**2
QMI_smin_Kspi = (PDG_m_Ks+PDG_m_pi)**2
QMI_zpmax_Kspi =  3.686290 # ;// GeV2
QMI_zpmin_Kspi =  1.894890 # ;// GeV2
QMI_zmmax_Kspi =  2.485740 # ;// GeV2
QMI_zmmin_Kspi = -2.485740 # ;// GeV2

# std::map<int, int> dict_poly_order = {
#   {0,0},
#   {1,2},
#   {2,4},
#   {3,6},
#   {4,9},
#   {5,12},
#   {6,16},
# };


### This order is hardcoded in the way the function PhaseCorrection is defined in functions.h
# std::map<int, std::vector<TString> > dict_name_coeff = {
#   {0, {""} },
#   {1, {"C00","C10"} },
#   {2, {"C00","C10","C20","C02"} },
#   {3, {"C00","C10","C20","C02","C30","C12"} },
#   {4, {"C00","C10","C20","C02","C30","C12","C40","C22","C04"} },
#   {5, {"C00","C10","C20","C02","C30","C12","C40","C22","C04","C50","C32","C14"} },
#   {6, {"C00","C10","C20","C02","C30","C12","C40","C22","C04","C50","C32","C14","C06","C24","C42","C60"} },
# };

#endif 

## paths to folders
pathname_storage = "/eos/lhcb/user/j/jocottee/Gamma_measurement/" # - usually eos, tuples stored here
pathname_local = "/afs/cern.ch/work/j/jocottee/private/camilles_pipeline/" # - local filepath, probably where you have cloned the repo, for plots etc
pathname_storage = "/dice/users/rj23972/safety_net/Gamma_measurement/" # - usually eos, tuples stored here
pathname_local = "/software/rj23972/safety_net/tfpcbpggsz/tfpcbpggsz/" # - local filepath, probably where you have cloned the repo, for plots etc


################## Fit stuff
COMPONENTS = {
    "SDATA":
    {
        "CB2DK_D2KSPIPI_DD": [ 
            ["DK_Kspipi", "Cruijff+Gaussian"],
            ["Dpi_Kspipi_misID", "SumCBShape"],
            # ["Dst0K_D0pi0_Kspipi", "HORNSdini"],
            # ["DstpK_D0pip_Kspipi", "HORNSdini"],
            # ["Dst0pi_D0pi0_Kspipi_misID_PartReco", "HORNSdini_misID"],
            # ["Dstppi_D0pip_Kspipi_misID_PartReco", "HORNSdini_misID"],
            # ["Dst0K_D0gamma_Kspipi", "HILLdini"],
            # ["Dst0pi_D0gamma_Kspipi_misID_PartReco", "HILLdini_misID"],
            # ["DKpi_Kspipi", "HORNSdini+Gaussian"],
            # ["Dpipi_Kspipi_misID_PartReco", "HORNSdini_misID"],
            # ["Bs2DKpi_Kspipi_PartReco", "HORNSdini"],
            # ["Combinatorial", "Exponential"],
        ],
        "CB2DPI_D2KSPIPI_DD": [ 
            ["Dpi_Kspipi", "Cruijff+Gaussian"],
            ["DK_Kspipi_misID", "SumCBShape"]
        ]
    },
    "MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow":
    {
        "CB2DK_D2KSPIPI_DD": [
            ["DK_Kspipi", "Cruijff+Gaussian"],
        ],
        "CB2Dpi_D2KSPIPI_DD": [
            ["DK_Kspipi_misID", "SumCBShape"],
        ],
    },
    "MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow":
    {
        "CB2DPi_D2KSPIPI_DD": [
            ["Dpi_Kspipi", "Cruijff+Gaussian"],
        ],
        "CB2DK_D2KSPIPI_DD": [
            ["Dpi_Kspipi_misID", "SumCBShape"],
        ],
    }
}

VARIABLE_TO_FIT = {
    "CB2DK_D2KSPIPI_DD" : "Bu_constD0KSPV_M",
    "CB2DPI_D2KSPIPI_DD": "Bu_constD0KSPV_swapBachToPi_M",
}

INDEX_SOURCE_TO_VARDICT = {
    "SDATA" : 0,
}

INDEX_CHANNEL_TO_VARDICT = {
    "CB2DK_D2KSPIPI_DD" : 0,
    "CB2DPI_D2KSPIPI_DD": 1,
}

SIGNAL_COMPONENTS_DK  = ["DK_Kspipi", "DK_Kspipi_misID"]
SIGNAL_COMPONENTS_DPI = ["Dpi_Kspipi_misID", "Dpi_Kspipi"]

BSIGNS = {
    "Bplus"  :  1,
    "Bminus" : -1
}


##### the parameters that are shared accross channels (CP obs, ratio of yields etc)
# are the last column of our long list of parameters
INDEX_SHARED_THROUGH_CHANNELS = -1
INDEX_YIELDS = {
    "Bplus"  : 0,
    "Bminus" : 1
}


#### efficiency shapes
EFFICIENCY_SHAPES = ["Flat", "Legendre_2_2", "Legendre_5_5"]


### This order is hardcoded in the way the functions Legendre_ZP_ZM are defined in functions.py
DICT_NAME_COEFF = {
    "Flat": [],
    "Legendre_2_2": ["c00","c10","c20","c01", "c02", "c11", "c21", "c12", "c22"],
    "Legendre_5_5": ["c00","c10","c20","c01", "c02", "c11", "c21", "c12", "c22", "c30", "c40", "c50", "c03", "c04", "c05", "c24", "c32", "c34", "c42", "c44", "c52", "c54"], # I dropped some odd powers of zm because they quantify pi+/pi- asymmetry so I don't think we'll need to go as far as we do for zp
}

