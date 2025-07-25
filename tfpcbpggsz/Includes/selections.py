# CUT_CLEANING_DATA = "(Bu_constD0KSPV_status == 0 ) & (Bu_constD0PV_status == 0) & (Bu_constKSPV_status == 0)"
CUT_CLEANING_DATA = "(Bu_constD0KSPV_status == 0 ) & (Bu_constD0PV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0KSPV_swapBachToPi_status == 0) & (Bu_constD0KSPV_swapBachToK_status == 0) & (h1_IPCHI2_OWNPV>0) & (h2_IPCHI2_OWNPV>0) & (Bach_IPCHI2_OWNPV>0) & (Ksh1_IPCHI2_OWNPV>0) & (Ksh2_IPCHI2_OWNPV>0) & (D0_IPCHI2_OWNPV>0) & (Bu_IPCHI2_OWNPV>0) & (Bu_FDCHI2_OWNPV>0) & (Bu_P>0) & (Bu_PT>0) & (D0_P>0) & (D0_PT>0) & (Bach_PT>0) & (Bach_P>0) & (Bu_RHO_BPV>0) & (D0_RHO_BPV>0) & (D0_VTXCHI2DOF>0) & (Ks_VTXCHI2DOF>0) & (Bu_VTXCHI2DOF>0) & (Bu_constD0KSPV_chi2>0)"


TRIGGER_RUN1 = "( (Bu_L0Global_TIS==True) | (Bu_L0HadronDecision_TOS==True) ) & (Bu_Hlt1TrackAllL0Decision_TOS == True ) & ( (Bu_Hlt2Topo2BodyBBDTDecision_TOS==True) | (Bu_Hlt2Topo3BodyBBDTDecision_TOS==True) | (Bu_Hlt2Topo4BodyBBDTDecision_TOS==True) )"
TRIGGER_RUN2 = "( (Bu_L0Global_TIS==True) | (Bu_L0HadronDecision_TOS==True) ) & ( (Bu_Hlt1TrackMVADecision_TOS==True)  | (Bu_Hlt1TwoTrackMVADecision_TOS==True) ) & ( (Bu_Hlt2Topo2BodyDecision_TOS==True) | (Bu_Hlt2Topo3BodyDecision_TOS==True) | (Bu_Hlt2Topo4BodyDecision_TOS==True) )"
MAP_TRIGGER = { # std::map<Year, TCut> 
    "YRUN1": TRIGGER_RUN1,
    "Y2011": TRIGGER_RUN1,
    "Y2012": TRIGGER_RUN1,
    "YRUN2": TRIGGER_RUN2,
    "Y2015": TRIGGER_RUN2,
    "Y2016": TRIGGER_RUN2,
    "Y2017": TRIGGER_RUN2,
    "Y2018": TRIGGER_RUN2,
}

CUT_PROBNN_DCHILDREN_PIPI_RUN1 = "(h1_PIDK < 20) & (h2_PIDK < 20)" #  to optimise
CUT_PROBNN_DCHILDREN_PIPI_RUN2 = "(h1_PIDK < 20) & (h2_PIDK < 20)" #  to optimise
CUT_PROBNN_DCHILDREN_KK_RUN1 = "(h1_PIDK > -5) & (h2_PIDK > -5) & (h1_hasRich == 1) & (h2_hasRich == 1)" 
CUT_PROBNN_DCHILDREN_KK_RUN2 = "(h1_PIDK > -5) & (h2_PIDK > -5) & (h1_hasRich == 1) & (h2_hasRich == 1)" 


MAP_PROBNN_DCHILDREN_RUN1 = { # std::map<Channel,TCut> 
    "CB2DK_D2KSPIPI_DD" : CUT_PROBNN_DCHILDREN_PIPI_RUN1,
    "CB2DPI_D2KSPIPI_DD": CUT_PROBNN_DCHILDREN_PIPI_RUN1,
    "CB2DK_D2KSPIPI_LL" : CUT_PROBNN_DCHILDREN_PIPI_RUN1,
    "CB2DPI_D2KSPIPI_LL": CUT_PROBNN_DCHILDREN_PIPI_RUN1,
    "CB2DK_D2KSKK_DD"   : CUT_PROBNN_DCHILDREN_KK_RUN1  ,
    "CB2DPI_D2KSKK_DD"  : CUT_PROBNN_DCHILDREN_KK_RUN1  ,
    "CB2DK_D2KSKK_LL"   : CUT_PROBNN_DCHILDREN_KK_RUN1  ,
    "CB2DPI_D2KSKK_LL"  : CUT_PROBNN_DCHILDREN_KK_RUN1  ,
}
MAP_PROBNN_DCHILDREN_RUN2 = { # std::map<Channel,TCut> 
    "CB2DK_D2KSPIPI_DD" : CUT_PROBNN_DCHILDREN_PIPI_RUN2,
    "CB2DPI_D2KSPIPI_DD": CUT_PROBNN_DCHILDREN_PIPI_RUN2,
    "CB2DK_D2KSPIPI_LL" : CUT_PROBNN_DCHILDREN_PIPI_RUN2,
    "CB2DPI_D2KSPIPI_LL": CUT_PROBNN_DCHILDREN_PIPI_RUN2,
    "CB2DK_D2KSKK_DD"   : CUT_PROBNN_DCHILDREN_KK_RUN2  ,
    "CB2DPI_D2KSKK_DD"  : CUT_PROBNN_DCHILDREN_KK_RUN2  ,
    "CB2DK_D2KSKK_LL"   : CUT_PROBNN_DCHILDREN_KK_RUN2  ,
    "CB2DPI_D2KSKK_LL"  : CUT_PROBNN_DCHILDREN_KK_RUN2  ,
    }
MAP_PROBNN_DCHILDREN = { # std::map<Year, std::map<Channel,TCut> > 
    "YRUN1": MAP_PROBNN_DCHILDREN_RUN1,
    "Y2011": MAP_PROBNN_DCHILDREN_RUN1,
    "Y2012": MAP_PROBNN_DCHILDREN_RUN1,
    "YRUN2": MAP_PROBNN_DCHILDREN_RUN2,
    "Y2015": MAP_PROBNN_DCHILDREN_RUN2,
    "Y2016": MAP_PROBNN_DCHILDREN_RUN2,
    "Y2017": MAP_PROBNN_DCHILDREN_RUN2,
    "Y2018": MAP_PROBNN_DCHILDREN_RUN2,
}

### removed 28/08/2024 as was used in testing, and not currently being used but to ensure not accidentally implemented again, superceeded by the PIDK cuts used for 05_Final_cuts.py
# #### optimised in cpp/canorman_Selections/2024_06_28/Optimising_PID_Bach.cxx
# MAP_PROBNN_BACH_RUN1 = { # std::map<Channel,TCut> 
#     ##        DK SIGNAL EFFICIENCY:    0.916069
#     ##   DPI BACKGROUND EFFICIENCY:    0.177365
#   "CB2DK_D2KSPIPI_DD" :  "( Bach_MC12TuneV2_ProbNNk*(1-Bach_MC12TuneV2_ProbNNpi) > 0.0833333 )",
#     ##      DPi SIGNAL EFFICIENCY:    0.799825
#     ##   DK BACKGROUND EFFICIENCY:    0.0942863
#   "CB2DPI_D2KSPIPI_DD": "( Bach_MC12TuneV2_ProbNNpi*(1-Bach_MC12TuneV2_ProbNNk) > 0.383333 )",
# }
# #### optimised in cpp/canorman_Selections/2024_06_28/Optimising_PID_Bach.cxx
# MAP_PROBNN_BACH_RUN2 = { # std::map<Channel,TCut> 
#     ##      DK SIGNAL EFFICIENCY:    0.922036
#     ## Dpi BACKGROUND EFFICIENCY:    0.170801
#     ##       BEST PROBNN CUT:    0.116667
#   "CB2DK_D2KSPIPI_DD":  "( Bach_MC15TuneV1_ProbNNk*(1-Bach_MC15TuneV1_ProbNNpi) > 0.116667 )",
#     ##    DPi SIGNAL EFFICIENCY:    0.894131
#     ## DK BACKGROUND EFFICIENCY:    0.150858
#     ##      BEST PROBNN CUT:    0.183333
#   "CB2DPI_D2KSPIPI_DD": "( Bach_MC15TuneV1_ProbNNpi*(1-Bach_MC15TuneV1_ProbNNk) > 0.183333 )",
# };
# MAP_PROBNN_BACH = { # std::map<Year, std::map<Channel,TCut> > 
#     "YRUN1": MAP_PROBNN_BACH_RUN1,
#     "Y2011": MAP_PROBNN_BACH_RUN1,
#     "Y2012": MAP_PROBNN_BACH_RUN1,
#     "YRUN2": MAP_PROBNN_BACH_RUN2,
#     "Y2015": MAP_PROBNN_BACH_RUN2,
#     "Y2016": MAP_PROBNN_BACH_RUN2,
#     "Y2017": MAP_PROBNN_BACH_RUN2,
#     "Y2018": MAP_PROBNN_BACH_RUN2,
# }

###### DTF mass cuts
MAP_DTF_CUTS = { # std::map<Channel,TCut> 
    "CB2DK_D2KSPIPI_DD" : "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
    "CB2DPI_D2KSPIPI_DD": "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
    "CB2DK_D2KSPIPI_LL" : "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
    "CB2DPI_D2KSPIPI_LL": "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
    "CB2DK_D2KSKK_DD"   : "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
    "CB2DPI_D2KSKK_DD"  : "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
    "CB2DK_D2KSKK_LL"   : "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
    "CB2DPI_D2KSKK_LL"  : "(Bu_constD0PV_KS0_M < 512.611) & (Bu_constD0PV_KS0_M > 482.611) & (Bu_constKSPV_D0_M < 1889.83) & (Bu_constKSPV_D0_M > 1839.83) & (Bu_constD0KSPV_status == 0) & (Bu_constKSPV_status == 0) & (Bu_constD0PV_status == 0)",
}

## KINEMATIC CUTS
CUTS_ENDVERTEX = "( (D0_ENDVERTEX_Z - Bu_ENDVERTEX_Z) / ( sqrt( D0_ENDVERTEX_ZERR*D0_ENDVERTEX_ZERR + Bu_ENDVERTEX_ZERR*Bu_ENDVERTEX_ZERR)) > 0.5 )" ## need to remove the individual reference
MAP_KINEMATIC_CUTS = {
    "CB2DK_D2KSPIPI_DD" : "(Bach_P < 100000)" + " & " + CUTS_ENDVERTEX,
    "CB2DPI_D2KSPIPI_DD": "(Bach_P < 100000)" + " & " +  CUTS_ENDVERTEX,
    "CB2DK_D2KSPIPI_LL" : "(Bach_P < 100000) & (Ks_FDCHI2_ORIVX > 49)" + " & " +  CUTS_ENDVERTEX,
    "CB2DPI_D2KSPIPI_LL": "(Bach_P < 100000) & (Ks_FDCHI2_ORIVX > 49)" + " & " +  CUTS_ENDVERTEX,
    "CB2DK_D2KSKK_DD"   : "(Bach_P < 100000) & (h1_P < 100000) & (h2_P < 100000)" + " & " +  CUTS_ENDVERTEX,
    "CB2DPI_D2KSKK_DD"  : "(Bach_P < 100000) & (h1_P < 100000) & (h2_P < 100000)" + " & " +  CUTS_ENDVERTEX,
    "CB2DK_D2KSKK_LL"   : "(Bach_P < 100000) & (h1_P < 100000) & (h2_P < 100000) & (Ks_FDCHI2_ORIVX > 49)" + " & " +  CUTS_ENDVERTEX,
    "CB2DPI_D2KSKK_LL"  : "(Bach_P < 100000) & (h1_P < 100000) & (h2_P < 100000) & (Ks_FDCHI2_ORIVX > 49)" + " & " +  CUTS_ENDVERTEX,
}

##### LEPTON VETO
MAP_LEPTON_VETO = {
    "CB2DK_D2KSPIPI_DD" : "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0) & (h2_PIDe < 0) ",
    "CB2DPI_D2KSPIPI_DD": "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0) & (h2_PIDe < 0) ",
    "CB2DK_D2KSPIPI_LL" : "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0) & (h2_PIDe < 0) ",
    "CB2DPI_D2KSPIPI_LL": "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0) & (h2_PIDe < 0) ",
    "CB2DK_D2KSKK_DD"   : "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0)",
    "CB2DPI_D2KSKK_DD"  : "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0)",
    "CB2DK_D2KSKK_LL"   : "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0)",
    "CB2DPI_D2KSKK_LL"  : "(h1_isMuon == 0) & (h2_isMuon == 0) & (Bach_isMuon == 0)",
}

##### BACHELOR PID - split before and afer bdt training

MAP_BACH_PID_PRELIM = {
    "CB2DK_D2KSPIPI_DD"  :  "(Bach_hasRich ==1)",
    "CB2DPI_D2KSPIPI_DD" :  "(Bach_hasRich ==1)",
    "CB2DK_D2KSPIPI_LL"  :  "(Bach_hasRich ==1)",
    "CB2DPI_D2KSPIPI_LL" :  "(Bach_hasRich ==1)",
    "CB2DK_D2KSKK_DD"    :  "(Bach_hasRich ==1)",
    "CB2DPI_D2KSKK_DD"   :  "(Bach_hasRich ==1)",
    "CB2DK_D2KSKK_LL"    :  "(Bach_hasRich ==1)",
    "CB2DPI_D2KSKK_LL"   :  "(Bach_hasRich ==1)",
}


MAP_BACH_PID_FINAL = {
    "CB2DK_D2KSPIPI_DD" : "(Bach_PIDK > 4)", ## to be optimised
    "CB2DPI_D2KSPIPI_DD": "(Bach_PIDK < 4)", ## to be optimised
    "CB2DK_D2KSPIPI_LL" : "(Bach_PIDK > 4)", ## to be optimised
    "CB2DPI_D2KSPIPI_LL": "(Bach_PIDK < 4)", ## to be optimised
    "CB2DK_D2KSKK_DD"   : "(Bach_PIDK > 4)", ## to be optimised
    "CB2DPI_D2KSKK_DD"  : "(Bach_PIDK < 4)", ## to be optimised
    "CB2DK_D2KSKK_LL"   : "(Bach_PIDK > 4)", ## to be optimised
    "CB2DPI_D2KSKK_LL"  : "(Bach_PIDK < 4)", ## to be optimised

}

#### TRUTH MATCHING
MAP_TRUTH_MATCHING = {
    "ALL": "( (abs(D0_TRUEID)==0              ) | (abs(D0_TRUEID)==421               ) )& "+
    "( (abs(D0_MC_MOTHER_ID)==0        ) | (abs(D0_MC_MOTHER_ID)==521         ) )&"+
    "( (abs(Bach_MC_MOTHER_ID)==0      ) | (abs(Bach_MC_MOTHER_ID)==521       ) )&"+
    "( (abs(Ks_TRUEID)==0              ) | (abs(Ks_TRUEID)==310               ) )&"+
    "( (abs(Ks_MC_MOTHER_ID)==0        ) | (abs(Ks_MC_MOTHER_ID)==421         ) )&"+
    "( (abs(Ks_MC_GD_MOTHER_ID)==0     ) | (abs(Ks_MC_GD_MOTHER_ID)==521      ) )&"+
    "( (abs(h1_MC_MOTHER_ID)==0        ) | (abs(h1_MC_MOTHER_ID)==421         ) )&"+
    "( (abs(h1_MC_GD_MOTHER_ID)==0     ) | (abs(h1_MC_GD_MOTHER_ID)==521      ) )&"+
    "( (abs(h2_MC_MOTHER_ID)==0        ) | (abs(h2_MC_MOTHER_ID)==421         ) )&"+
    "( (abs(h2_MC_GD_MOTHER_ID)==0     ) | (abs(h2_MC_GD_MOTHER_ID)==521      ) )&"+
    "( (abs(Ksh1_TRUEID)==0            ) | (abs(Ksh1_TRUEID)==211             ) )&"+
    "( (abs(Ksh1_MC_MOTHER_ID)==0      ) | (abs(Ksh1_MC_MOTHER_ID)==310       ) )&"+
    "( (abs(Ksh1_MC_GD_MOTHER_ID)==0   ) | (abs(Ksh1_MC_GD_MOTHER_ID)==421    ) )&"+
    "( (abs(Ksh1_MC_GD_GD_MOTHER_ID)==0) | (abs(Ksh1_MC_GD_GD_MOTHER_ID)==521 ) )&"+
    "( (abs(Ksh2_TRUEID)==0            ) | (abs(Ksh2_TRUEID)==211             ) )&"+
    "( (abs(Ksh2_MC_MOTHER_ID)==0      ) | (abs(Ksh2_MC_MOTHER_ID)==310       ) )&"+
    "( (abs(Ksh2_MC_GD_MOTHER_ID)==0   ) | (abs(Ksh2_MC_GD_MOTHER_ID)==421    ) )&"+
    "( (abs(Ksh2_MC_GD_GD_MOTHER_ID)==0) | (abs(Ksh2_MC_GD_GD_MOTHER_ID)==521 ) )",
    "DPI": "( (abs(Bach_TRUEID)==0              ) | (abs(Bach_TRUEID)== 211               ) )",
    "DK" : "( (abs(Bach_TRUEID)==0              ) | (abs(Bach_TRUEID)== 321               ) )",
    "KSPIPI": "( (abs(h1_TRUEID)==0              ) | (abs(h1_TRUEID)== 211               ) )&"+
    "( (abs(h2_TRUEID)==0              ) | (abs(h2_TRUEID)== 211               ) )",
    "KSKK": "( (abs(h1_TRUEID)==0              ) | (abs(h1_TRUEID)== 321               ) )&"+
    "( (abs(h2_TRUEID)==0              ) | (abs(h2_TRUEID)== 321               ) )",
}

MAP_TRUTH_MATCHING["MC_Bu_D0K_KSpipi_TightCut_LooserCuts_fixArrow"]  = MAP_TRUTH_MATCHING["ALL"] + " & " + MAP_TRUTH_MATCHING["KSPIPI"] + " & " + MAP_TRUTH_MATCHING["DK"]
MAP_TRUTH_MATCHING["MC_Bu_D0pi_KSpipi_TightCut_LooserCuts_fixArrow"] = MAP_TRUTH_MATCHING["ALL"] + " & " + MAP_TRUTH_MATCHING["KSPIPI"] + " & " + MAP_TRUTH_MATCHING["DPI"]
MAP_TRUTH_MATCHING["MC_Bu_D0K_KSKK_TightCut_LooserCuts_fixArrow"]    = MAP_TRUTH_MATCHING["ALL"] + " & " + MAP_TRUTH_MATCHING["KSKK"]   + " & " + MAP_TRUTH_MATCHING["DK"]
MAP_TRUTH_MATCHING["MC_Bu_D0pi_KSKK_TightCut_LooserCuts_fixArrow"]   = MAP_TRUTH_MATCHING["ALL"] + " & " + MAP_TRUTH_MATCHING["KSKK"]   + " & " + MAP_TRUTH_MATCHING["DPI"]


#endif 
