import math
import numpy as np
import os
import json
#import ROOT as rt
import glob


class yields:
    """
    Minimum class to access the yields from the provided file
    Obtained yields from the provided file, api to access bes data. Set to 20 ifb dataset as default
    """
    def __init__(self, D02KsPiPi):
        self.round = None
        self._data = None
        self.D02KsPiPi = D02KsPiPi

    def load(self,file_path=None):
        with open(file_path) as f:
            data_temp = json.load(f)
            self._data = data_temp

            return self._data

    def get(self, round='all', type='data', tag='full'):
        """
        return the certain type
        """
        if self._data is None:
            self.load(type=type, round=round)
        
        if type == 'fit_result':
            return self._data[type]
        else:
            return self._data[type][round][self.D02KsPiPi.catogery(tag=tag)][tag]


class D02KsPiPi:
    def __init__(self):
        self.round = None
        self._data = None


    def xtitle(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return xtitle_list[_wrapper.index(i_name)]

    def find_tag(self, index):
        i_tag_x = 0
        i_tag_y = 0

        if index < 3:
            i_tag_x = 0
            i_tag_y = index
        elif 3 <= index < 10:
            i_tag_x = 1
            i_tag_y = index - 3
        elif 10 <= index < 15:
            i_tag_x = 2
            i_tag_y = index - 10
        elif 15 <= index < 19:
            i_tag_x = 3
            i_tag_y = index - 15

        return i_tag_x, i_tag_y
     

    #Could move to plot eventually
    def x_min(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return x_min[_wrapper.index(i_name)]
            
    def x_max(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return x_max[_wrapper.index(i_name)]

    def sig_range_min(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return _sig_range_min[_wrapper.index(i_name)]
    
    def sig_range_max(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return _sig_range_max[_wrapper.index(i_name)]

    def nbins(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return nbins[_wrapper.index(i_name)]
    
    def cuts(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return cuts[_wrapper.index(i_name)]
            
    def tag_latex(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return tag_latex_list[_wrapper.index(i_name)]
            
    def topo_cut(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return new_topo_cut[_wrapper.index(i_name)]
    
    def var_name(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return var_name[_wrapper.index(i_name)]
            
    def root_name_list(self, tag):
        for i_name in _wrapper:
            if tag == i_name:
                return sigmc_list[_wrapper.index(i_name)]
    
    def catogery(self, tag):

        for key in tags:
            if tag in tags[key]:
                return key

    

tags = {'dks': ['full', 'misspi', 'misspi0'],
        'cpodd': ['kspi0', 'kseta_gamgam', 'ksetap_pipieta', 'kseta_3pi', 'ksetap_gamrho', 'ksomega', 'klpi0pi0'],
        'cpeven': ['kk', 'pipi', 'pipipi0', 'kspi0pi0', 'klpi0'],
        'flav': ['kpipi0', 'kpi', 'k3pi', 'kenu']
}

var_name = [
    "mBC1", "mm2", "mm2",
    "mBC1", "mBC1", "mBC1", "mBC1", "mBC1", "mBC1", "mm2",
    "mBC1", "mBC1", "mBC1", "mBC1", "mm2",
    "mBC1", "mBC1", "mBC1", "UMISS"]

xtitle_list = ["#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{miss}^{2} (GeV^{2}/#it{c}^{4})", "#it{M}_{miss}^{2} (GeV^{2}/#it{c}^{4})",
    "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{miss}^{2} (GeV^{2}/#it{c}^{4})",
    "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{miss}^{2} (GeV^{2}/#it{c}^{4})",
    "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{M}_{BC} (GeV/#it{c}^{2})", "#it{U}_{miss} (GeV^{2}/#it{c})"]


x_max = [
    1.89, 0.2, 0.15, 
    1.89, 1.89, 1.89, 1.89, 1.89, 1.89, 0.6,
    1.89, 1.89, 1.89, 1.89, 0.6, 
    1.89, 1.89, 1.89, 0.2]

x_min = [
    1.84, -0.15, -0.1,
    1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 0,
    1.84, 1.84, 1.84, 1.84, 0.0,
    1.84, 1.84, 1.84, -0.2
]

nbins = [
    100, 80, 80,
    100, 100, 100, 100, 100, 100, 80,
    100, 100, 100, 100, 80,
    100, 100, 100, 80

]

cuts = [
    "(mBC1>1.86) & (mBC1<1.87)", "(mm2>-0.05) & (mm2<0.08)", "(mm2>-0.05) & (mm2<0.05)",
    "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86)  &(mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", " (mm2>0.2) & (mm2<0.4)",
    "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86)  &(mBC1<1.87)", "(mm2>0.15) & (mm2<0.5)",
    "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", "(mBC1>1.86) & (mBC1<1.87)", "(UMISS>-0.05)&(UMISS<0.08)"
]

_sig_range_min = [
    1.86, -0.05, -0.05,
    1.86, 1.86, 1.86, 1.86, 1.86, 1.86, 0.2,
    1.86, 1.86, 1.86, 1.86, 0.15,
    1.86, 1.86, 1.86, -0.05
]

_sig_range_max = [
    1.87, 0.08, 0.05,
    1.87, 1.87, 1.87, 1.87, 1.87, 1.87, 0.4,
    1.87, 1.87, 1.87, 1.87, 0.5,
    1.87, 1.87, 1.87, 0.08
]

tag_latex_list = [
    "K_{S}#pi^{+}#pi^{-} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}#pi_{Miss} #pi vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}(Miss #pi^{0})#pi^{+}#pi^{-} vs. K_{S}#pi^{+}#pi^{-}",
     
    "K_{S}#pi^{0} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}#eta_{#gamma#gamma} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}#eta'_{#eta#pi#pi} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}#eta_{#pi#pi#pi^{0}} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}#eta'_{#gamma#rho} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}#omega vs. K_{S}#pi^{+}#pi^{-}",
     "K_{L}#pi^{0}#pi^{0} vs. K_{S}#pi^{+}#pi^{-}",

    "K^{+}K^{-} vs. K_{S}#pi^{+}#pi^{-}",
     "#pi^{+}#pi^{-} vs. K_{S}#pi^{+}#pi^{-}",
     "#pi^{+}#pi^{-}#pi^{0} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{S}#pi^{0}#pi^{0} vs. K_{S}#pi^{+}#pi^{-}",
     "K_{L}#pi^{0} vs. K_{S}#pi^{+}#pi^{-}",
     
    "K^{+}#pi^{-}#pi^{0} vs. K_{S}#pi^{+}#pi^{-}",
     "K^{+}#pi^{-} vs. K_{S}#pi^{+}#pi^{-}",
     "K^{+}#pi^{+}#pi^{-}#pi^{-} vs. K_{S}#pi^{+}#pi^{-}",
     "K^{+}e^{-}#nu_{e} vs. K_{S}#pi^{+}#pi^{-}"
]

topo_cut = [
    "(nSigDcyBr_0 != 1 || nSigDcyBr_1 != 1)", "(nSigDcyBr_0 != 1 || nSigDcyBr_1 != 1)",
     "(nSigDcyBr_0 != 1 || nSigDcyBr_1 != 1)",

    "!(iDcyTr==2||iDcyTr==0)", "!(iDcyTr==1||iDcyTr==0)", "!(iDcyTr==1||iDcyTr==0)",
     "!(iDcyTr==1||iDcyTr==0)", "!(iDcyTr==1||iDcyTr==0)", "!(iDcyTr==1||iDcyTr==0)",
     " !(iDcyTr==4||iDcyTr==0||iDcyTr==1||iDcyTr==3)&( mkl01 >0.7 & mkl02>0.7)",

    "!(iDcyTr==1||iDcyTr==0||iDcyTr==3||iDcyTr==6)",
     "!(iDcyTr==1||iDcyTr==0||iDcyTr==2||iDcyTr==6)", "(iDcyIFSts!=0||iDcyTr==3||iDcyTr==19)",
     "!(iDcyTr==1||iDcyTr==0||iDcyTr==7||iDcyTr==6)",
     "!(iDcyTr==1||iDcyTr==0||iDcyTr==3||iDcyTr==19||iDcyTr==5||iDcyTr==2)",

    "!(iDcyTr==1||iDcyTr==0||iDcyTr==4||iDcyTr==9||iDcyTr==8||iDcyTr==3)",
     "!(iDcyTr==1||iDcyTr==0||iDcyTr==4||iDcyTr==5||iDcyTr==10||iDcyTr==2)",
     "!(iDcyTr==1||iDcyTr==0||iDcyTr==9||iDcyTr==12||iDcyTr==5||iDcyTr==6||iDcyTr==24||iDcyTr==2||iDcyTr==18||iDcyTr==4||iDcyTr==22||iDcyTr==7||iDcyTr==10||iDcyTr==21 || iDcyTr==2||iDcyTr==25||iDcyTr==19||iDcyTr==4||iDcyTr==8||iDcyTr==5||iDcyTr==11||iDcyTr==22)",
     "!(iDcyTr==1||iDcyTr==0||iDcyTr==18||iDcyTr==30)"
]

new_topo_cut = [
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_3 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_3 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_3 ==1))",

    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_6 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_7 ==1) || (nSigDcyBr_5 ==1 & nSigDcyBr_6 ==1) || (nSigDcyBr_5 ==1 & nSigDcyBr_7 ==1))",

    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_4 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_4 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_6 ==1 & nSigDcyBr_8 ==1) || (nSigDcyBr_6 ==1 & nSigDcyBr_9 ==1) || (nSigDcyBr_6 ==1 & nSigDcyBr_10 ==1) || (nSigDcyBr_6 ==1 & nSigDcyBr_11 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_8 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_9 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_10 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_11 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_6 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_7 ==1) || (nSigDcyBr_5 ==1 & nSigDcyBr_6 ==1) || (nSigDcyBr_5 ==1 & nSigDcyBr_7 ==1))",


    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_4 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_0 ==1 & nSigDcyBr_6 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_3 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_4 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_6 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_9 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_10 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_11 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_12 ==1) || (nSigDcyBr_7 ==1 & nSigDcyBr_13 ==1) || (nSigDcyBr_8 ==1 & nSigDcyBr_9 ==1) || (nSigDcyBr_8 ==1 & nSigDcyBr_10 ==1) || (nSigDcyBr_8 ==1 & nSigDcyBr_11 ==1) || (nSigDcyBr_8 ==1 & nSigDcyBr_12 ==1) || (nSigDcyBr_8 ==1 & nSigDcyBr_13 ==1))",
    "!((nSigDcyBr_0 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_1 ==1 & nSigDcyBr_2 ==1) || (nSigDcyBr_3 ==1 & nSigDcyBr_5 ==1) || (nSigDcyBr_4 ==1 & nSigDcyBr_5 ==1))"
    ]

tag_name_list = [
    "dks_full", "misspi", "misspi0",
    "kspi0", "kseta_gg", "ksetap_eta", "kseta_3pi", "ksetap_gpi", "ksomega", "klpi0pi0",
    "kk", "pipi", "pipipi0", "kspi0pi0", "klpi0",
    "kpipi0", "kpi", "k3pi", "kenu"
]

sigmc_list = [
    "dksfull_0", "dks_data_1", "misspi0_data",
    "kstag_data_1", "kstag_data_2", "kstag_data_3", "kstag_data_4", "kstag_data_5", "kstag_data_6", "klpi0pi0_data",
    "oth_data_2", "oth_data_1", "oth_data_0", "kstag_data_0", "klpi0_data",
    "ftag_data_0", "ftag_data_1", "ftag_data_2", "kenu_data_0"
]

_wrapper = ["full", "misspi", "misspi0", "kspi0", "kseta_gamgam", "ksetap_pipieta", "kseta_3pi", "ksetap_gamrho", "ksomega", "klpi0pi0", "kk", "pipi", "pipipi0", "kspi0pi0", "klpi0", "kpipi0", "kpi", "k3pi", "kenu"]


'''
_Lumi={'0304': '2.9', '03-04': '2.9', '030415': '7.9', '15': '5.0', '16': '8.2', '17': '4.2', 'all': '20.0'}

def Writedataset(round='16'):
    dataset = rt.TLatex(0.87, 0.87, f'{_Lumi[round]}'+ ' fb^{-1}')
    dataset.SetNDC()
    dataset.SetTextFont(132)
    dataset.SetTextSize(0.08)
    dataset.SetTextAlign(33)
    return dataset

def Write(latex='', x=0.6, y=0.75):

    dataset = rt.TLatex(x, y, latex)  # 0.88 for dp
    dataset.SetNDC()
    dataset.SetTextFont(42)
    dataset.SetTextSize(0.05)
    dataset.SetTextAlign(11)
    return dataset
'''
