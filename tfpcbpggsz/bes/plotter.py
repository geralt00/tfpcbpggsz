import numpy as np
from scipy.interpolate import interp1d
import tfpcbpggsz.core as core
from plothist import make_hist, get_color_palette, plot_data_model_comparison, plot_model, plot_function
import os

class Hist:
    def __init__(self, Model):
        self.model = Model
        self.config = Model.config_loader
        self.bins = {}
        self.count = {}
        self.bins_sum = {}
        self.count_sum = {}
        self.range = None
        self.nbins = None
        self.weights = {}
        self.plot_list={}
        self.pc = self.model.pc
        self._DEBUG = False


    def fun_Kspipi(self, tag):

        phase_correction_sig = self.pc.eval_corr(self.config.get_phsp_srd(tag,'sig'),reduce_retracing=True)
        phase_correction_tag = self.pc.eval_corr(self.config.get_phsp_srd(tag,'tag'),reduce_retracing=True)
        #need to be flexible with the function name
        ret = core.prob_totalAmplitudeSquared_CP_mix(self.config.get_phsp_amp(tag,'sig'), self.config.get_phsp_ampbar(tag,'sig'), self.config.get_phsp_amp(tag,'tag'), self.config.get_phsp_ampbar(tag,'tag'), phase_correction_sig, phase_correction_tag)

        return ret
    
    def fun_CP(self, tag, Dsign):

        phase_correction = self.pc.eval_corr(self.config.get_phsp_srd(tag),reduce_retracing=True)
        if tag != 'pipipi0' :
            ret = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config.get_phsp_amp(tag), self.config.get_phsp_ampbar(tag), pc=phase_correction) 
        else:
            ret = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config.get_phsp_amp(tag), self.config.get_phsp_ampbar(tag), pc=phase_correction,  Fplus=0.9406)
        return ret

    def get_hist_each(self, cato='dks', tag='full', mc_type='phsp'):

        if mc_type == 'phsp':
            if cato == 'dks':
                self.weights[tag] = self.fun_Kspipi(tag)
            elif cato == 'cp_odd':
                self.weights[tag] = self.fun_CP(tag, -1)
            elif cato == 'cp_even':
                self.weights[tag] = self.fun_CP(tag, 1)

        self.count[tag]={} if tag not in self.count.keys() else self.count[tag]
        self.bins[tag]={} if tag not in self.bins.keys() else self.bins[tag]
        self.count[tag][mc_type]={} if mc_type not in self.count[tag].keys() else self.count[tag][mc_type]
        self.bins[tag][mc_type]={} if mc_type not in self.bins[tag].keys() else self.bins[tag][mc_type]

        if mc_type == 'phsp':
            if cato!='dks':
                self.count[tag][mc_type]['s12'], self.bins[tag][mc_type]['s12'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[0], self.nbins*3, weights=self.weights[tag], range=self.range)
                self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s13'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[1], self.nbins*3, weights=self.weights[tag], range=self.range)
            else:
                self.count[tag][mc_type]['s12'], self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s12'], self.bins[tag][mc_type]['s13'] = {}, {}, {}, {}
                self.count[tag][mc_type]['s12']['sig'], self.bins[tag][mc_type]['s12']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[0], self.nbins*3, weights=self.weights[tag], range=self.range)
                self.count[tag][mc_type]['s13']['sig'], self.bins[tag][mc_type]['s13']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[1], self.nbins*3, weights=self.weights[tag], range=self.range)
                self.count[tag][mc_type]['s12']['tag'], self.bins[tag][mc_type]['s12']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[0], self.nbins*3, weights=self.weights[tag], range=self.range)
                self.count[tag][mc_type]['s13']['tag'], self.bins[tag][mc_type]['s13']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[1], self.nbins*3, weights=self.weights[tag], range=self.range)


        else:
            if cato!='dks':
                #Test with null dict
                self.count[tag][mc_type]['s12'], self.bins[tag][mc_type]['s12'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[0], self.nbins*3, range=self.range)
                self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s13'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[1], self.nbins*3, range=self.range)
            else:
                self.count[tag][mc_type]['s12'], self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s12'], self.bins[tag][mc_type]['s13'] = {}, {}, {}, {}
                self.count[tag][mc_type]['s12']['sig'], self.bins[tag][mc_type]['s12']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[0], self.nbins*3, range=self.range)
                self.count[tag][mc_type]['s13']['sig'], self.bins[tag][mc_type]['s13']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[1], self.nbins*3, range=self.range)
                self.count[tag][mc_type]['s12']['tag'], self.bins[tag][mc_type]['s12']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[0], self.nbins*3, range=self.range)
                self.count[tag][mc_type]['s13']['tag'], self.bins[tag][mc_type]['s13']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[1], self.nbins*3, range=self.range)



    def get_hist_sum(self):

        mc_type = ['phsp', 'qcmc', 'dpdm', 'qqbar', 'sigmc_um', 'qcmc_oth']
        for key in self.plot_list:
            if key != 'dks':
                self.bins_sum[key]={}
                self.count_sum[key]={}
                for tag in self.plot_list[key]:
                    for i_mc_type in mc_type:
                        if (tag not in ['klpi0', 'klpi0pi0', 'ksomega'] and i_mc_type in ['qcmc_oth']) or (tag not in ['klpi0pi0'] and i_mc_type in ['sigmc_um']):
                            self.count[tag][i_mc_type]={} if i_mc_type not in self.count[tag].keys() else self.count[tag][i_mc_type]
                            self.bins[tag][i_mc_type]={} if i_mc_type not in self.bins[tag].keys() else self.bins[tag][i_mc_type]
                            self.count[tag][i_mc_type]['s12'], self.count[tag][i_mc_type]['s13'], self.bins[tag][i_mc_type]['s12'], self.bins[tag][i_mc_type]['s13'] = np.zeros_like(self.count[tag]['phsp']['s12']), np.zeros_like(self.count[tag]['phsp']['s13']), self.bins[tag]['phsp']['s12'], self.bins[tag]['phsp']['s13']
                        else: 
                            self.get_hist_each(cato=key, tag=tag, mc_type=i_mc_type)

                for i_mc_type in mc_type:
                    self.count_sum[key][i_mc_type]={} if i_mc_type not in self.count_sum[key].keys() else self.count_sum[key][i_mc_type]
                    self.bins_sum[key][i_mc_type]={} if i_mc_type not in self.bins_sum[key].keys() else self.bins_sum[key][i_mc_type]
                    #for tag in self.plot_list[key]:
                    #    self.get_hist_each(cato=key, tag=tag, mc_type=i_mc_type)
                    self.count_sum[key][i_mc_type]['s12'] = np.sum([self.count[tag][i_mc_type]['s12'] for tag in self.plot_list[key]], axis=0)
                    self.count_sum[key][i_mc_type]['s13'] = np.sum([self.count[tag][i_mc_type]['s13'] for tag in self.plot_list[key]], axis=0)
                    self.bins_sum[key][i_mc_type]['s12'] = self.bins[tag][i_mc_type]['s12']
                    self.bins_sum[key][i_mc_type]['s13'] = self.bins[tag][i_mc_type]['s13']
            else:
                self.bins_sum[key]={}
                self.count_sum[key]={}
                mc_type = ['phsp', 'qcmc', 'dpdm', 'qqbar', 'sigmc_um']

                        
                for i_mc_type in mc_type:
                    for tag in self.plot_list[key]:
                        if tag in ['full', 'misspi'] and i_mc_type == 'sigmc_um':
                            self.count[tag][i_mc_type]={} if i_mc_type not in self.count[tag].keys() else self.count[tag][i_mc_type]
                            self.bins[tag][i_mc_type]={} if i_mc_type not in self.bins[tag].keys() else self.bins[tag][i_mc_type]
                            self.count[tag][i_mc_type]['s12'], self.count[tag][i_mc_type]['s13'], self.bins[tag][i_mc_type]['s12'], self.bins[tag][i_mc_type]['s13'] = {}, {}, {}, {}

                            self.count[tag][i_mc_type]['s12']['tag'] = np.zeros_like(self.count[tag]['phsp']['s12']['sig'])
                            self.count[tag][i_mc_type]['s13']['tag'] = np.zeros_like(self.count[tag]['phsp']['s13']['sig'])
                            self.bins[tag][i_mc_type]['s12']['tag'] = self.bins[tag]['phsp']['s12']['sig']
                            self.bins[tag][i_mc_type]['s13']['tag'] = self.bins[tag]['phsp']['s13']['sig']
                            self.count[tag][i_mc_type]['s12']['sig'] = np.zeros_like(self.count[tag]['phsp']['s12']['sig'])
                            self.count[tag][i_mc_type]['s13']['sig'] = np.zeros_like(self.count[tag]['phsp']['s13']['sig'])
                            self.bins[tag][i_mc_type]['s12']['sig'] = self.bins[tag]['phsp']['s12']['sig']
                            self.bins[tag][i_mc_type]['s13']['sig'] = self.bins[tag]['phsp']['s13']['sig']
                        else:
                            self.get_hist_each(cato=key, tag=tag, mc_type=i_mc_type)
                          
                        

                for i_mc_type in mc_type:
                    self.count_sum[key][i_mc_type]={} if i_mc_type not in self.count_sum[key].keys() else self.count_sum[key][i_mc_type]
                    self.bins_sum[key][i_mc_type]={} if i_mc_type not in self.bins_sum[key].keys() else self.bins_sum[key][i_mc_type]
                    self.count_sum[key][i_mc_type]['s12'], self.count_sum[key][i_mc_type]['s13'], self.bins_sum[key][i_mc_type]['s12'], self.bins_sum[key][i_mc_type]['s13'] = {}, {}, {}, {}
                    self.count_sum[key][i_mc_type]['s12']['sig'] = np.sum([self.count[tag][i_mc_type]['s12']['sig'] for tag in self.plot_list[key]], axis=0)
                    self.count_sum[key][i_mc_type]['s13']['sig'] = np.sum([self.count[tag][i_mc_type]['s13']['sig'] for tag in self.plot_list[key]], axis=0)
                    self.bins_sum[key][i_mc_type]['s12']['sig'] = self.bins[tag][i_mc_type]['s12']['sig']
                    self.bins_sum[key][i_mc_type]['s13']['sig'] = self.bins[tag][i_mc_type]['s13']['sig']
                    self.count_sum[key][i_mc_type]['s12']['tag'] = np.sum([self.count[tag][i_mc_type]['s12']['tag'] for tag in self.plot_list[key]], axis=0)
                    self.count_sum[key][i_mc_type]['s13']['tag'] = np.sum([self.count[tag][i_mc_type]['s13']['tag'] for tag in self.plot_list[key]], axis=0)
                    self.bins_sum[key][i_mc_type]['s12']['tag'] = self.bins[tag][i_mc_type]['s12']['tag']
                    self.bins_sum[key][i_mc_type]['s13']['tag'] = self.bins[tag][i_mc_type]['s13']['tag']


    def hist_to_fun(self, count, bins, scale, kind='linear'):
        x = (bins[:-1] + bins[1:])/2

        #Do normalization to count
        count = count*(3*scale/np.sum(count))
        f = interp1d(x, count, kind=kind, fill_value='extrapolate')
        return f

class Plotter:
    def __init__(self, Model, **kwargs):
        self.config = Model.config_loader
        self.hist = Hist(Model)
        self.weights = {}
        self.count = {}
        self.bins = {}
        self.get_plot_into()
        self.hist.get_hist_sum()
        self.save_path = os.environ['PWD']+'/plots'
        if 'save_path' in kwargs:
            self.save_path = kwargs['save_path']
        

    
    def get_plot_into(self):
        self.hist.nbins = self.config._config_data['plot']['bins']
        self.hist.range = self.config._config_data['plot']['range']
        self.hist.plot_list = self.config._config_data['plot']['plot_sum']

    def plot_cato(self, cato='dks'):

        count={}
        bins={}
        mc_type = ['phsp', 'qcmc', 'dpdm', 'qqbar']
        if cato == 'dks':
            mc_type.append('sigmc_um')
            for i_mc_type in mc_type:
                count[i_mc_type], bins[i_mc_type] = {}, {} 
                count[i_mc_type]['s12'], count[i_mc_type]['s13'], bins[i_mc_type]['s12'], bins[i_mc_type]['s13'] = {}, {}, {}, {}
                count[i_mc_type]['s12']['sig'], bins[i_mc_type]['s12']['sig'], count[i_mc_type]['s13']['sig'], bins[i_mc_type]['s13']['sig'] = self.hist.count_sum[cato][i_mc_type]['s12']['sig'], self.hist.bins_sum[cato][i_mc_type]['s12']['sig'], self.hist.count_sum[cato][i_mc_type]['s13']['sig'], self.hist.bins_sum[cato][i_mc_type]['s13']['sig']
                count[i_mc_type]['s12']['tag'], bins[i_mc_type]['s12']['tag'], count[i_mc_type]['s13']['tag'], bins[i_mc_type]['s13']['tag'] = self.hist.count_sum[cato][i_mc_type]['s12']['tag'], self.hist.bins_sum[cato][i_mc_type]['s12']['tag'], self.hist.count_sum[cato][i_mc_type]['s13']['tag'], self.hist.bins_sum[cato][i_mc_type]['s13']['tag']
        elif cato == 'cp_even':
            mc_type.append('qcmc_oth')
            for i_mc_type in mc_type:
                count[i_mc_type], bins[i_mc_type] = {}, {} 
                count[i_mc_type]['s12'], bins[i_mc_type]['s12'], count[i_mc_type]['s13'], bins[i_mc_type]['s13'] = self.hist.count_sum[cato][i_mc_type]['s12'], self.hist.bins_sum[cato][i_mc_type]['s12'], self.hist.count_sum[cato][i_mc_type]['s13'], self.hist.bins_sum[cato][i_mc_type]['s13']
        else:
            mc_type.append('qcmc_oth')
            mc_type.append('sigmc_um')
            for i_mc_type in mc_type:
                count[i_mc_type], bins[i_mc_type] = {}, {} 
                count[i_mc_type]['s12'], bins[i_mc_type]['s12'], count[i_mc_type]['s13'], bins[i_mc_type]['s13'] = self.hist.count_sum[cato][i_mc_type]['s12'], self.hist.bins_sum[cato][i_mc_type]['s12'], self.hist.count_sum[cato][i_mc_type]['s13'], self.hist.bins_sum[cato][i_mc_type]['s13']


#        for i_mc_type in mc_type:
#            if cato!='dks':
#                count[i_mc_type], bins[i_mc_type] = {}, {} 
#                count[i_mc_type]['s12'], bins[i_mc_type]['s12'], count[i_mc_type]['s13'], bins[i_mc_type]['s13'] = self.hist.count_sum[cato][i_mc_type]['s12'], self.hist.bins_sum[cato][i_mc_type]['s12'], self.hist.count_sum[cato][i_mc_type]['s13'], self.hist.bins_sum[cato][i_mc_type]['s13']
#            else:
#                count[i_mc_type], bins[i_mc_type] = {}, {} 
#                count[i_mc_type]['s12'], count[i_mc_type]['s13'], bins[i_mc_type]['s12'], bins[i_mc_type]['s13'] = {}, {}, {}, {}
#                count[i_mc_type]['s12']['sig'], bins[i_mc_type]['s12']['sig'], count[i_mc_type]['s13']['sig'], bins[i_mc_type]['s13']['sig'] = self.hist.count_sum[cato][i_mc_type]['s12']['sig'], self.hist.bins_sum[cato][i_mc_type]['s12']['sig'], self.hist.count_sum[cato][i_mc_type]['s13']['sig'], self.hist.bins_sum[cato][i_mc_type]['s13']['sig']
#                count[i_mc_type]['s12']['tag'], bins[i_mc_type]['s12']['tag'], count[i_mc_type]['s13']['tag'], bins[i_mc_type]['s13']['tag'] = self.hist.count_sum[cato][i_mc_type]['s12']['tag'], self.hist.bins_sum[cato][i_mc_type]['s12']['tag'], self.hist.count_sum[cato][i_mc_type]['s13']['tag'], self.hist.bins_sum[cato][i_mc_type]['s13']['tag']

        data_hist = {}
        if cato != 'dks':
            data_hist['s12'], data_hist['s13'], scale = self.make_data_hist(cato=cato)
        else:
            data_hist['s12'], data_hist['s13'] = {}, {}
            data_hist['s12']['sig'], data_hist['s13']['sig'], data_hist['s12']['tag'], data_hist['s13']['tag'], scale = self.make_data_hist(cato=cato)



        for key in count['phsp'].keys():
            #create data hist
            if cato =='cp_odd':
                f_sig = self.hist.hist_to_fun(count=count['phsp'][key], bins=bins['phsp'][key],scale=scale['sig'])
                f_qcmc = self.hist.hist_to_fun(count=count['qcmc'][key], bins=bins['qcmc'][key],scale=scale['qcmc'])
                f_dpdm = self.hist.hist_to_fun(count=count['dpdm'][key], bins=bins['dpdm'][key],scale=scale['dpdm'])
                f_qqbar = self.hist.hist_to_fun(count=count['qqbar'][key], bins=bins['qqbar'][key],scale=scale['qqbar'])
                f_qcmc_oth = self.hist.hist_to_fun(count=count['qcmc_oth'][key], bins=bins['qcmc_oth'][key],scale=scale['qcmc_oth'])
                f_sigmc_um = self.hist.hist_to_fun(count=count['sigmc_um'][key], bins=bins['sigmc_um'][key],scale=scale['sigmc_um'])
                f_trans = lambda x: 0*x
                fig, ax_main, ax_comparison = plot_data_model_comparison(
                    data_hist=data_hist[key],
                    unstacked_components=[f_sig, f_trans],
                    unstacked_labels=["", 'Signal'],
                    unstacked_colors=["#00000000", "#8EBA42"],

                    stacked_components=[f_qqbar,  f_dpdm, f_sigmc_um, f_qcmc_oth, f_qcmc],
                    stacked_labels=["$q\\bar{q}$", "$D^+D^-$", "Mis. Comb.", "QCMC Oth.", "QCMC"],
                    xlabel=key,
                    ylabel="Entries",
                    model_sum_kwargs={"show": True, "label": "Model", "color": "navy"},
                    comparison="pull",
                    range=self.hist.range,
                )
                plot_function([f_sig], range=self.hist.range, ax=ax_main, npoints=100000, label='signal', color='#8EBA42')


                os.makedirs(self.save_path, exist_ok=True)
                fig.savefig(
                    f"{self.save_path}/{cato}_{key}.pdf",
                    bbox_inches="tight",
                    )
            elif cato =='cp_even':
                f_sig = self.hist.hist_to_fun(count=count['phsp'][key], bins=bins['phsp'][key],scale=scale['sig'])
                f_qcmc = self.hist.hist_to_fun(count=count['qcmc'][key], bins=bins['qcmc'][key],scale=scale['qcmc'])
                f_dpdm = self.hist.hist_to_fun(count=count['dpdm'][key], bins=bins['dpdm'][key],scale=scale['dpdm'])
                f_qqbar = self.hist.hist_to_fun(count=count['qqbar'][key], bins=bins['qqbar'][key],scale=scale['qqbar'])
                f_qcmc_oth = self.hist.hist_to_fun(count=count['qcmc_oth'][key], bins=bins['qcmc_oth'][key],scale=scale['qcmc_oth'])
                f_trans = lambda x: 0*x
            #x = np.linspace(bins[key][0], bins[key][-1],bins[key].shape[0]*10)

                fig, ax_main, ax_comparison = plot_data_model_comparison(
                    data_hist=data_hist[key],
                    unstacked_components=[f_sig, f_trans],
                    unstacked_labels=["", 'Signal'],
                    unstacked_colors=["#00000000", "#8EBA42"],

                    stacked_components=[f_qqbar,  f_dpdm, f_qcmc_oth, f_qcmc],
                    stacked_labels=["$q\\bar{q}$", "$D^+D^-$", "QCMC Oth.", "QCMC"],
                    xlabel=key,
                    ylabel="Entries",
                    model_sum_kwargs={"show": True, "label": "Model", "color": "navy"},
                    comparison="pull",
                    range=self.hist.range,
                )
                plot_function([f_sig], range=self.hist.range, ax=ax_main, npoints=100000, label='signal', color='#8EBA42')


                os.makedirs(self.save_path, exist_ok=True)
                fig.savefig(
                    f"{self.save_path}/{cato}_{key}.pdf",
                    bbox_inches="tight",
                    )

            elif cato == 'dks':
                for i_key_tag in ['sig', 'tag']:
                    f_sig = self.hist.hist_to_fun(count=count['phsp'][key][i_key_tag], bins=bins['phsp'][key][i_key_tag],scale=scale['sig'])
                    f_qcmc = self.hist.hist_to_fun(count=count['qcmc'][key][i_key_tag], bins=bins['qcmc'][key][i_key_tag],scale=scale['qcmc'])
                    f_dpdm = self.hist.hist_to_fun(count=count['dpdm'][key][i_key_tag], bins=bins['dpdm'][key][i_key_tag],scale=scale['dpdm'])
                    f_qqbar = self.hist.hist_to_fun(count=count['qqbar'][key][i_key_tag], bins=bins['qqbar'][key][i_key_tag],scale=scale['qqbar'])
                    f_sigmc_um = self.hist.hist_to_fun(count=count['sigmc_um'][key][i_key_tag], bins=bins['sigmc_um'][key][i_key_tag],scale=scale['sigmc_um'])
                    f_trans = lambda x: 0*x

                    #x = np.linspace(bins[key][0], bins[key][-1],bins[key].shape[0]*10)

                    fig, ax_main, ax_comparison = plot_data_model_comparison(
                        data_hist=data_hist[key][i_key_tag],
                        unstacked_components=[f_sig, f_trans],
                        unstacked_labels=["", 'Signal'],
                        unstacked_colors=["#00000000", "#8EBA42"],
                        stacked_components=[f_qqbar, f_dpdm, f_sigmc_um, f_qcmc],
                        stacked_labels=["$q\\bar{q}$", "$D^+D^-$", "Mis. Comb.", "QCMC"],
                        #stacked_colors=get_color_palette("default", n_colors=4),
                        xlabel=key,
                        ylabel="Entries",
                        model_sum_kwargs={"show": True, "label": "Model", "color": "navy"},
                        comparison="pull",
                        range=self.hist.range,
                    )
                    plot_function([f_sig], range=self.hist.range, ax=ax_main, npoints=100000, label='signal', color='#8EBA42')

                    os.makedirs(self.save_path, exist_ok=True)
                    fig.savefig(
                        f"{self.save_path}/{cato}_{key}_{i_key_tag}.pdf",
                        bbox_inches="tight",
                        )

        #Not yet consider the double kspipi yet

    def make_data_hist(self, cato='dks'):
        s12_sig, s13_sig, s12_tag, s13_tag = [], [], [], []
        scale = {'sig': 0, 'qcmc': 0, 'dpdm': 0, 'qqbar': 0, 'sigmc_um': 0, 'qcmc_oth': 0}
        for tag in self.config._config_data['plot']['plot_sum'][cato]:

            scale['sig'] += self.config.get_sig_num(tag)#len(s12_sig_i)*(1-np.sum(self.config.get_bkg_frac(tag), axis=0))
            scale['qcmc'] += self.config.get_bkg_num(tag, 'qcmc')#len(s12_sig_i)*np.sum(self.config.get_bkg_frac(tag), axis=0)
            scale['dpdm'] += self.config.get_bkg_num(tag, 'dpdm')#len(s12_sig_i)*np.sum(self.config.get_bkg_frac(tag), axis=0)
            scale['qqbar'] += self.config.get_bkg_num(tag, 'qqbar')#len(s12_sig_i)*np.sum(self.config.get_bkg_frac(tag), axis=0)
            if tag in ['misspi0', 'klpi0pi0']:
                scale['sigmc_um'] += self.config.get_bkg_num(tag, 'sigmc_um')
            if tag in ['klpi0', 'ksomega']:
                scale['qcmc_oth'] += self.config.get_bkg_num(tag, 'qcmc_oth')
            if cato != 'dks':
                s12_sig_i, s13_sig_i = self.config.get_data_mass(tag=tag)
                s12_sig.append(s12_sig_i)
                s13_sig.append(s13_sig_i)
                #print(f'{s12_sig_i.shape} the expected shape is {self.config.get_sig_num(tag)} for {tag}')
            else:
                s12_sig_i, s13_sig_i = self.config.get_data_mass(tag=tag, key='sig')
                s12_tag_i, s13_tag_i = self.config.get_data_mass(tag=tag, key='tag')

                s12_sig.append(s12_sig_i)
                s13_sig.append(s13_sig_i)
                s12_tag.append(s12_tag_i)
                s13_tag.append(s13_tag_i)


        s12_sig = np.concatenate(s12_sig)
        s13_sig = np.concatenate(s13_sig)

        s12_tag = np.concatenate(s12_tag) if cato == 'dks' else None
        s13_tag = np.concatenate(s13_tag) if cato == 'dks' else None
        if cato != 'dks':
            data_s12_hist = make_hist(s12_sig, bins=self.hist.nbins,  range=self.hist.range)
            data_s13_hist = make_hist(s13_sig, bins=self.hist.nbins, range=self.hist.range)
            return data_s12_hist, data_s13_hist, scale
        else:
            data_s12_sig_hist = make_hist(s12_sig, bins=self.hist.nbins, range=self.hist.range)
            data_s13_sig_hist = make_hist(s13_sig, bins=self.hist.nbins, range=self.hist.range)
            data_s12_tag_hist = make_hist(s12_tag, bins=self.hist.nbins, range=self.hist.range)
            data_s13_tag_hist = make_hist(s13_tag, bins=self.hist.nbins, range=self.hist.range)
            return data_s12_sig_hist, data_s13_sig_hist, data_s12_tag_hist, data_s13_tag_hist, scale