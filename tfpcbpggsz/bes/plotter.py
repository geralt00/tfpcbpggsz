import numpy as np
from scipy.interpolate import interp1d
import tfpcbpggsz.core as core
from plothist import make_hist, get_color_palette, plot_data_model_comparison
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


    def fun_Kspipi(self, tag):

        phase_correction_sig = self.pc.eval_corr(self.config.get_phsp_srd(tag,'sig'))
        phase_correction_tag = self.pc.eval_corr(self.config.get_phsp_srd(tag,'tag'))
        #need to be flexible with the function name
        ret = core.prob_totalAmplitudeSquared_CP_mix(self.config.get_phsp_amp(tag,'sig'), self.config.get_phsp_ampbar(tag,'sig'), self.config.get_phsp_amp(tag,'tag'), self.config.get_phsp_amp(tag,'tag'), phase_correction_sig, phase_correction_tag)

        return ret
    
    def fun_CP(self, tag, Dsign):

        phase_correction = self.pc.eval_corr(self.config.get_phsp_srd(tag))
        ret = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config.get_phsp_amp(tag), self.config.get_phsp_ampbar(tag), phase_correction)
        return ret

    def get_hist_each(self, cato='dks', tag='full'):

        if cato == 'dks':
            self.weights[tag] = self.fun_Kspipi(tag)
        elif cato == 'cp_odd':
            self.weights[tag] = self.fun_CP(tag, -1)
        elif cato == 'cp_even':
            self.weights[tag] = self.fun_CP(tag, 1)

        self.count[tag]={}
        self.bins[tag]={}

        self.count[tag]['s12'], self.bins[tag]['s12'] = np.histogram(self.config.get_phsp_mass(tag)[0], self.nbins, weights=self.weights[tag], range=self.range)
        self.count[tag]['s13'], self.bins[tag]['s13'] = np.histogram(self.config.get_phsp_mass(tag)[1], self.nbins, weights=self.weights[tag], range=self.range)

    def get_hist_sum(self):

        for key in self.plot_list:
            self.bins_sum[key]={}
            self.count_sum[key]={}

            for tag in self.plot_list[key]:
                self.get_hist_each(cato=key, tag=tag)
            self.count_sum[key]['s12'] = np.sum([self.count[tag]['s12'] for tag in self.plot_list[key]], axis=0)
            self.count_sum[key]['s13'] = np.sum([self.count[tag]['s13'] for tag in self.plot_list[key]], axis=0)
            self.bins_sum[key]['s12'] = self.bins[tag]['s12']
            self.bins_sum[key]['s13'] = self.bins[tag]['s13']


    def hist_to_fun(self, count, bins, scale, kind='linear'):
        x = (bins[:-1] + bins[1:])/2

        #Do normalization to count
        count = count*(scale/np.sum(count))
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
        count['s12'], bins['s12'], count['s13'], bins['s13'] = self.hist.count_sum[cato]['s12'], self.hist.bins_sum[cato]['s12'], self.hist.count_sum[cato]['s13'], self.hist.bins_sum[cato]['s13']
        data_hist = {}
        data_hist['s12'], data_hist['s13'], scale = self.make_data_hist(cato=cato)

        for key in count:
            #create data hist
            f = self.hist.hist_to_fun(count=count[key], bins=bins[key],scale=scale)
            #x = np.linspace(bins[key][0], bins[key][-1],bins[key].shape[0]*10)

            fig, ax_main, ax_comparison = plot_data_model_comparison(
                data_hist=data_hist[key],
                #stacked_components=[f_background1, f_background2],
                #stacked_labels=["c0", "c1"],
                unstacked_components=[f],
                unstacked_labels=["Signal"],
                unstacked_colors=["#8EBA42"],
                xlabel=key,
                ylabel="Entries",
                model_sum_kwargs={"show": True, "label": "Model", "color": "navy"},
                comparison="pull",
                range=self.hist.range,
            )
            os.makedirs(self.save_path, exist_ok=True)
            fig.savefig(
                f"{self.save_path}/{cato}_{key}.pdf",
                bbox_inches="tight",
                )

        #Not yet consider the double kspipi yet

    def make_data_hist(self, cato='dks'):
        s12_sig, s13_sig, s12_tag, s13_tag = [], [], [], []
        for tag in self.config._config_data['plot']['plot_sum'][cato]:
            if cato != 'dks':
                s12_sig_i, s13_sig_i = self.config.get_data_mass(tag=tag)
                s12_sig.append(s12_sig_i)
                s13_sig.append(s13_sig_i)
            else:
                s12_i, s13_i = self.config.get_data_mass(tag=tag)
                s12_sig_i, s13_sig_i, s12_tag_i, s13_tag_i = s12_i['sig'], s13_i['sig'], s12_i['tag'], s13_i['tag']
                s12_sig.append(s12_sig_i)
                s13_sig.append(s13_sig_i)
                s12_tag.append(s12_tag_i)
                s13_tag.append(s13_tag_i)

        s12_sig = np.concatenate(s12_sig)
        s13_sig = np.concatenate(s13_sig)
        s12_tag = np.concatenate(s12_tag) if cato == 'dks' else None
        s13_tag = np.concatenate(s13_tag) if cato == 'dks' else None
        scale = len(s12_sig)
        if cato != 'dks':
            data_s12_hist = make_hist(s12_sig, bins=self.hist.nbins, weights=1, range=self.hist.range)
            data_s13_hist = make_hist(s13_sig, bins=self.hist.nbins, weights=1, range=self.hist.range)
            return data_s12_hist, data_s13_hist, scale
        else:
            data_s12_sig_hist = make_hist(s12_sig, bins=self.hist.nbins, weights=1, range=self.hist.range)
            data_s13_sig_hist = make_hist(s13_sig, bins=self.hist.nbins, weights=1, range=self.hist.range)
            data_s12_tag_hist = make_hist(s12_tag, bins=self.hist.nbins, weights=1, range=self.hist.range)
            data_s13_tag_hist = make_hist(s13_tag, bins=self.hist.nbins, weights=1, range=self.hist.range)
            return data_s12_sig_hist, data_s13_sig_hist, data_s12_tag_hist, data_s13_tag_hist, scale