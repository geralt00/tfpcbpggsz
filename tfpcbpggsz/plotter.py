from plothist import plot_two_hist_comparison
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from plothist import make_2d_hist, plot_2d_hist
from matplotlib.colors import ListedColormap
from plothist import get_color_palette
from plothist import make_hist, plot_error_hist
from plothist import plot_hist, add_text
from core import DecayNLLCalculator
import os 
import time

class plotter:

    def __init__(self, fit, **kwargs):
        self._config = fit._config
        self._params = None
        self._fit = fit
        self._params = None
        self._nll_constructor = {}
        self._decays = self._config._n_yields.keys()
        self._path = kwargs.get('path')
        


    
    def plot_and_save(self, nbins=100):
        
        #if no key mc_noeff in the dictionary self._config._amp, load the data

        #if 'mc_noeff' not in self._config._amp.keys():
        self._config.load_data('mc_noeff')
        self._config.load_norm('mc_noeff')
        #self._config.update_yields('mc_noeff')
        self._params = self._fit._fit_result.values

        if len(self._nll_constructor) == 0:
            for decay in self._fit._nll_constructor.keys():
                print(f'Initialising {decay}')
                self._nll_constructor[decay] = DecayNLLCalculator(amp_data=self._config._amp['mc_noeff'], ampbar_data=self._config._ampbar['mc_noeff'], normalisations=self._config._normalisation, mass_pdfs=self._config._mass_pdfs['mc_noeff'],fracDD=self._config._frac_DD_dic, eff_arr=self._config._eff_dic['mc_noeff'], params=self._params, name=decay)
                self._nll_constructor[decay].initialise()


        for decay in self._decays:
        #for decay in ['b2dk_LL_m', 'b2dk_LL_p']:
            time1 = time.time()
            fig, histos = self.plot_decays(decay, nbins)
            os.makedirs(self._path, exist_ok=True)
            fig.savefig(f'{self._path}/{decay}.png')
            fig.clear()
            #not displaying the plots
            plt.close(fig)
            time2 = time.time()
            print(f'{decay} took {time2-time1} seconds to plot')

        

        
    def plot_decays(self, decay='b2dk_LL_p', nbins=100):
        histos={}
        config = self._config
        new_decay = decay
        DecayNLL = self._nll_constructor[new_decay[:-2]]
        print(f'Plotting {new_decay}')
        histos[new_decay] = {}
        histos[new_decay]['data_B_M'] = make_hist(config._Bu_M['data'][new_decay].flatten(),range=(5080, 5800), bins=nbins)
        histos[new_decay]['data_s12'] = make_hist(config._dalitz['data'][new_decay]['s12'], bins=nbins, range=(0.3, 3.2))
        histos[new_decay]['data_s13'] = make_hist(config._dalitz['data'][new_decay]['s13'], bins=nbins, range=(0.3, 3.2))
        histos[new_decay]['data_2d'] = make_2d_hist([config._dalitz['data'][new_decay]['s12'], config._dalitz['data'][new_decay]['s13']], bins=[nbins, nbins], range=[[0.3, 3.2], [0.3, 3.2]])
        for comp in DecayNLL._prod_prob[new_decay].keys():
            histos[new_decay][comp+'_B_M'] = make_hist(config._Bu_M['mc_noeff'][new_decay].flatten(), bins=nbins, weights=config._n_yields[new_decay][comp]* DecayNLL._prod_prob[new_decay][comp].numpy().flatten()/DecayNLL._prod_prob[new_decay][comp].numpy().flatten().sum())
            histos[new_decay][comp+'_s12'] = make_hist(config._dalitz['mc_noeff'][new_decay]['s12'], bins=nbins, range=(0.3, 3.2),weights=config._n_yields[new_decay][comp]* DecayNLL._prod_prob[new_decay][comp].numpy().flatten()/DecayNLL._prod_prob[new_decay][comp].numpy().flatten().sum())
            histos[new_decay][comp+'_s13'] = make_hist(config._dalitz['mc_noeff'][new_decay]['s13'], bins=nbins, range=(0.3, 3.2),weights=config._n_yields[new_decay][comp]* DecayNLL._prod_prob[new_decay][comp].numpy().flatten()/DecayNLL._prod_prob[new_decay][comp].numpy().flatten().sum())
        histos[new_decay]['total_B_M'] = make_hist(config._Bu_M['mc_noeff'][new_decay].flatten(),range=(5080, 5800), bins=nbins, weights=sum([config._n_yields[new_decay][comp]* DecayNLL._prod_prob[new_decay][comp].numpy().flatten()/DecayNLL._prod_prob[new_decay][comp].numpy().flatten().sum() for comp in DecayNLL._prod_prob[new_decay].keys()]))
        histos[new_decay]['total_s12'] = make_hist(config._dalitz['mc_noeff'][new_decay]['s12'], bins=nbins, range=(0.3, 3.2),weights=sum([config._n_yields[new_decay][comp]* DecayNLL._prod_prob[new_decay][comp].numpy().flatten()/DecayNLL._prod_prob[new_decay][comp].numpy().flatten().sum() for comp in DecayNLL._prod_prob[new_decay].keys()]))
        histos[new_decay]['total_s13'] = make_hist(config._dalitz['mc_noeff'][new_decay]['s13'], bins=nbins, range=(0.3, 3.2),weights=sum([config._n_yields[new_decay][comp]* DecayNLL._prod_prob[new_decay][comp].numpy().flatten()/DecayNLL._prod_prob[new_decay][comp].numpy().flatten().sum() for comp in DecayNLL._prod_prob[new_decay].keys()]))
        fig = plt.figure(figsize=(16, 7*2))
        gs = GridSpec(4, 2, figure=fig, height_ratios=[8, 2, 8, 2]) 

        ax0, ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
        ax1_pull, ax2_pull, ax3_pull =  fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])


        ax_pull = [ax1_pull, ax2_pull, ax3_pull]
        scale = [1.02, 1.1, 1.1]
        for i, ax in enumerate(ax_pull):
            pos_pull = ax.get_position()
            pos_new = [pos_pull.x0, pos_pull.y0*scale[i], pos_pull.width, pos_pull.height]  
            ax.set_position(pos_new)

        plot_two_hist_comparison(
            histos[decay]['data_B_M'],
            histos[decay]['total_B_M'],
            xlabel='$M(B^{-})$',  # Use correct LaTeX for B_M
            ylabel="Entries",
            h1_label="Data",  # Renaming labels for clarity
            h2_label="Model",
            comparison="pull",
            ax_main=ax1,  # Assign to the top-left subplot
            ax_comparison=ax1_pull,  # Assign to the bottom-left subplot
            fig = fig
        )
        plot_two_hist_comparison(
            histos[decay]['data_s12'],
            histos[decay]['total_s12'],
            xlabel='$s12$',  # Use correct LaTeX for B_M
            ylabel="Entries",
            h1_label="Data",  # Renaming labels for clarity
            h2_label="Model",
            comparison="pull",
            ax_main=ax2,  # Assign to the top-left subplot
            ax_comparison=ax2_pull,  # Assign to the bottom-left subplot
            fig = fig
        )
        plot_two_hist_comparison(
            histos[decay]['data_s13'],
            histos[decay]['total_s13'],
            xlabel='$s12$',  # Use correct LaTeX for B_M
            ylabel="Entries",
            h1_label="Data",  # Renaming labels for clarity
            h2_label="Model",
            comparison="pull",
            ax_main=ax3,  # Assign to the top-left subplot
            ax_comparison=ax3_pull,  # Assign to the bottom-left subplot
            fig = fig
        )

        #remove the legend 
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax_colorbar = ax0.inset_axes([1, 0.1, 0.02, 0.8])
        cmap = ListedColormap(
            ["white"] + list(get_color_palette("plasma", int(histos[decay]['data_2d'].values().max()) * 2 - 1))
        )
        plot_2d_hist(histos[decay]['data_2d'], ax=ax0, ax_colorbar=ax_colorbar, pcolormesh_kwargs={"cmap": cmap}, fig=fig,square_ax=False)
        ax_colorbar.remove()
        ax0.set_xlabel('$s_{12}$')
        ax0.set_ylabel('$s_{13}$')
        ax0.set_title('Dalitz plot')
        ax0.set_xlim(xmin=0.3)
        ax0.set_ylim(ymin=0.3)
        ax0.set_zorder(1)
        #Reorder the keys
        Keys={}
        Keys['b2dk']= ['comb', 'low_Bs2DKPi', 'low_misID', 'low',  'misid', 'sig']
        Keys['b2dpi']= ['comb', 'low',  'misid','sig']
        plot_error_hist(histos[decay]['data_B_M'], ax=ax1, color="black", label="Toy data")
        plot_hist([histos[decay][comp+'_B_M'] for comp in Keys[decay[:-5]]], ax=ax1, stacked=True, label=[comp for comp in Keys[decay[:-5]]])
        #ax1.set_xlabel('$B_M$')
        ax1.set_ylabel("Entries")
        ax1.get_xaxis().set_visible(False)  # Remove ax1's x-axis
        ax1.set_ylim(ymin=0)
        ax1.set_xlim(xmin=5080, xmax=5800)

        ax1.legend()
        plot_error_hist(histos[decay]['data_s12'], ax=ax2, color="black", label="Toy data")
        plot_hist([histos[decay][comp+'_s12'] for comp in Keys[decay[:-5]]], ax=ax2, stacked=True, label=[comp for comp in Keys[decay[:-5]]])
        #ax2.set_xlabel('$s_{12}$')
        ax2.set_ylabel("Entries")
        ax2.get_xaxis().set_visible(False)  # Remove ax1's x-axis
        ax2.set_ylim(ymin=0.0)
        ax2.set_xlim(xmin=0.3, xmax=3.2)
        ax2.legend()
        plot_error_hist(histos[decay]['data_s13'], ax=ax3, color="black", label="Toy data")
        plot_hist([histos[decay][comp+'_s13'] for comp in Keys[decay[:-5]]], ax=ax3, stacked=True, label=[comp for comp in Keys[decay[:-5]]])
        # sax3.set_xlabel('$s_{13}$')
        ax3.set_ylabel("Entries")
        ax3.get_xaxis().set_visible(False)  # Remove ax1's x-axis
        ax3.set_ylim(ymin=0)
        ax3.set_xlim(xmin=0.3, xmax=3.2)
        ax3.legend()


        return fig, histos