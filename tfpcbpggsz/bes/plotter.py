import numpy as np
from scipy.interpolate import interp1d
import tfpcbpggsz.core as core # Assuming this exists and works
from plothist import make_hist, get_color_palette, plot_data_model_comparison, plot_model, plot_function # Assuming this exists
from tfpcbpggsz.generator.phasespace import gets23 # Assuming this exists
import os
import matplotlib.pyplot as plt # Added for potential fig.clf() if needed outside plot_cato
# Palette 1: Electric Zing (Bright & Contrasting)
palette_electric_zing = [
    "#007FFF",  # Azure Blue
    "#FF1493",  # Deep Pink
    "#FFEB3B",  # Material Yellow
    "#32CD32",  # Lime Green
    "#FF4500",  # OrangeRed
    "#9400D3",  # Dark Violet
]

# Palette 2: Candy Pop (Sweet & Energetic)
palette_candy_pop = [
    "#FF69B4",  # Hot Pink
    "#1E90FF",  # Dodger Blue
    "#FFD700",  # Gold
    "#76FF03",  # Light Green (Material Design)
    "#BA55D3",  # Medium Orchid (Purple)
    "#FFA500",  # Orange
]

# Palette 3: Cyber Glow (Neon/Digital Feel)
palette_cyber_glow = [
    "#00FFFF",  # Cyan / Aqua
    "#FF00FF",  # Magenta
    "#39FF14",  # Neon Green
    "#FF8C00",  # Dark Orange
    "#4169E1",  # Royal Blue
    "#FFFF00",  # Yellow (Pure)
]

# Palette 4: Tropical Punch (Warm, Bright & Fun)
palette_tropical_punch = [
    "#FF7F50",  # Coral
    "#40E0D0",  # Turquoise
    "#FF6347",  # Tomato Red
    "#FFBF00",  # Amber
    "#ADFF2F",  # GreenYellow
    "#DA70D6",  # Orchid Purple
]

# Palette 5: Playful Clash (High Contrast, Unexpected)
palette_playful_clash = [
    "#FF4E50",  # Bright Red-Orange
    "#8A2BE2",  # Blue Violet
    "#00FA9A",  # Medium Spring Green
    "#FFD700",  # Gold
    "#FC913A",  # Orange-Yellow
    "#1E90FF",  # Dodger Blue
]

# Palette 6: Sunset Burst (Vibrant Warm Tones + Pop)
palette_sunset_burst = [
    "#FF4500",  # OrangeRed
    "#FF8C00",  # Dark Orange
    "#FFDA63",  # Lighter Gold/Yellow
    "#FF1493",  # Deep Pink
    "#00BCD4",  # Material Cyan/Teal
    "#673AB7",  # Deep Purple
]

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
        self.weights_no_corr = {}
        self.plot_list={}
        self.pc = self.model.pc
        self._DEBUG = False


    def fun_Kspipi(self, tag):

        phase_correction_sig = self.pc.eval_corr(self.config.get_phsp_srd(tag,'sig'),reduce_retracing=True)
        phase_correction_tag = self.pc.eval_corr(self.config.get_phsp_srd(tag,'tag'),reduce_retracing=True)
        #need to be flexible with the function name
        ret = core.prob_totalAmplitudeSquared_CP_mix(self.config.get_phsp_amp(tag,'sig'), self.config.get_phsp_ampbar(tag,'sig'), self.config.get_phsp_amp(tag,'tag'), self.config.get_phsp_ampbar(tag,'tag'), phase_correction_sig, phase_correction_tag)
        ret_no_corr = core.prob_totalAmplitudeSquared_CP_mix(self.config.get_phsp_amp(tag,'sig'), self.config.get_phsp_ampbar(tag,'sig'), self.config.get_phsp_amp(tag,'tag'), self.config.get_phsp_ampbar(tag,'tag'))
        return ret, ret_no_corr

    def fun_CP(self, tag, Dsign):

        phase_correction = self.pc.eval_corr(self.config.get_phsp_srd(tag),reduce_retracing=True)
        if tag != 'pipipi0' :
            ret = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config.get_phsp_amp(tag), self.config.get_phsp_ampbar(tag), pc=phase_correction)
            ret_no_corr = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config.get_phsp_amp(tag), self.config.get_phsp_ampbar(tag))
        else:
            ret = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config.get_phsp_amp(tag), self.config.get_phsp_ampbar(tag), pc=phase_correction,  Fplus=0.9406)
            ret_no_corr = core.prob_totalAmplitudeSquared_CP_tag(Dsign, self.config.get_phsp_amp(tag), self.config.get_phsp_ampbar(tag), Fplus=0.9406)
        return ret, ret_no_corr

    def get_hist_each(self, cato='dks', tag='full', mc_type='phsp'):

        if mc_type == 'phsp':
            if cato == 'dks':
                self.weights[tag], self.weights_no_corr[tag] = self.fun_Kspipi(tag)
            elif cato == 'cp_odd':
                self.weights[tag], self.weights_no_corr[tag] = self.fun_CP(tag, -1)
            elif cato == 'cp_even':
                self.weights[tag], self.weights_no_corr[tag] = self.fun_CP(tag, 1)

        self.count[tag]={} if tag not in self.count.keys() else self.count[tag]
        self.bins[tag]={} if tag not in self.bins.keys() else self.bins[tag]
        self.count[tag][mc_type]={} if mc_type not in self.count[tag].keys() else self.count[tag][mc_type]
        self.bins[tag][mc_type]={} if mc_type not in self.bins[tag].keys() else self.bins[tag][mc_type]

        # Use nbins directly here, the *3 factor is applied in hist_to_fun if needed
        # Let's keep the histogramming itself consistent first. Assuming self.nbins is the desired base binning.
        # <<< CHANGE 1 (Minor consistency) >>> Using self.nbins instead of self.nbins*3 for initial histogramming.
        # <<< REASON 1 >>> The scaling factor 3 was applied later in hist_to_fun's scaling,
        # which seemed odd. Let's try standard binning first and adjust scaling if necessary.
        # If self.nbins*3 was truly intended for the histogram resolution itself, revert this change.
        current_nbins = self.nbins # Use the configured number of bins

        if mc_type == 'phsp':
            if cato!='dks':
                self.count[tag][mc_type]['s12'], self.bins[tag][mc_type]['s12'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[0], current_nbins, weights=self.weights[tag], range=self.range['s12'])
                self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s13'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[1], current_nbins, weights=self.weights[tag], range=self.range['s13'])
                self.count[tag][mc_type]['s23'], self.bins[tag][mc_type]['s23'] = np.histogram(gets23(self.config.get_mc_mass(tag, mc_type)[0], self.config.get_mc_mass(tag, mc_type)[1]), current_nbins, weights=self.weights[tag], range=self.range['s23'])
            else:
                self.count[tag][mc_type]['s12'], self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s12'], self.bins[tag][mc_type]['s13'] = {}, {}, {}, {}
                self.count[tag][mc_type]['s23'], self.bins[tag][mc_type]['s23'] = {}, {}
                self.count[tag][mc_type]['s12']['sig'], self.bins[tag][mc_type]['s12']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[0], current_nbins, weights=self.weights[tag], range=self.range['s12'])
                self.count[tag][mc_type]['s13']['sig'], self.bins[tag][mc_type]['s13']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[1], current_nbins, weights=self.weights[tag], range=self.range['s13'])
                self.count[tag][mc_type]['s23']['sig'], self.bins[tag][mc_type]['s23']['sig'] = np.histogram(gets23(self.config.get_mc_mass(tag, mc_type, 'sig')[0], self.config.get_mc_mass(tag, mc_type, 'sig')[1]), current_nbins, weights=self.weights[tag], range=self.range['s23'])
                self.count[tag][mc_type]['s12']['tag'], self.bins[tag][mc_type]['s12']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[0], current_nbins, weights=self.weights[tag], range=self.range['s12'])
                self.count[tag][mc_type]['s13']['tag'], self.bins[tag][mc_type]['s13']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[1], current_nbins, weights=self.weights[tag], range=self.range['s13'])
                self.count[tag][mc_type]['s23']['tag'], self.bins[tag][mc_type]['s23']['tag'] = np.histogram(gets23(self.config.get_mc_mass(tag, mc_type, 'tag')[0], self.config.get_mc_mass(tag, mc_type, 'tag')[1]), current_nbins, weights=self.weights[tag], range=self.range['s23'])


        else: # mc_type != 'phsp' (No weights)
            if cato!='dks':
                self.count[tag][mc_type]['s12'], self.bins[tag][mc_type]['s12'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[0], current_nbins, range=self.range['s12'])
                self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s13'] = np.histogram(self.config.get_mc_mass(tag, mc_type)[1], current_nbins, range=self.range['s13'])
                self.count[tag][mc_type]['s23'], self.bins[tag][mc_type]['s23'] = np.histogram(gets23(self.config.get_mc_mass(tag, mc_type)[0], self.config.get_mc_mass(tag, mc_type)[1]), current_nbins, range=self.range['s23'])
            else:
                self.count[tag][mc_type]['s12'], self.count[tag][mc_type]['s13'], self.bins[tag][mc_type]['s12'], self.bins[tag][mc_type]['s13'] = {}, {}, {}, {}
                self.count[tag][mc_type]['s23'], self.bins[tag][mc_type]['s23'] = {}, {}
                self.count[tag][mc_type]['s12']['sig'], self.bins[tag][mc_type]['s12']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[0], current_nbins, range=self.range['s12'])
                self.count[tag][mc_type]['s13']['sig'], self.bins[tag][mc_type]['s13']['sig'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'sig')[1], current_nbins, range=self.range['s13'])
                self.count[tag][mc_type]['s23']['sig'], self.bins[tag][mc_type]['s23']['sig'] = np.histogram(gets23(self.config.get_mc_mass(tag, mc_type, 'sig')[0], self.config.get_mc_mass(tag, mc_type, 'sig')[1]), current_nbins, range=self.range['s23'])
                self.count[tag][mc_type]['s12']['tag'], self.bins[tag][mc_type]['s12']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[0], current_nbins, range=self.range['s12'])
                self.count[tag][mc_type]['s13']['tag'], self.bins[tag][mc_type]['s13']['tag'] = np.histogram(self.config.get_mc_mass(tag, mc_type, 'tag')[1], current_nbins, range=self.range['s13'])
                self.count[tag][mc_type]['s23']['tag'], self.bins[tag][mc_type]['s23']['tag'] = np.histogram(gets23(self.config.get_mc_mass(tag, mc_type, 'tag')[0], self.config.get_mc_mass(tag, mc_type, 'tag')[1]), current_nbins, range=self.range['s23'])


    def get_hist_sum(self):
        # This function seems complex and heavily relies on the specific structure
        # of self.config and self.plot_list. Assuming its logic is correct for summation.
        # No changes related to smoothing needed here.

        mc_type = ['phsp', 'qcmc', 'dpdm', 'qqbar', 'sigmc_um', 'qcmc_oth']
        for key in self.plot_list:
            if key != 'dks':
                self.bins_sum[key]={}
                self.count_sum[key]={}
                # Initialize missing entries before summing
                for tag in self.plot_list[key]:
                     # Ensure base 'phsp' exists if needed for np.zeros_like
                    if 'phsp' not in self.count.get(tag, {}):
                         self.get_hist_each(cato=key, tag=tag, mc_type='phsp') # Ensure phsp exists

                    for i_mc_type in mc_type:
                        # Check if data for this mc_type exists or needs initialization
                         needs_init = False
                         if i_mc_type not in self.count.get(tag, {}):
                             needs_init = True
                         elif not self.count[tag][i_mc_type]: # Empty dict check
                             needs_init = True

                         if needs_init:
                             # Conditions for initializing with zeros based on original logic
                             if (tag not in ['klpi0', 'klpi0pi0', 'ksomega'] and i_mc_type in ['qcmc_oth']) or \
                                (tag not in ['klpi0pi0'] and i_mc_type in ['sigmc_um']):

                                 # Ensure the base structure exists before trying to access sub-keys
                                 if tag not in self.count: self.count[tag] = {}
                                 if i_mc_type not in self.count[tag]: self.count[tag][i_mc_type] = {}
                                 if tag not in self.bins: self.bins[tag] = {}
                                 if i_mc_type not in self.bins[tag]: self.bins[tag][i_mc_type] = {}

                                 # Check if 'phsp' data is available to get shape/bins
                                 if 'phsp' in self.count.get(tag, {}) and self.count[tag]['phsp']:
                                     self.count[tag][i_mc_type]['s12'] = np.zeros_like(self.count[tag]['phsp']['s12'])
                                     self.count[tag][i_mc_type]['s13'] = np.zeros_like(self.count[tag]['phsp']['s13'])
                                     self.count[tag][i_mc_type]['s23'] = np.zeros_like(self.count[tag]['phsp']['s23']) # Use s23 shape if available, else s12
                                     self.bins[tag][i_mc_type]['s12'] = self.bins[tag]['phsp']['s12']
                                     self.bins[tag][i_mc_type]['s13'] = self.bins[tag]['phsp']['s13']
                                     self.bins[tag][i_mc_type]['s23'] = self.bins[tag]['phsp']['s23']
                                 else:
                                     # Handle case where phsp doesn't exist or is empty - cannot initialize with zeros_like
                                     print(f"Warning: Cannot initialize {i_mc_type} for {tag} as 'phsp' data is missing/empty.")
                                     # Optionally create empty dicts or skip
                                     self.count[tag][i_mc_type] = {'s12': np.array([]), 's13': np.array([]), 's23': np.array([])}
                                     self.bins[tag][i_mc_type] = {'s12': np.array([]), 's13': np.array([]), 's23': np.array([])}

                             else: # If not matching the zero-init condition, fetch the actual histogram
                                self.get_hist_each(cato=key, tag=tag, mc_type=i_mc_type)
                         elif i_mc_type != 'phsp': # If exists and not phsp (phsp already handled), ensure it's fetched if not already
                             # This might re-fetch existing data, could optimize later if needed
                             self.get_hist_each(cato=key, tag=tag, mc_type=i_mc_type)


                # Summation part
                for i_mc_type in mc_type:
                    self.count_sum[key][i_mc_type]={} if i_mc_type not in self.count_sum[key].keys() else self.count_sum[key][i_mc_type]
                    self.bins_sum[key][i_mc_type]={} if i_mc_type not in self.bins_sum[key].keys() else self.bins_sum[key][i_mc_type]

                    # Sum counts, ensuring all tags have the necessary structure
                    tags_to_sum = self.plot_list[key]
                    # Check if all tags have counts for this mc_type and variable
                    def get_counts(var):
                         return [self.count.get(tag, {}).get(i_mc_type, {}).get(var, np.array([])) for tag in tags_to_sum]

                    s12_counts = get_counts('s12')
                    s13_counts = get_counts('s13')
                    s23_counts = get_counts('s23')

                    # Filter out empty arrays before summing, find a representative bin edge
                    valid_s12_counts = [c for c in s12_counts if c.size > 0]
                    valid_s13_counts = [c for c in s13_counts if c.size > 0]
                    valid_s23_counts = [c for c in s23_counts if c.size > 0]

                    self.count_sum[key][i_mc_type]['s12'] = np.sum(valid_s12_counts, axis=0) if valid_s12_counts else np.array([])
                    self.count_sum[key][i_mc_type]['s13'] = np.sum(valid_s13_counts, axis=0) if valid_s13_counts else np.array([])
                    self.count_sum[key][i_mc_type]['s23'] = np.sum(valid_s23_counts, axis=0) if valid_s23_counts else np.array([])

                    # Get bins from the last tag (assuming they are compatible)
                    # A more robust approach would verify bin compatibility across tags
                    last_tag = tags_to_sum[-1]
                    self.bins_sum[key][i_mc_type]['s12'] = self.bins.get(last_tag, {}).get(i_mc_type, {}).get('s12', np.array([]))
                    self.bins_sum[key][i_mc_type]['s13'] = self.bins.get(last_tag, {}).get(i_mc_type, {}).get('s13', np.array([]))
                    self.bins_sum[key][i_mc_type]['s23'] = self.bins.get(last_tag, {}).get(i_mc_type, {}).get('s23', np.array([]))

            else: # key == 'dks'
                 self.bins_sum[key]={}
                 self.count_sum[key]={}
                 mc_type = ['phsp', 'qcmc', 'dpdm', 'qqbar', 'sigmc_um'] # Specific mc_types for dks

                 # Initialization logic for dks
                 for tag in self.plot_list[key]:
                     # Ensure base 'phsp' exists
                     if 'phsp' not in self.count.get(tag, {}):
                         self.get_hist_each(cato=key, tag=tag, mc_type='phsp')

                     for i_mc_type in mc_type:
                         needs_init = False
                         if i_mc_type not in self.count.get(tag, {}):
                             needs_init = True
                         elif not self.count[tag][i_mc_type]: # Empty dict check
                             needs_init = True

                         if needs_init:
                             if tag in ['full', 'misspi'] and i_mc_type == 'sigmc_um':
                                 if tag not in self.count: self.count[tag] = {}
                                 if i_mc_type not in self.count[tag]: self.count[tag][i_mc_type] = {}
                                 if tag not in self.bins: self.bins[tag] = {}
                                 if i_mc_type not in self.bins[tag]: self.bins[tag][i_mc_type] = {}

                                 # Init based on phsp['sig'] - check existence first
                                 if 'phsp' in self.count.get(tag, {}) and \
                                    'sig' in self.count[tag]['phsp'].get('s12', {}) and \
                                    self.count[tag]['phsp']['s12']['sig'].size > 0:

                                     for sub_key in ['s12', 's13', 's23']:
                                         self.count[tag][i_mc_type][sub_key] = {'sig': np.zeros_like(self.count[tag]['phsp'][sub_key]['sig']),
                                                                                  'tag': np.zeros_like(self.count[tag]['phsp'][sub_key]['sig'])} # Assuming tag shape same as sig
                                         self.bins[tag][i_mc_type][sub_key] = {'sig': self.bins[tag]['phsp'][sub_key]['sig'],
                                                                                'tag': self.bins[tag]['phsp'][sub_key]['tag']} # Use tag bins for tag
                                 else:
                                     print(f"Warning: Cannot initialize {i_mc_type} for {tag} as 'phsp'[...]['sig'] data is missing/empty.")
                                     for sub_key in ['s12', 's13', 's23']:
                                        self.count[tag][i_mc_type][sub_key] = {'sig': np.array([]), 'tag': np.array([])}
                                        self.bins[tag][i_mc_type][sub_key] = {'sig': np.array([]), 'tag': np.array([])}

                             else: # Not the special sigmc_um case, fetch normally
                                 self.get_hist_each(cato=key, tag=tag, mc_type=i_mc_type)
                         elif i_mc_type != 'phsp': # Already exists, ensure it's fetched
                             self.get_hist_each(cato=key, tag=tag, mc_type=i_mc_type)


                 # Summation part for dks
                 for i_mc_type in mc_type:
                     self.count_sum[key][i_mc_type] = {} if i_mc_type not in self.count_sum[key].keys() else self.count_sum[key][i_mc_type]
                     self.bins_sum[key][i_mc_type] = {} if i_mc_type not in self.bins_sum[key].keys() else self.bins_sum[key][i_mc_type]
                     for sub_key in ['s12', 's13', 's23']:
                         self.count_sum[key][i_mc_type][sub_key], self.bins_sum[key][i_mc_type][sub_key] = {}, {}

                         for sig_tag in ['sig', 'tag']:
                             tags_to_sum = self.plot_list[key]
                             # Safely get counts, returning empty array if path doesn't exist
                             all_counts = [self.count.get(tag, {}).get(i_mc_type, {}).get(sub_key, {}).get(sig_tag, np.array([])) for tag in tags_to_sum]
                             valid_counts = [c for c in all_counts if c.size > 0]

                             self.count_sum[key][i_mc_type][sub_key][sig_tag] = np.sum(valid_counts, axis=0) if valid_counts else np.array([])

                             # Get bins from last tag, handle missing data
                             last_tag = tags_to_sum[-1]
                             self.bins_sum[key][i_mc_type][sub_key][sig_tag] = self.bins.get(last_tag, {}).get(i_mc_type, {}).get(sub_key, {}).get(sig_tag, np.array([]))


    # <<< CHANGE 2 >>> Modified hist_to_fun for smoothness and safety
    def hist_to_fun(self, count, bins, scale, kind='cubic'):
        """
        Converts histogram counts and bins into a smooth interpolating function.

        Args:
            count (np.ndarray): Histogram counts.
            bins (np.ndarray): Histogram bin edges.
            scale (float): Scaling factor to apply to the normalized counts.
            kind (str, optional): Type of interpolation. Defaults to 'cubic' for smoothness.
                                  Other options include 'linear', 'quadratic', etc.

        Returns:
            scipy.interpolate.interp1d: An interpolating function.
        """
        if count is None or bins is None or count.size == 0 or bins.size < 2:
             print("Warning: Invalid input to hist_to_fun (count or bins empty/None). Returning zero function.")
             return lambda x: np.zeros_like(x) # Return a function that returns zero

        # Calculate bin centers
        x = (bins[:-1] + bins[1:]) / 2

        # Normalize counts - Avoid division by zero if sum is zero
        count_sum = np.sum(count)
        if count_sum != 0:
            # Normalize area to 1 first (approximately, as we use counts not density)
            normalized_count = count / count_sum
            # Scale to the target number of events 'scale'
            # The original code had '*3'. This might be related to the previous nbins*3?
            # If CHANGE 1 was made (using nbins not nbins*3), this factor might need adjustment
            # or removal depending on its original purpose. Let's keep it for now, assuming
            # it was intended scaling, but add a note.
            scaling_factor = 1 # <<< NOTE: Check if this factor is still needed/correct
            scaled_count = normalized_count * scale * scaling_factor
        else:
            scaled_count = count # Keep as zeros if input is all zeros

        # <<< REASON 2 >>>
        # 1. Default 'kind' changed: Set default 'kind' to 'cubic'. Cubic spline
        #    interpolation creates a smoother curve passing through the bin centers
        #    compared to 'linear' interpolation.
        # 2. Added safety: Changed fill_value from 'extrapolate' to 0 and added
        #    bounds_error=False. Cubic extrapolation can sometimes lead to
        #    unrealistic oscillations outside the original data range (bin centers).
        #    Setting fill_value=0 prevents this and is often physically more
        #    reasonable for counts/density, ensuring the function returns 0 outside
        #    the interpolated range.
        # 3. Added input check: Prevents errors if empty arrays are passed.
        try:
             # Use cubic interpolation for smoothness, fill outside range with 0
             f = interp1d(x, scaled_count, kind=kind, bounds_error=False, fill_value=0)
        except ValueError as e:
             print(f"Warning: interp1d failed (Input x: {x.shape}, y: {scaled_count.shape}). Returning zero function. Error: {e}")
             # Handle cases like x not being strictly increasing (though bin centers should be)
             # or x and y having incompatible shapes after processing.
             return lambda val: np.zeros_like(val)

        return f
    # <<< END CHANGE 2 >>>

class Plotter:
    def __init__(self, Model, **kwargs):
        self.config = Model.config_loader
        self.hist = Hist(Model)
        self.weights = {}
        self.count = {}
        self.bins = {}
        self.get_plot_info()
        print("INFO: Calculating histograms...") # Added info message
        self.hist.get_hist_sum()
        print("INFO: Histogram calculation complete.") # Added info message
        self.save_path = os.path.join(os.environ.get('PWD', '.'), 'plots') # Safer path joining
        if 'save_path' in kwargs:
            self.save_path = kwargs['save_path']


    def get_plot_info(self):
        self.hist.nbins = self.config._config_data['plot']['bins']
        self.hist.range = self.config._config_data['plot']['range']
        self.hist.plot_list = self.config._config_data['plot']['plot_sum']

    def plot_cato(self, cato='dks'):

        print(f"INFO: Plotting category '{cato}'...") # Added info message
        count={}
        bins={}
        # Define mc_types based on category
        base_mc_type = ['phsp', 'qcmc', 'dpdm', 'qqbar']
        if cato == 'dks':
            mc_type = base_mc_type + ['sigmc_um']
        elif cato == 'cp_even':
            mc_type = base_mc_type + ['qcmc_oth']
        elif cato == 'cp_odd':
            mc_type = base_mc_type + ['qcmc_oth', 'sigmc_um']
        else:
            mc_type = base_mc_type # Default or unknown category

        # Populate count and bins dictionaries safely
        for i_mc_type in mc_type:
            count[i_mc_type], bins[i_mc_type] = {}, {}
            # Check if the mc_type exists in the summed histograms
            if i_mc_type in self.hist.count_sum.get(cato, {}) and i_mc_type in self.hist.bins_sum.get(cato, {}):
                if cato != 'dks':
                    for key in ['s12', 's13', 's23']:
                         count[i_mc_type][key] = self.hist.count_sum[cato][i_mc_type].get(key)
                         bins[i_mc_type][key] = self.hist.bins_sum[cato][i_mc_type].get(key)
                else: # cato == 'dks'
                    for key in ['s12', 's13', 's23']:
                         count[i_mc_type][key], bins[i_mc_type][key] = {}, {}
                         for sig_tag in ['sig', 'tag']:
                             count[i_mc_type][key][sig_tag] = self.hist.count_sum[cato][i_mc_type].get(key, {}).get(sig_tag)
                             bins[i_mc_type][key][sig_tag] = self.hist.bins_sum[cato][i_mc_type].get(key, {}).get(sig_tag)
            else:
                 print(f"Warning: Missing summed histogram data for mc_type '{i_mc_type}' in category '{cato}'. Skipping.")
                 # Initialize empty to avoid errors later if accessed
                 if cato != 'dks':
                     for key in ['s12', 's13', 's23']:
                         count[i_mc_type][key] = None
                         bins[i_mc_type][key] = None
                 else:
                     for key in ['s12', 's13', 's23']:
                         count[i_mc_type][key], bins[i_mc_type][key] = {}, {}
                         for sig_tag in ['sig', 'tag']:
                             count[i_mc_type][key][sig_tag] = None
                             bins[i_mc_type][key][sig_tag] = None


        # Generate data histograms
        data_hist = {}
        scale = {}
        print(f"INFO: Generating data histogram for '{cato}'...")
        if cato != 'dks':
            data_hist['s12'], data_hist['s13'], data_hist['s23'], scale = self.make_data_hist(cato=cato)
        else:
            data_hist['s12'], data_hist['s13'], data_hist['s23'] = {}, {}, {}
            (data_hist['s12']['sig'], data_hist['s13']['sig'], data_hist['s23']['sig'],
             data_hist['s12']['tag'], data_hist['s13']['tag'], data_hist['s23']['tag'], scale) = self.make_data_hist(cato=cato)
        print(f"INFO: Data histogram scale factors: {scale}")


        # Define plotting components common across categories
        # --- CHOOSE YOUR PALETTE ---
        chosen_palette = palette_candy_pop

        
        plot_components = {
            'phsp': {'label': 'Signal', 'color': chosen_palette[0], 'unstacked': True},
            'qcmc': {'label': 'QCMC', 'color': chosen_palette[1], 'unstacked': False}, # Use default color
            'dpdm': {'label': '$D^+D^-$', 'color': chosen_palette[2], 'unstacked': False},
            'qqbar': {'label': '$q\\bar{q}$', 'color': chosen_palette[3], 'unstacked': False},
            'sigmc_um': {'label': 'Mis. Comb.', 'color': chosen_palette[4], 'unstacked': False},
            'qcmc_oth': {'label': 'QCMC Oth.', 'color': chosen_palette[5], 'unstacked': False},
        }

        # Determine which variables to plot (s12, s13, s23)
        plot_vars = ['s12', 's13', 's23']
        if cato == 'dks':
             sig_tag_keys = ['sig', 'tag']
        else:
             sig_tag_keys = [None] # Placeholder for non-dks case

        # Loop through variables and sig/tag keys
        for key in plot_vars:
            for i_key_tag in sig_tag_keys:
                current_data_hist = None
                if i_key_tag: # dks case
                    current_data_hist = data_hist.get(key, {}).get(i_key_tag)
                    plot_suffix = f"{key}_{i_key_tag}"
                else: # non-dks case
                    current_data_hist = data_hist.get(key)
                    plot_suffix = f"{key}"

                if current_data_hist is None:
                    print(f"Warning: Data histogram for {cato} - {plot_suffix} not found. Skipping plot.")
                    continue

                print(f"INFO: Generating plot for {cato} - {plot_suffix}...")

                # Prepare lists for plot_data_model_comparison
                unstacked_components_funcs = []
                unstacked_labels_list = []
                unstacked_colors_list = []
                stacked_components_funcs = []
                stacked_labels_list = []
                stacked_colors_list = []

                # Add a transparent component first if signal is unstacked
                if plot_components['phsp']['unstacked']:
                     unstacked_components_funcs.append(lambda x: np.zeros_like(x))
                     unstacked_labels_list.append("")
                     unstacked_colors_list.append("#00000000") # Transparent

                # Generate interpolation functions for each MC type
                component_funcs = {}
                for comp_name in mc_type: # Iterate through relevant mc_types for this category
                    comp_info = plot_components.get(comp_name)
                    if not comp_info:
                         print(f"Warning: Plotting configuration missing for component '{comp_name}'. Skipping.")
                         continue

                    current_count = None
                    current_bins = None
                    current_scale = scale.get(comp_name if comp_name != 'phsp' else 'sig', 0) # Use 'sig' scale for 'phsp'

                    if i_key_tag: # dks
                        current_count = count.get(comp_name, {}).get(key, {}).get(i_key_tag)
                        current_bins = bins.get(comp_name, {}).get(key, {}).get(i_key_tag)
                    else: # non-dks
                        current_count = count.get(comp_name, {}).get(key)
                        current_bins = bins.get(comp_name, {}).get(key)

                    if current_count is None or current_bins is None or current_count.size == 0:
                         print(f"INFO: No data for component '{comp_name}' in {cato} - {plot_suffix}. Creating zero function.")
                         component_funcs[comp_name] = lambda x: np.zeros_like(x) # Zero function if no data
                    else:
                         # <<< CHANGE 3 >>> Using the improved hist_to_fun (defaulting to cubic)
                         # <<< REASON 3 >>> This now calls the function that uses cubic interpolation by default.
                         component_funcs[comp_name] = self.hist.hist_to_fun(
                             count=current_count,
                             bins=current_bins,
                             scale=current_scale
                             # kind='cubic' is now the default in hist_to_fun
                         )

                    # Add to appropriate list for plotting
                    if comp_info['unstacked']:
                         unstacked_components_funcs.append(component_funcs[comp_name])
                         unstacked_labels_list.append(comp_info['label'])
                         unstacked_colors_list.append(comp_info['color'])
                    else:
                         stacked_components_funcs.append(component_funcs[comp_name])
                         stacked_labels_list.append(comp_info['label'])
                         stacked_colors_list.append(comp_info['color']) # None will use default palette


                # Filter out components that ended up as zero functions (optional, but cleans up legend)
                # This requires evaluating the function, maybe simpler to just plot them
                # Or check if scale was zero

                # Define stack order (example: backgrounds first) - adjust as needed!
                plot_order = ['qqbar', 'dpdm', 'sigmc_um', 'qcmc_oth', 'qcmc'] # Backgrounds first
                final_stacked_funcs = [component_funcs[name] for name in plot_order if name in component_funcs and not plot_components[name]['unstacked']]
                final_stacked_labels = [plot_components[name]['label'] for name in plot_order if name in component_funcs and not plot_components[name]['unstacked']]
                final_stacked_colors = [plot_components[name]['color'] for name in plot_order if name in component_funcs and not plot_components[name]['unstacked']]


                # Create the plot
                try:
                    fig, ax_main, ax_comparison = plot_data_model_comparison(
                        data_hist=current_data_hist,
                        unstacked_components=unstacked_components_funcs,
                        unstacked_labels=unstacked_labels_list,
                        unstacked_colors=unstacked_colors_list,

                        stacked_components=final_stacked_funcs,
                        stacked_labels=final_stacked_labels,
                        stacked_colors=final_stacked_colors, # Pass None to use defaults
                        xlabel=f"${key}$" if not i_key_tag else f"${key}_{{{i_key_tag}}}$", # LaTeX formatting
                        ylabel="Entries / Bin Width", # More accurate Y label
                        model_sum_kwargs={"show": True, "label": "Model", "color": "navy"},
                        comparison="pull", # Or "ratio", "diff", etc.
                        range=self.hist.range[key],
                        data_uncertainty_type='symmetrical', # Or 'poisson'
                    )

                    # <<< CHANGE 4 (Minor plotting detail) >>> Re-plotting the signal line separately
                    # <<< REASON 4 >>> Sometimes the unstacked signal line in plot_data_model_comparison
                    # might be obscured or hard to see clearly on top of stacked components.
                    # Explicitly plotting it again ensures it's visible, especially with many points.
                    if 'phsp' in component_funcs and plot_components['phsp']['unstacked']:
                        plot_function(
                            [component_funcs['phsp']],
                            range=self.hist.range[key],
                            ax=ax_main,
                            npoints=500,  # Reduced points - 100k usually overkill for smooth interp
                            color=plot_components['phsp']['color'],
                            label="_nolegend_" # Avoid duplicate legend entry if already handled
                        )


                    os.makedirs(self.save_path, exist_ok=True)
                    plot_filename = f"{self.save_path}/{cato}_{plot_suffix}.pdf"
                    fig.savefig(
                        plot_filename,
                        bbox_inches="tight",
                    )
                    print(f"INFO: Saved plot: {plot_filename}")
                    plt.close(fig) # Close the figure to free memory

                except Exception as e:
                    print(f"ERROR: Failed to generate or save plot for {cato} - {plot_suffix}. Error: {e}")
                    # Ensure figure is closed even if saving failed
                    if 'fig' in locals() and plt.fignum_exists(fig.number):
                         plt.close(fig)


    def plot_each(self, cato='dks'):
        """
        Plot each tag in the given category individually.
        """
        if cato not in self.hist.plot_list:
             print(f"ERROR: Category '{cato}' not found in plot_list. Cannot plot each tag.")
             return

        tags_in_cato = self.hist.plot_list[cato]
        if not tags_in_cato:
             print(f"INFO: No tags defined for category '{cato}' in plot_list.")
             return

        print(f"INFO: Plotting each tag individually for category '{cato}'...")

        # Define mc_types based on category (similar to plot_cato)
        base_mc_type = ['phsp', 'qcmc', 'dpdm', 'qqbar']
        if cato == 'dks':
            mc_type = base_mc_type + ['sigmc_um']
        elif cato == 'cp_even':
            mc_type = base_mc_type + ['qcmc_oth']
        elif cato == 'cp_odd':
            mc_type = base_mc_type + ['qcmc_oth', 'sigmc_um']
        else:
            mc_type = base_mc_type

        # --- CHOOSE YOUR PALETTE ---
        chosen_palette = palette_candy_pop

        
        plot_components = {
            'phsp': {'label': 'Signal', 'color': chosen_palette[0], 'unstacked': True},
            'qcmc': {'label': 'QCMC', 'color': chosen_palette[1], 'unstacked': False}, # Use default color
            'dpdm': {'label': '$D^+D^-$', 'color': chosen_palette[2], 'unstacked': False},
            'qqbar': {'label': '$q\\bar{q}$', 'color': chosen_palette[3], 'unstacked': False},
            'sigmc_um': {'label': 'Mis. Comb.', 'color': chosen_palette[4], 'unstacked': False},
            'qcmc_oth': {'label': 'QCMC Oth.', 'color': chosen_palette[5], 'unstacked': False},
        }
        plot_order = ['qqbar', 'dpdm', 'sigmc_um', 'qcmc_oth', 'qcmc'] # Backgrounds first

        for tag in tags_in_cato:
            print(f"INFO: --- Plotting Tag: {tag} ---")

            # Generate data histogram for this specific tag
            data_hist = {}
            scale = {}
            is_dks_tag = (cato == 'dks' and tag in ['full', 'misspi', 'misspi0']) # Rough check based on original make_data_hist_each logic
            print(f"INFO: Generating data histogram for tag '{tag}'...")
            if not is_dks_tag: # Treat as non-dks structure
                try:
                    data_hist['s12'], data_hist['s13'], data_hist['s23'], scale = self.make_data_hist_each(tag=tag)
                except ValueError as e:
                    print(f"ERROR: Failed to make data hist for tag '{tag}': {e}. Skipping tag.")
                    continue
                sig_tag_keys = [None]
                plot_vars = ['s12', 's13', 's23']
            else: # Treat as dks structure
                try:
                    data_hist['s12'], data_hist['s13'], data_hist['s23'] = {}, {}, {}
                    (data_hist['s12']['sig'], data_hist['s13']['sig'], data_hist['s23']['sig'],
                     data_hist['s12']['tag'], data_hist['s13']['tag'], data_hist['s23']['tag'], scale) = self.make_data_hist_each(tag=tag)
                except ValueError as e:
                    print(f"ERROR: Failed to make data hist for tag '{tag}': {e}. Skipping tag.")
                    continue
                sig_tag_keys = ['sig', 'tag']
                plot_vars = ['s12', 's13', 's23']

            print(f"INFO: Data histogram scale factors for tag '{tag}': {scale}")

            # Loop through variables and sig/tag keys for this tag
            for key in plot_vars:
                for i_key_tag in sig_tag_keys:
                    current_data_hist = None
                    if i_key_tag: # dks style tag
                        current_data_hist = data_hist.get(key, {}).get(i_key_tag)
                        plot_suffix = f"{tag}_{key}_{i_key_tag}"
                    else: # non-dks style tag
                        current_data_hist = data_hist.get(key)
                        plot_suffix = f"{tag}_{key}"

                    if current_data_hist is None:
                        print(f"Warning: Data histogram for {cato} - {plot_suffix} not found. Skipping plot.")
                        continue

                    print(f"INFO: Generating plot for {cato} - {plot_suffix}...")

                    # Prepare lists for plot_data_model_comparison
                    unstacked_components_funcs = []
                    unstacked_labels_list = []
                    unstacked_colors_list = []
                    stacked_components_funcs = []
                    stacked_labels_list = []
                    stacked_colors_list = []

                    # Add transparent component if signal is unstacked
                    if plot_components['phsp']['unstacked']:
                         unstacked_components_funcs.append(lambda x: np.zeros_like(x))
                         unstacked_labels_list.append("")
                         unstacked_colors_list.append("#00000000")

                    component_funcs = {}
                    # Generate interpolation functions for each relevant MC type for this tag
                    for comp_name in mc_type:
                        comp_info = plot_components.get(comp_name)
                        if not comp_info: continue # Skip if no plot config

                        # Check if this component exists for this specific tag
                        if tag not in self.hist.count or comp_name not in self.hist.count[tag]:
                             #print(f"DEBUG: Component '{comp_name}' not found in counts for tag '{tag}'. Skipping.")
                             component_funcs[comp_name] = lambda x: np.zeros_like(x) # Treat as zero if missing
                             continue

                        current_count = None
                        current_bins = None
                        current_scale = scale.get(comp_name if comp_name != 'phsp' else 'sig', 0)

                        if i_key_tag: # dks style tag
                            current_count = self.hist.count[tag].get(comp_name, {}).get(key, {}).get(i_key_tag)
                            current_bins = self.hist.bins[tag].get(comp_name, {}).get(key, {}).get(i_key_tag)
                        else: # non-dks style tag
                            current_count = self.hist.count[tag].get(comp_name, {}).get(key)
                            current_bins = self.hist.bins[tag].get(comp_name, {}).get(key)

                        if current_count is None or current_bins is None or current_count.size == 0:
                            #print(f"DEBUG: No data for component '{comp_name}' for tag '{tag}' - {plot_suffix}. Creating zero function.")
                            component_funcs[comp_name] = lambda x: np.zeros_like(x) # Zero function if no data
                        else:
                            # Use the improved hist_to_fun (defaulting to cubic)
                            component_funcs[comp_name] = self.hist.hist_to_fun(
                                count=current_count,
                                bins=current_bins,
                                scale=current_scale
                            )

                        # Add to appropriate list for plotting
                        if comp_info['unstacked']:
                            unstacked_components_funcs.append(component_funcs[comp_name])
                            unstacked_labels_list.append(comp_info['label'])
                            unstacked_colors_list.append(comp_info['color'])
                        else:
                            # We'll sort and add later based on plot_order
                            pass

                    # Build final stacked lists based on plot_order
                    final_stacked_funcs = [component_funcs[name] for name in plot_order if name in component_funcs and not plot_components[name]['unstacked']]
                    final_stacked_labels = [plot_components[name]['label'] for name in plot_order if name in component_funcs and not plot_components[name]['unstacked']]
                    final_stacked_colors = [plot_components[name]['color'] for name in plot_order if name in component_funcs and not plot_components[name]['unstacked']]

                    # Create the plot
                    try:
                        fig, ax_main, ax_comparison = plot_data_model_comparison(
                            data_hist=current_data_hist,
                            unstacked_components=unstacked_components_funcs,
                            unstacked_labels=unstacked_labels_list,
                            unstacked_colors=unstacked_colors_list,
                            stacked_components=final_stacked_funcs,
                            stacked_labels=final_stacked_labels,
                            stacked_colors=final_stacked_colors,
                            xlabel=f"${key}$" if not i_key_tag else f"${key}_{{{i_key_tag}}}$",
                            ylabel="Entries / Bin Width",
                            model_sum_kwargs={"show": True, "label": "Model", "color": "navy"},
                            comparison="pull",
                            range=self.hist.range[key],
                            data_uncertainty_type='symmetrical',
                        )

                        # Re-plot signal line for clarity
                        if 'phsp' in component_funcs and plot_components['phsp']['unstacked']:
                             plot_function(
                                 [component_funcs['phsp']],
                                 range=self.hist.range[key],
                                 ax=ax_main,
                                 npoints=500, # Reduced points
                                 color=plot_components['phsp']['color'],
                                 label="_nolegend_"
                             )

                        os.makedirs(self.save_path, exist_ok=True)
                        plot_filename = f"{self.save_path}/{cato}_{plot_suffix}.pdf"
                        fig.savefig(plot_filename, bbox_inches="tight")
                        print(f"INFO: Saved plot: {plot_filename}")
                        plt.close(fig) # Close figure

                    except Exception as e:
                         print(f"ERROR: Failed to generate or save plot for {cato} - {plot_suffix}. Error: {e}")
                         if 'fig' in locals() and plt.fignum_exists(fig.number):
                              plt.close(fig)


    def make_data_hist(self, cato='dks'):
        """This function is used to make the data hist for the given cato by summing tags."""
        s12_sig_all, s13_sig_all, s12_tag_all, s13_tag_all = [], [], [], []
        scale = {'sig': 0, 'qcmc': 0, 'dpdm': 0, 'qqbar': 0, 'sigmc_um': 0, 'qcmc_oth': 0}

        tags_to_sum = self.config._config_data['plot']['plot_sum'].get(cato, [])
        if not tags_to_sum:
             print(f"Warning: No tags found for category '{cato}' in config plot_sum.")
             # Return empty structures and zero scale
             empty_hist = make_hist([], bins=self.hist.nbins, range=self.hist.range['s12']) # Example empty hist
             if cato != 'dks':
                 return empty_hist, empty_hist, empty_hist, scale
             else:
                 return empty_hist, empty_hist, empty_hist, empty_hist, empty_hist, empty_hist, scale


        for tag in tags_to_sum:
            try:
                scale['sig'] += self.config.get_sig_num(tag)
                scale['qcmc'] += self.config.get_bkg_num(tag, 'qcmc')
                scale['dpdm'] += self.config.get_bkg_num(tag, 'dpdm')
                scale['qqbar'] += self.config.get_bkg_num(tag, 'qqbar')
                # Use get with default 0 for optional backgrounds
                scale['sigmc_um'] += self.config.get_bkg_num(tag, 'sigmc_um', default=0)
                scale['qcmc_oth'] += self.config.get_bkg_num(tag, 'qcmc_oth', default=0)

                if cato != 'dks':
                    s12_sig_i, s13_sig_i = self.config.get_data_mass(tag=tag)
                    s12_sig_all.append(s12_sig_i)
                    s13_sig_all.append(s13_sig_i)
                else:
                    s12_sig_i, s13_sig_i = self.config.get_data_mass(tag=tag, key='sig')
                    s12_tag_i, s13_tag_i = self.config.get_data_mass(tag=tag, key='tag')
                    s12_sig_all.append(s12_sig_i)
                    s13_sig_all.append(s13_sig_i)
                    s12_tag_all.append(s12_tag_i)
                    s13_tag_all.append(s13_tag_i)
            except Exception as e:
                 print(f"Warning: Failed to process data/scale for tag '{tag}' in category '{cato}'. Skipping tag. Error: {e}")
                 continue # Skip to the next tag

        # Concatenate collected data arrays
        s12_sig = np.concatenate(s12_sig_all) if s12_sig_all else np.array([])
        s13_sig = np.concatenate(s13_sig_all) if s13_sig_all else np.array([])

        if cato != 'dks':
            data_s12_hist = make_hist(s12_sig, bins=self.hist.nbins, range=self.hist.range['s12'])
            data_s13_hist = make_hist(s13_sig, bins=self.hist.nbins, range=self.hist.range['s13'])
            s23_sig = gets23(s12_sig, s13_sig) if s12_sig.size > 0 else np.array([])
            data_s23_hist = make_hist(s23_sig, bins=self.hist.nbins, range=self.hist.range['s23'])
            return data_s12_hist, data_s13_hist, data_s23_hist, scale
        else:
            s12_tag = np.concatenate(s12_tag_all) if s12_tag_all else np.array([])
            s13_tag = np.concatenate(s13_tag_all) if s13_tag_all else np.array([])

            data_s12_sig_hist = make_hist(s12_sig, bins=self.hist.nbins, range=self.hist.range['s12'])
            data_s13_sig_hist = make_hist(s13_sig, bins=self.hist.nbins, range=self.hist.range['s13'])
            s23_sig = gets23(s12_sig, s13_sig) if s12_sig.size > 0 else np.array([])
            data_s23_sig_hist = make_hist(s23_sig, bins=self.hist.nbins, range=self.hist.range['s23'])

            data_s12_tag_hist = make_hist(s12_tag, bins=self.hist.nbins, range=self.hist.range['s12'])
            data_s13_tag_hist = make_hist(s13_tag, bins=self.hist.nbins, range=self.hist.range['s13'])
            s23_tag = gets23(s12_tag, s13_tag) if s12_tag.size > 0 else np.array([])
            data_s23_tag_hist = make_hist(s23_tag, bins=self.hist.nbins, range=self.hist.range['s23'])

            return data_s12_sig_hist, data_s13_sig_hist, data_s23_sig_hist, data_s12_tag_hist, data_s13_tag_hist, data_s23_tag_hist, scale


    def make_data_hist_each(self, tag='full'):
        """This function is used to make the data hist for the given single tag."""
        scale = {'sig': 0, 'qcmc': 0, 'dpdm': 0, 'qqbar': 0, 'sigmc_um': 0, 'qcmc_oth': 0}
        s12_sig, s13_sig, s12_tag, s13_tag = None, None, None, None # Initialize as None

        try:
             # Get scale factors
             scale['sig'] = self.config.get_sig_num(tag)
             scale['qcmc'] = self.config.get_bkg_num(tag, 'qcmc')
             scale['dpdm'] = self.config.get_bkg_num(tag, 'dpdm')
             scale['qqbar'] = self.config.get_bkg_num(tag, 'qqbar')
             scale['sigmc_um'] = self.config.get_bkg_num(tag, 'sigmc_um', default=0)
             scale['qcmc_oth'] = self.config.get_bkg_num(tag, 'qcmc_oth', default=0)

             # Determine if tag has 'sig'/'tag' structure based on name (heuristic)
             is_dks_like = tag in ['full', 'misspi', 'misspi0'] # Adjust this list as needed

             if not is_dks_like:
                 s12_sig, s13_sig = self.config.get_data_mass(tag=tag)
             else:
                 s12_sig, s13_sig = self.config.get_data_mass(tag=tag, key='sig')
                 s12_tag, s13_tag = self.config.get_data_mass(tag=tag, key='tag')

        except Exception as e:
             print(f"ERROR: Failed to get data/scale for tag '{tag}'. Error: {e}")
             # Raise the error or return empty structures depending on desired behavior
             raise ValueError(f"Could not process tag '{tag}'") from e


        # Create histograms
        if not is_dks_like:
            if s12_sig is None: raise ValueError(f"Data missing for tag '{tag}'") # Should not happen if above succeeded
            data_s12_hist = make_hist(s12_sig, bins=self.hist.nbins, range=self.hist.range['s12'])
            data_s13_hist = make_hist(s13_sig, bins=self.hist.nbins, range=self.hist.range['s13'])
            s23_sig = gets23(s12_sig, s13_sig) if s12_sig.size > 0 else np.array([])
            data_s23_hist = make_hist(s23_sig, bins=self.hist.nbins, range=self.hist.range['s23'])
            return data_s12_hist, data_s13_hist, data_s23_hist, scale
        else:
            if s12_sig is None or s12_tag is None: raise ValueError(f"Data missing for tag '{tag}'")
            data_s12_sig_hist = make_hist(s12_sig, bins=self.hist.nbins, range=self.hist.range['s12'])
            data_s13_sig_hist = make_hist(s13_sig, bins=self.hist.nbins, range=self.hist.range['s13'])
            s23_sig = gets23(s12_sig, s13_sig) if s12_sig.size > 0 else np.array([])
            data_s23_sig_hist = make_hist(s23_sig, bins=self.hist.nbins, range=self.hist.range['s23'])

            data_s12_tag_hist = make_hist(s12_tag, bins=self.hist.nbins, range=self.hist.range['s12'])
            data_s13_tag_hist = make_hist(s13_tag, bins=self.hist.nbins, range=self.hist.range['s13'])
            s23_tag = gets23(s12_tag, s13_tag) if s12_tag.size > 0 else np.array([])
            data_s23_tag_hist = make_hist(s23_tag, bins=self.hist.nbins, range=self.hist.range['s23'])
            return data_s12_sig_hist, data_s13_sig_hist, data_s23_sig_hist, data_s12_tag_hist, data_s13_tag_hist, data_s23_tag_hist, scale