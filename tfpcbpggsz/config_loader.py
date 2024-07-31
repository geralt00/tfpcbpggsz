from .amp import *
import yaml
from .core import *

def get_mass(p1,p2):
    return ((p1[:,0]+p2[:,0])**2 - (p1[:,1]+p2[:,1])**2 - (p1[:,2]+p2[:,2])**2 - (p1[:,3]+p2[:,3])**2)

def load_int_amp(args):
    p1, p2, p3 = args

    return Kspipi.AMP(p1.tolist(), p2.tolist(), p3.tolist())    

class ConfigLoader:
    def __init__(self, config_file):

        self.file_path = config_file
        self._config_data = None
        self._amp = {}
        self._ampbar = {}
        self._name = {}
        self._cut = None
        self._config_file_name = {}
        self._eff_dic={}
        self._mass_pdfs={}
        self._n_yields = {}
        self._amp = {}
        self._ampbar = {}
        self._dalitz = {}
        self._Bu_M = {}
        self._frac_DD_dic = {}
        self._varDict = {}
        self._normalisation = {}

        

    def load_each(self):
        for type in ['data', 'mc', 'mc_noeff']:
            self._config_file_name[type] = {}

            if type == 'data':
                for decay in self._config_data.get('do_fit'):
                    self._config_file_name[type][decay] = self._config_data.get(type)[decay]        
            else:
                for decay in self._config_data.get('load_mc'):
                    self._config_file_name[type][decay] = self._config_data.get(type)[decay]

            


    def load_config(self):
        """Reads a YAML configuration file and returns its contents as a dictionary."""
        try:
            with open(self.file_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._config_data = config_data
                #return config_data


        except FileNotFoundError:
            print(f"Error: Configuration file '{self.file_path}' not found.")
            return None  # or raise an exception, depending on your needs
        
    
    def get_p4(self, file_name):  
        cut = self._cut


        tree = up.open(file_name)



        branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz"]
        if cut is not None:
            array = tree.arrays(branch_names, cut)
        else:
            array = tree.arrays(branch_names)

        _p1 = np.asarray([array["_1_K0S0_E"], array["_1_K0S0_Px"], array["_1_K0S0_Py"], array["_1_K0S0_Pz"]])
        _p2 = np.asarray([array["_2_pi#_E"], array["_2_pi#_Px"], array["_2_pi#_Py"], array["_2_pi#_Pz"]])
        _p3 = np.asarray([array["_3_pi~_E"], array["_3_pi~_Px"], array["_3_pi~_Py"], array["_3_pi~_Pz"]])
    
        # convert 4*1000 into a vectot<double>
        p1 = np.transpose(_p1)
        p2 = np.transpose(_p2)
        p3 = np.transpose(_p3)

        p1bar = np.hstack((p1[:, :1], np.negative(p1[:, 1:])))
        p2bar = np.hstack((p2[:, :1], np.negative(p2[:, 1:])))
        p3bar = np.hstack((p3[:, :1], np.negative(p3[:, 1:])))

        return p1, p2, p3, p1bar, p2bar, p3bar
    
    def getAmp(self, filename=''):
        
        start_time = time.time()
        p1, p2, p3, p1bar, p2bar, p3bar = self.get_p4(filename)
        amplitude = []
        amplitudeBar = []

        p1_np = np.array(p1)
        p2_np = np.array(p2)
        p3_np = np.array(p3)
        p1bar_np = np.array(p1bar)
        p2bar_np = np.array(p2bar)
        p3bar_np = np.array(p3bar)

        data = [(p1_np[i], p2_np[i], p3_np[i]) for i in range(len(p1_np))]
        amplitude = [load_int_amp(args) for args in data]
        data_bar = [(p1bar_np[i], p3bar_np[i], p2bar_np[i]) for i in range(len(p1bar_np))]
        amplitudeBar = [load_int_amp(args) for args in data_bar]
    
        end_time = time.time()
        print(f'Amplitude loaded in {end_time-start_time} seconds')
        amplitude = np.array(amplitude)
        amplitudeBar = np.negative(np.array(amplitudeBar))

        return amplitude, amplitudeBar
    
    def get_p4_v2(self, file_name):

        cut = self._cut

        tree = up.open(file_name)

        branch_names = ["_1_K0S0_E", "_1_K0S0_Px", "_1_K0S0_Py", "_1_K0S0_Pz",
                         "_2_pi#_E", "_2_pi#_Px", "_2_pi#_Py", "_2_pi#_Pz",
                         "_3_pi~_E", "_3_pi~_Px", "_3_pi~_Py", "_3_pi~_Pz", "B_M"]
        if cut is not None:
            array = tree.arrays(branch_names, cut)
        else:
            array = tree.arrays(branch_names)

        _p1 = np.asarray([array["_1_K0S0_E"], array["_1_K0S0_Px"], array["_1_K0S0_Py"], array["_1_K0S0_Pz"]])
        _p2 = np.asarray([array["_2_pi#_E"], array["_2_pi#_Px"], array["_2_pi#_Py"], array["_2_pi#_Pz"]])
        _p3 = np.asarray([array["_3_pi~_E"], array["_3_pi~_Px"], array["_3_pi~_Py"], array["_3_pi~_Pz"]])
    
        # convert 4*1000 into a vectot<double>
        p1 = np.transpose(_p1)
        p2 = np.transpose(_p2)
        p3 = np.transpose(_p3)

        p1bar = np.hstack((p1[:, :1], np.negative(p1[:, 1:])))
        p2bar = np.hstack((p2[:, :1], np.negative(p2[:, 1:])))
        p3bar = np.hstack((p3[:, :1], np.negative(p3[:, 1:])))

        B_M = np.asarray([array["B_M"]])


        return p1, p2, p3, p1bar, p2bar, p3bar, B_M



    def getMass_v2(self, filename):

        p1, p2, p3, _, _, _, B_M = self.get_p4_v2(filename)


        p1_np = np.array(p1)
        p2_np = np.array(p2)
        p3_np = np.array(p3)


        s12 = get_mass(p1_np, p2_np)
        s13 = get_mass(p1_np, p3_np)

        return s12, s13, B_M
    

    def load_data(self, type='data'):
        
        file_data=self._config_file_name[type]
        files = {}
        self._amp[type] = {}
        self._ampbar[type] = {}
        self._dalitz[type] = {}
        self._Bu_M[type] = {}
        self._eff_dic[type] = {}
        self._mass_pdfs[type] = {}
        
        if type == 'data':
            for key_decay in file_data:
                for mag_type in file_data[key_decay]:
                    for branch in file_data[key_decay][mag_type]['p4']['branch']:
                        new_decay = key_decay+'_'+mag_type+'_'+branch[1:2]
                        self._dalitz[type][new_decay] = {}
                        self._Bu_M[type][new_decay] = {}
                        files[new_decay] = f"{file_data[key_decay][mag_type]['p4']['file']}:{branch}"
                        self._amp[type][new_decay], self._ampbar[type][new_decay] = self.getAmp(files[new_decay])
                        self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13'], self._Bu_M[type][new_decay] = self.getMass_v2(files[new_decay])
        elif type == 'mc_noeff':
            dalitz_mc = np.load(self._config_data.get(type)['dalitz'], allow_pickle=True).item()
            Bu_mc = np.load(self._config_data.get(type)['Bu_mass'], allow_pickle=True).item()
            eff_dic = np.load(self._config_data.get(type)['eff_dic'], allow_pickle=True).item()
            mass_pdfs = np.load(self._config_data.get(type)['mass_pdfs'], allow_pickle=True).item()
            for key_decay in file_data:
                for mag_type in file_data[key_decay]:
                    for branch in ['Bplus', 'Bminus']:
                        new_decay = key_decay+'_'+mag_type+'_'+branch[1:2]
                        self._dalitz[type][new_decay] = {}                    
                        self._Bu_M[type][new_decay] = {}
                        self._amp[type][new_decay], self._ampbar[type][new_decay] = np.load(file_data[key_decay][mag_type]['amp']['amp'][branch], allow_pickle=True), np.load(file_data[key_decay][mag_type]['amp']['ampbar'][branch], allow_pickle=True)
                        self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13'], self._Bu_M[type][new_decay] = dalitz_mc[new_decay]['s12'], dalitz_mc[new_decay]['s13'], Bu_mc[new_decay]
                        self._eff_dic[type][new_decay] = eff_dic[new_decay]
                        self._mass_pdfs[type][new_decay] = mass_pdfs[new_decay]

        elif type == 'mc':
            for key_decay in file_data:
                for mag_type in file_data[key_decay]:
                    for branch in ['Bplus', 'Bminus']:
                        new_decay = key_decay+'_'+mag_type+'_'+branch[1:2]
                        self._amp[type][new_decay], self._ampbar[type][new_decay] = np.load(file_data[key_decay][mag_type]['amp']['amp'][branch], allow_pickle=True), np.load(file_data[key_decay][mag_type]['amp']['ampbar'][branch], allow_pickle=True)


    def load_mass_pdfs(self, type='data'):

        print('INFO: Loading mass pdfs...')
        file_data=self._config_file_name[type]
        config_mass_shape_output = SourceFileLoader('config_mass_shape_output', self._config_data.get('mass_fit_config')).load_module()
        self._varDict = config_mass_shape_output.getconfig()
        comps = ['sig', 'misid', 'comb', 'low', 'low_misID', 'low_Bs2DKPi']
        if type == 'data':
            for key_decay in file_data:
                for mag_type in file_data[key_decay]:
                    pdfs_data = preparePdf_data(self._varDict, key_decay+'_'+mag_type)
                    for branch in file_data[key_decay][mag_type]['p4']['branch']:
                        new_decay = key_decay+'_'+mag_type+'_'+branch[1:2]
                        self._mass_pdfs[type][new_decay] = {}
                        for comp in comps:
                            if key_decay == 'b2dpi' and (comp == 'low_Bs2DKPi' or comp == 'low_misID'): continue
                            self._mass_pdfs[type][new_decay][comp] = pdfs_data[comp](self._Bu_M[type][new_decay])
                        self._eff_dic[type][new_decay] = {}
                        self._eff_dic[type][new_decay]['sig'] = eff_fun(dalitz_transform(self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13']), branch[1:2], key_decay+'_'+mag_type)
                        self._eff_dic[type][new_decay]['comb_a'] = eff_fun(dalitz_transform(self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13']), 'm', (key_decay+'_'+mag_type).replace('dk', 'dpi'))
                        self._eff_dic[type][new_decay]['comb_abar'] = eff_fun(dalitz_transform(self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13']), 'p', (key_decay+'_'+mag_type).replace('dk', 'dpi'))
                        self._eff_dic[type][new_decay]['low'] = eff_fun(dalitz_transform(self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13']), branch[1:2], (key_decay+'_'+mag_type).replace('dk', 'dpi'))
                        if key_decay == 'b2dpi':
                            self._eff_dic[type][new_decay]['misid'] = eff_fun(dalitz_transform(self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13']), branch[1:2], (key_decay+'_'+mag_type).replace('dpi', 'dk'))
                        elif key_decay == 'b2dk':
                            self._eff_dic[type][new_decay]['misid'] = eff_fun(dalitz_transform(self._dalitz[type][new_decay]['s12'], self._dalitz[type][new_decay]['s13']), branch[1:2], (key_decay+'_'+mag_type).replace('dk', 'dpi'))  


    def load_norm(self, type='mc'):

        file_data=self._config_file_name[type]
        for key_decay in file_data:
            for mag_type in file_data[key_decay]:
                for charge in ['p', 'm']:
                    new_decay = key_decay+'_'+mag_type+'_'+charge
                    self._normalisation[new_decay] = Normalisation(self._amp[type], self._ampbar[type], new_decay)
                    self._normalisation[new_decay].initialise()


        

    def update_yields(self,type='data'):
        
        print('INFO: Updating Log...')
        file_data=self._config_file_name[type]
        files = {}
        comp_tag = {'sig': 1, 'comb': 2, 'misid': 3, 'low': 4, 'low_misID': 5, 'low_Bs2DKPi': 6}
        if type == 'data':
            for key_decay in file_data:
                for mag_type in file_data[key_decay]:
                    for branch in file_data[key_decay][mag_type]['p4']['branch']:
                        new_decay = key_decay+'_'+mag_type+'_'+branch[1:2]
                        self._n_yields[new_decay] = {}
                        files[new_decay] = f"{file_data[key_decay][mag_type]['p4']['file']}:{branch}"
                        tree = up.open(files[new_decay])
                        charge_flag = '(Bac_ID>0) '
                        if branch[1:2] == 'm':
                            charge_flag = '(Bac_ID<0) ' 
                        for comp in comp_tag:
                            cuts = f'(B_M>5080) & (tagmode=={comp_tag[comp]}) & {charge_flag}'
                            if comp == 'comb':
                                cuts_flat = cuts + ' & (flav==9)'
                                cuts_DD = cuts + ' & (flav!=9)'
                                array_flat = tree.arrays(f'tagmode',cuts_flat)
                                array_dd = tree.arrays(f'tagmode',cuts_DD)
                                self._frac_DD_dic[new_decay] = tf.cast(len(array_dd)/(len(array_dd)+len(array_flat)), tf.float64)
                            array = tree.arrays('tagmode',cuts)
                            self._varDict[f'n_{comp}_{name_convert(new_decay)}'] = tf.cast(len(array), tf.float64)
                            self._n_yields[new_decay][comp] = self._varDict[f'n_{comp}_{name_convert(new_decay)}'].numpy()

                            if name_convert(new_decay).split('_')[0] == 'DPi' and (comp == 'low_Bs2DKPi' or comp == 'low_misID'): continue
                            self._mass_pdfs[type][new_decay][comp] = self._mass_pdfs[type][new_decay][comp]*self._varDict[f'n_{comp}_{name_convert(new_decay)}']

        else:
            for key_decay in file_data:
                for mag_type in file_data[key_decay]:
                    for branch in ['Bplus', 'Bminus']:
                        new_decay = key_decay+'_'+mag_type+'_'+branch[1:2]
                        #self._n_yields[new_decay] = {}
                        for comp in comp_tag:
                            if name_convert(new_decay).split('_')[0] == 'DPi' and (comp == 'low_Bs2DKPi' or comp == 'low_misID'): continue
                            self._mass_pdfs[type][new_decay][comp] = self._mass_pdfs[type][new_decay][comp]*self._n_yields[new_decay][comp]


    @classmethod
    def register_function(cls, name=None):
        def _f(f):
            my_name = name
            if my_name is None:
                my_name = f.__name__
            if hasattr(cls, my_name):
                warnings.warn("override function {}".format(name))
            setattr(cls, my_name, f)
            return f
        return _f
