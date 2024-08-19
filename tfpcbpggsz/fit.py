import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import iminuit
from tfpcbpggsz.core import DecayNLLCalculator

class fit:

    def __init__(self, config, minimizer='iminuit'):

        self._config = config
        self._params = None
        self._n_params = 0

        self._nll = {}
        self._nll_constructor = {}
        self._nll_data = {}
        self._frac = False
        self._fit_result = None
        self._minimizer = minimizer


    @tf.function
    def neg_like_and_gradient(parms):
        return tfp.math.value_and_gradient(nll, parms)


    @tf.function
    def prod_nll(self, decay, params):
        """
        Calculates the negative log-likelihood for either decays.
    
        Args:
            x: Input parameters for the calculation.
            decay: The type of decay ('b2dk_DD' or 'b2dk_LL').
    
        Returns:
            The negative log-likelihood value.
        """
        
        # Data loading (common for both decays)
        DecayNLL = DecayNLLCalculator(amp_data=self._config._amp['data'], ampbar_data=self._config._ampbar['data'], normalisations=self._config._normalisation, mass_pdfs=self._config._mass_pdfs['data'],fracDD=self._config._frac_DD_dic, eff_arr=self._config._eff_dic['data'], params=params, name=decay)
        DecayNLL.initialise()
        self._nll_constructor[decay] = DecayNLL
        nll = 0
        if self._frac==False:
            nll = DecayNLL._nll[decay + '_p']+DecayNLL._nll[decay + '_m']
    
    
        return nll
    
    @tf.function
    def nll(self, params):
        """
        Calculates the negative log-likelihood for both decays.
    
        Args:
            x: Input parameters for the calculation.
    
        Returns:
            The negative log-likelihood value.
        """

        sim_nll = 0

        for decay in self._config._config_data.get('do_fit'):
            for mag in self._config._config_data.get('data')[decay]:
                sim_nll += self.prod_nll(decay + '_' + mag, params)

        return sim_nll
    
    ##self.prod_nll('b2dpi_LL', params) + self.prod_nll('b2dpi_DD', params) + self.prod_nll('b2dk_LL', params) + self.prod_nll('b2dk_DD', params)
    
    def fit(self, method='L-BFGS-B'):
        """
        Fit the model to the data.
        """
        if len(self._config._config_data.get('load_mc')) == 1:
            self._n_params = 4
            self._params = tf.Variable(np.zeros(self._n_params), shape=(self._n_params), dtype=tf.float64)
        else:
            self._n_params = 6
            self._params = tf.Variable(np.zeros(self._n_params), shape=(self._n_params), dtype=tf.float64)


        if self._minimizer == 'iminuit': 
            m = iminuit.Minuit(self.nll, self._params)
            mg = m.migrad()
            self._fit_result = mg

        if self._minimizer == 'tfp':
                Val = tf.Variable(np.zeros(self._params.shape), shape=(self._params.shape), dtype=tf.float64)
                #optimization
                optim_results = tfp.optimizer.bfgs_minimize(
                    self.neg_like_and_gradient, Val, tolerance=1e-8)

                est_params = optim_results.position.numpy()
                est_serr = np.sqrt(np.diagonal(optim_results.inverse_hessian_estimate.numpy()))
                print("Estimated parameters: ", est_params)
                print("Estimated standard errors: ", est_serr)
                self._fit_result = {'est_params': est_params, 'est_serr': est_serr}
                #with open(f'{logpath}/simfit_output_{index}.npy', 'wb') as f:
                #    np.save(f, fit_results)

        return mg, self._fit_result.values, self._fit_result.errors


    def fit_results(self):
        if self._fit_result is not None:
            if self._minimizer == 'iminuit':
                return self._fit_result.values, self._fit_result.errors

            if self._minimizer == 'tfp':
                return self._fit_result['est_params'], self._fit_result['est_serr']


    def scanner(m, scan_range=1, size=50):
        """
        Perform a scan of the likelihood function.

        Args:
            m: The Minuit object.
            scan_range: The range of the scan.
            size: The size of the scan.

        Returns:
            The points of the scan.
        """

        pts = {}
        for i in range(0, 6):
            for j in range(i + 1, 6):
                a = m.parameters[i]
                b = m.parameters[j]
                a_err = m.errors[i]
                b_err = m.errors[j]
                a_val = m.values[i]
                b_val = m.values[j]
                x0, x1 = i, j  # Assign x0 and x1 for key naming
                pts[f"{x0}_{x1}"] = m.contour(
                    a, b,
                    bound=[
                        [a_val - scan_range * a_err, a_val + scan_range * a_err],
                        [b_val - scan_range * b_err, b_val + scan_range * b_err]
                    ],
                    size=size
                )
        return pts
