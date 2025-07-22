import numpy as np
from tfpcbpggsz.tensorflow_wrapper import tf


class PhaseCorrection:
    """
    Class for phase correction
    """


    def __init__(self, vm=None):
        self.order = 0
        self.correctionType = "singleBias"
        self.DEBUG = False
        self.coefficients = {}
        self.nBias_ = None
        self.A_ = [None, None]
        self.epsilon_ = [None, None]
        self.mu_= {'p': [None, None], 'm': [None, None]}
        self.sigma_ = {'p': [None, None], 'm': [None, None]}
        self.doBias_ = False
        self.nTerms_ = 0
        self.iTerms_ = []
        self.vm = vm


    def do_bias(self):
        """
        Returns True if the phase correction is for adding bias, set defult parameter first
        """
        if self.correctionType == "singleBias":
            print("Setting up phase correction for a single gaussian bias") if self.DEBUG else None
            self.nBias_ = 1
            self.A_[0] = -1.0
            self.epsilon_[0] = 0.1
            self.mu_['p'][0] = 2.0
            self.sigma_['p'][0] = 0.75
            self.mu_['m'][0] = 0.9
            self.sigma_['m'][0] = 0.25

        else:
            print("Setting up phase correction for a double gaussian bias") if self.DEBUG else None
            self.nBias_ = 2
            self.A_[0] = 1
            self.A_[1] = -1
            self.epsilon_[0] = 0.1
            self.epsilon_[1] = 0.1
            self.mu_['p'][0] = 1.0
            self.mu_['p'][1] = 2.5
            self.sigma_['p'][0] = 0.25
            self.sigma_['p'][1] = 0.25
            self.mu_['m'][0] = 1.25
            self.mu_['m'][1] = 1.25
            self.sigma_['m'][0] = 1.0
            self.sigma_['m'][1] = 1.0
        
        self.doBias_ = True
        return self.doBias_


    def PhaseCorrection(self):
        """
        The phase correction for the given coordinates
        .. math:: \delta_{corr} = \sum_{i=0}^{order} \sum_{j=0}^{(order-i)/2} C_{2j+1} P_i(z'_{+})P_{2j+1}(z^''_{-})

        """

        if self.correctionType == "singleBias" or self.correctionType == "doubleBias":
            self.do_bias()
            print(f"Setting up phase correction for a {self.correctionType}") if self.DEBUG else None
        elif self.correctionType in ["antiSym_legendre", "simple_polynomial"]:
            if self.order != 0 :
                for order_i in range(self.order):
                    for order_j in range(1, self.order - order_i + 1, 2):
                        self.coefficients[f'C_{order_i}_{order_j}'] = tf.Variable(tf.random.normal(shape=(1,), dtype=tf.float64))
                        self.iTerms_.append(f'C_{order_i}_{order_j}')
                        self.nTerms_+=1
                self.vm.variables = self.coefficients
                self.vm.trainable_vars = self.coefficients
                self.vm.set_all(vals=self.coefficients, val_in_fit=True)
                
        

    def set_coefficients(self, **kwargs):
        """
        Sets the coefficients for the phase correction

        coefficients: dict or list
            The coefficients for the phase correction

        """
        #if self.vm is not None:
        #    self.coefficients = self.vm.get_all_dic(trainable_only=True)

        #else:
        coefficients = kwargs.get('coefficients', None)
        if isinstance(coefficients, dict):
            self.coefficients = coefficients
        elif isinstance(coefficients, list) or isinstance(coefficients, np.ndarray) or isinstance(coefficients, tf.Tensor):
            for i in range(coefficients.shape[0]):
                self.coefficients[self.term_to_string(i)] = coefficients[i]

        else:
            raise ValueError("Invalid type for coefficients. Must be dict, list or numpy array; Given: ", type(coefficients))
                         

    def polynomial(self, s, i, j):
        """
            Returns the nth order polynomial of coordinate x, where the coordinate is based on the SRD coordinates
            x: SRD coordinate[Z', Z'']
        """

        if self.correctionType == "antiSym_legendre":
            Pi = self.legendre(s[0], i)
            Pj = self.legendre(s[1], j)
        elif self.correctionType == "simple_polynomial":
            Pi = self.simple_polynomial(s[0], i)
            Pj = self.simple_polynomial(s[1], j)

        return Pi*Pj

    def legendre(self, s, n):
        """
        Returns the nth order Legendre polynomial of the given coordinate
        """
        if n == 0:
            return tf.ones_like(s)
        elif n == 1:
            return s
        else:
            return tf.cast((2 - 1/n) * s * self.legendre(s, n-1) - (1 - 1/n) * self.legendre(s, n-2), tf.float64)

    def simple_polynomial(self, s, n):
        """
        Returns the nth order polynomial of coordinate x, where the coordinate is based on the SRD coordinates
        x: SRD coordinate[Z', Z'']
        """
        if n == 0:
            return tf.ones_like(s)
        else:
            return s**n
        

    def gaussianExponential(self, s, mu, sigma):
        """
        Returns the exponential term of the Gaussian model for the given coordinates
        \exp(-((s-mu)/sigma)^2)
        """

        return tf.cast(((s-mu)/sigma)**2, tf.float64)
    

    def bias(self, coords, index):
        """Calculates the bias value at the given coordinates using a Gaussian model."""

        x, y = coords  # Unpack coordinates for clarity

        # Determine which Gaussian parameters to use based on the relationship between x and y
        is_x_greater = x > y
        mu_x = tf.where(is_x_greater, self.mu_['p'][index], self.mu_['m'][index])
        sigma_x = tf.where(is_x_greater, self.sigma_['p'][index], self.sigma_['m'][index])
        mu_y = tf.where(is_x_greater, self.mu_['m'][index], self.mu_['p'][index])
        sigma_y = tf.where(is_x_greater, self.sigma_['m'][index], self.sigma_['p'][index])


        # Calculate the standardized difference and the Gaussian exponential term
        erf_argument = (x - y) / self.epsilon_[index]
        print("erf_argument:", erf_argument) if self.DEBUG else None
        gaussian_exp = self.gaussianExponential(x, mu_x, sigma_x) + self.gaussianExponential(y, mu_y, sigma_y)

        # Compute and return the bias value
        bias_value = self.A_[index] * tf.math.erf(erf_argument) * tf.math.exp(-gaussian_exp)

        if self.DEBUG:
            print("Bias value:", bias_value)  

        return bias_value
    
    def eval_bias(self, coords):
        """
        Returns the bias for the given coordinates
        """
        bias = 0.0
        for i in range(self.nBias_):
            bias += self.bias(coords, i)
        return bias

    def eval_corr_norm(self, coords):
        """
        Returns the phase correction for the given coordinates
        """
        corr = 0.0
        if self.order != 0:
            for i in range(self.nTerms_):
                tf.print("i:", i,'coeff:', self.coefficients[self.iTerms_[i]]) if self.DEBUG else None
                tf.print("term:",self.iTerms_[i].split('_')[1], self.iTerms_[i].split('_')[2])  if self.DEBUG else None
                corr += self.polynomial(coords, int(self.iTerms_[i].split('_')[1]), int( self.iTerms_[i].split('_')[2])) * self.coefficients[self.iTerms_[i]]
            return corr

        else:
            return None

    def eval_corr(self, coords, reduce_retracing=False):
        """
        Returns the phase correction for the given coordinates
        """

        return tf.function(self.eval_corr_norm,reduce_retracing=reduce_retracing)(coords)  

    def eval_corr_gen(self, coords):
        """
        Returns the phase correction for the given coordinates
        """

        return self.eval_corr_norm(coords)  
        
    def term_to_string(self, i):
        """
        Returns the string representation of the term
        """
        return self.iTerms_[i]
    
    