import tensorflow as tf
import numpy as np
from tfpcbpggsz.ulti import get_mass, phsp_to_srd




class PhaseCorrection:


    def __init__(self):
        self.order = 1
        self.order = 0
        self.correctionType = "singleBias"
        self.DEBUG = True
        self.coefficients = {}
        self.nBias_ = None
        self.A_ = [None, None]
        self.epsilon_ = [None, None]
        self.mu_= {'p': [None, None], 'm': [None, None]}
        self.sigma_ = {'p': [None, None], 'm': [None, None]}
        self.doBias_ = False


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
        Returns the phase correction for the given coordinates
        definition:
        \delta_corr = \sum_{i=0}^{order} \sum_{j=0}^{(order-i)/2} C_{2j+1} P_i(z^prime_{+})P_{2j+1}(z^dual_prime_{-})
        """
        if self.correctionType == "singleBias" or self.correctionType == "doubleBias":
            self.do_bias()
            print(f"Setting up phase correction for a {self.correctionType}") if self.DEBUG else None
        elif self.correctionType == "antiSym_legendre":
            for order_i in range(0, self.order):
                for order_j in range(1, self.order-order_i+1):
                    self.coefficients['C'+str(order_i)+str(order_j)] = tf.Variable(np.random.rand(1), dtype=tf.float32)
                    order_i+=1
                    order_j+=2

                         

    def polynomial(self, x, n):
        """
            Returns the nth order polynomial of coordinate x, where the coordinate is based on the SRD coordinates
            x: SRD coordinate[Z', Z'']
        """

        Pi = self.legendre(x[0], n)
        Pj = self.legendre(x[1], n)

        return Pi*Pj

    def legendre(self, x, n):
        if n == 0:
            return tf.ones_like(x)
        elif n == 1:
            return x
        else:
            return (2*n-1)*x*self.legendre(x, n-1)/n - (n-1)*self.legendre(x, n-2)/n

    def gaussianExponential(self, s, mu, sigma):


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
    