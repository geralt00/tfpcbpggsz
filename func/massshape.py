import cupy as cp
import numpy as np
import math

def HORNSdini(m, a, b, csi, shift, sigma, ratio_sigma, fraction_sigma):

    a_new = a
    b_new = b
    B_NEW = (a_new + b_new) / 2.0
    sigma2 = sigma * ratio_sigma

    firstG1 = ((2*(a_new-2*B_NEW+(m-shift))*sigma)/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*sigma*sigma)) - (2*(b_new-2*B_NEW+(m-shift))*sigma)/cp.exp((b_new-(m-shift))*(b_new-(m-shift))/(2*sigma*sigma))+ cp.sqrt(2*math.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma)*cp.erf((-a_new+(m-shift))/(cp.sqrt(2)*sigma))  - cp.sqrt(2*math.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma) * cp.erf((-b_new+(m-shift))/(cp.sqrt(2)*sigma)))/(2*cp.sqrt(2*math.pi))
    secondG1 = (((2*sigma*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(a_new+(m-shift)) + 2*(sigma*sigma)))/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma*sigma))) - (2*sigma*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(b_new + (m-shift)) + 2*(sigma*sigma)))/cp.exp((b_new - (m-shift))*(b_new - (m-shift))/(2*(sigma*sigma))) - cp.sqrt(2*math.pi)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma*sigma))*cp.erf((-a_new + (m-shift))/(cp.sqrt(2)*sigma)) + cp.sqrt(2*math.pi)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma*sigma)) *cp.erf((-b_new + (m-shift))/(cp.sqrt(2)*sigma)))/(2 *cp.sqrt(2*math.pi)))
    CURVEG1 = math.fabs((1-csi)*secondG1 + (b_new*csi - a_new)*firstG1)

    firstG2 = ((2*(a_new-2*B_NEW+(m-shift))*sigma2)/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*sigma2*sigma2)) - (2*(b_new-2*B_NEW+(m-shift))*sigma2)/cp.exp((b_new-(m-shift))*(b_new-(m-shift))/(2*sigma2*sigma2))+ cp.sqrt(2*math.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2*sigma2)*cp.erf((-a_new+(m-shift))/(cp.sqrt(2)*sigma2))  - cp.sqrt(2*math.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2*sigma2) * cp.erf((-b_new+(m-shift))/(cp.sqrt(2)*sigma2)))/(2*cp.sqrt(2*math.pi))
    secondG2 = (((2*sigma2*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(a_new+(m-shift)) + 2*(sigma2*sigma2)))/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma2*sigma2))) - (2*sigma2*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(b_new + (m-shift)) + 2*(sigma2*sigma2)))/cp.exp((b_new - (m-shift))*(b_new - (m-shift))/(2*(sigma2*sigma2))) - cp.sqrt(2*math.pi)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma2*sigma2))*cp.erf((-a_new + (m-shift))/(cp.sqrt(2)*sigma2)) + cp.sqrt(2*math.pi)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma2*sigma2)) *cp.erf((-b_new + (m-shift))/(cp.sqrt(2)*sigma2)))/(2 *cp.sqrt(2*math.pi)))

    CURVEG2 = cp.abs((1-csi)*secondG2 + (b_new*csi - a_new)*firstG2)

    return fraction_sigma*CURVEG1 + (1-fraction_sigma)*CURVEG2