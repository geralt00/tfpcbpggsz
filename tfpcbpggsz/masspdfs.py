import tensorflow as tf
import tensorflow_probability as tfp
import math
from tfpcbpggsz.lhcb.common_constants import *


#Repare for the class build

_PI = tf.constant(math.pi, dtype=tf.float64)

def norm_distribution(x, pdf_values):
    # Flatten the tensors for sorting purposes
    x_flat = tf.reshape(x, [-1])
    y_flat = tf.reshape(pdf_values, [-1])

    # Get the sorted indices and sort both x and y
    sorted_indices = tf.argsort(x_flat)
    sorted_x = tf.gather(x_flat, sorted_indices)
    sorted_y = tf.gather(y_flat, sorted_indices)

    # Perform integration using tfp.math.trapz along the second axis
    norm_const = tfp.math.trapz(sorted_y, sorted_x)
    print(norm_const)

    a = tf.convert_to_tensor(pdf_values/norm_const)

    return a



def norm_pdf(x, pdf):
    # Flatten the tensors for sorting purposes
    x_flat = tf.reshape(x, [-1])
    y_flat = tf.reshape(pdf(x), [-1])

    # Get the sorted indices and sort both x and y
    sorted_indices = tf.argsort(x_flat)
    sorted_x = tf.gather(x_flat, sorted_indices)
    sorted_y = tf.gather(y_flat, sorted_indices)

    # Perform integration using tfp.math.trapz along the second axis
    norm_const = tfp.math.trapz(sorted_y, sorted_x)

    a = tf.convert_to_tensor(pdf(x)/norm_const)

    return a


def HORNSdini(m, variables):
    # print(" ")
    # print(" ")
    # print("PRINTIN SOME STUFF IN HORNSdini")
    # print("PRINTIN SOME STUFF IN HORNSdini")
    # print(variables)
    a_new          = tf.cast(variables[2], tf.float64)
    b_new          = tf.cast(variables[3], tf.float64)
    csi            = tf.cast(variables[4], tf.float64)
    shift          = tf.cast(variables[5], tf.float64)
    sigma          = tf.cast(variables[6], tf.float64)
    ratio_sigma    = tf.cast(variables[7], tf.float64)
    fraction_sigma = tf.cast(variables[8], tf.float64)

    sigma2 = tf.cast(sigma * ratio_sigma, tf.float64)
    B_NEW = (a_new + b_new) / 2.0
    constant = tf.constant(2.0, dtype=tf.float64)

    firstG1 = ((constant*(a_new-constant*B_NEW+(m-shift))*sigma)/tf.math.exp((a_new-(m-shift))*(a_new-(m-shift))/(constant*sigma*sigma)) - (constant*(b_new-constant*B_NEW+(m-shift))*sigma)/tf.math.exp((b_new-(m-shift))*(b_new-(m-shift))/(constant*sigma*sigma))+ tf.math.sqrt(constant*_PI)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma)*tf.math.erf((-a_new+(m-shift))/(tf.math.sqrt(constant)*sigma))  - tf.math.sqrt(constant*_PI)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma) * tf.math.erf((-b_new+(m-shift))/(tf.math.sqrt(constant)*sigma)))/(constant*tf.math.sqrt(constant*_PI))
    secondG1 = (((constant*sigma*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - constant*B_NEW*(a_new+(m-shift)) + constant*(sigma*sigma)))/tf.math.exp((a_new-(m-shift))*(a_new-(m-shift))/(constant*(sigma*sigma))) - (constant*sigma*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - constant*B_NEW*(b_new + (m-shift)) + constant*(sigma*sigma)))/tf.math.exp((b_new - (m-shift))*(b_new - (m-shift))/(constant*(sigma*sigma))) - tf.math.sqrt(constant*_PI)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (constant*B_NEW - 3*(m-shift))*(sigma*sigma))*tf.math.erf((-a_new + (m-shift))/(tf.math.sqrt(constant)*sigma)) + tf.math.sqrt(constant*_PI)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (constant*B_NEW - 3*(m-shift))*(sigma*sigma)) *tf.math.erf((-b_new + (m-shift))/(tf.math.sqrt(constant)*sigma)))/(2 *tf.math.sqrt(constant*_PI)))

    CURVEG1 = tf.math.abs((1-csi)*secondG1 + (b_new*csi - a_new)*firstG1)

    firstG2 = ((constant*(a_new-constant*B_NEW+(m-shift))*sigma2)/tf.math.exp((a_new-(m-shift))*(a_new-(m-shift))/(constant*sigma2**2)) - (constant*(b_new-constant*B_NEW+(m-shift))*sigma2)/tf.math.exp((b_new-(m-shift))*(b_new-(m-shift))/(constant*sigma2**2))+ tf.math.sqrt(constant*_PI)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2**2)*tf.math.erf((-a_new+(m-shift))/(tf.math.sqrt(constant)*sigma2))  - tf.math.sqrt(constant*_PI)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2**2) * tf.math.erf((-b_new+(m-shift))/(tf.math.sqrt(constant)*sigma2)))/(constant*tf.math.sqrt(constant*_PI))
    secondG2 = (((constant*sigma2*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - constant*B_NEW*(a_new+(m-shift)) + constant*(sigma2**2)))/tf.math.exp((a_new-(m-shift))*(a_new-(m-shift))/(constant*(sigma2**2))) - (constant*sigma2*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - constant*B_NEW*(b_new + (m-shift)) + constant*(sigma2**2)))/tf.math.exp((b_new - (m-shift))*(b_new - (m-shift))/(constant*(sigma2**2))) - tf.math.sqrt(constant*_PI)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (constant*B_NEW - 3*(m-shift))*(sigma2**2))*tf.math.erf((-a_new + (m-shift))/(tf.math.sqrt(constant)*sigma2)) + tf.math.sqrt(constant*_PI)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (constant*B_NEW - 3*(m-shift))*(sigma2**2)) *tf.math.erf((-b_new + (m-shift))/(tf.math.sqrt(constant)*sigma2)))/(2 *tf.math.sqrt(constant*_PI)))

    CURVEG2 = tf.math.abs((1-csi)*secondG2 + (b_new*csi - a_new)*firstG2)
    return (fraction_sigma*CURVEG1 + (1-fraction_sigma)*CURVEG2)

def CruijffExtended(m, variables):
    # print(" ")
    # print(" ")
    # print("PRINTIN SOME STUFF IN CruijffExtended")
    # print("PRINTIN SOME STUFF IN CruijffExtended")
    # print(variables)
    m0     = tf.cast(variables[2],tf.float64)
    sigmaL = tf.cast(variables[3],tf.float64)
    sigmaR = tf.cast(variables[4],tf.float64)
    alphaL = tf.cast(variables[5],tf.float64)
    alphaR = tf.cast(variables[6],tf.float64)
    beta   = tf.cast(variables[7],tf.float64)
    sigma  = 0.0
    alpha  = 0.0
    dx     = tf.cast(m - m0, tf.float64)
    beta   = tf.cast(beta, tf.float64)

    sigma = tf.where(dx < 0.0, sigmaL, sigmaR)
    alpha = tf.where(dx < 0.0, alphaL, alphaR)
    sigma = tf.cast(sigma, tf.float64)
    alpha = tf.cast(alpha, tf.float64)

    f = tf.cast(2.0*sigma*sigma + alpha*dx*dx, tf.float64)
    return tf.exp(-dx**2 *(1 + beta * dx **2)/f)  

def HORNSdini_misID(m, variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN HORNSdini_misID")
    #    print("PRINTIN SOME STUFF IN HORNSdini_misID")
    #    print(variables)
    a_new = tf.cast(variables[2], tf.float64)
    b_new = tf.cast(variables[3], tf.float64)
    csi = tf.cast(variables[4], tf.float64)
    m1  = tf.cast(variables[5], tf.float64)
    s1  = tf.cast(variables[6], tf.float64)
    m2  = tf.cast(variables[7], tf.float64)
    s2  = tf.cast(variables[8], tf.float64)
    m3  = tf.cast(variables[9], tf.float64)
    s3  = tf.cast(variables[10], tf.float64)
    m4  = tf.cast(variables[11], tf.float64)
    s4  = tf.cast(variables[12], tf.float64)
    f1  = tf.cast(variables[13], tf.float64)
    f2  = tf.cast(variables[14], tf.float64)
    f3  = tf.cast(variables[15], tf.float64)

    B_NEW = (a_new + b_new) / 2.0
    constant = tf.constant(2.0, dtype=tf.float64)



    firstG1 = ((constant * (a_new - constant * B_NEW + (m - m1)) * s1) / tf.math.exp((a_new - (m - m1)) **2 / (constant * s1**2 )) - (constant * (b_new - constant * B_NEW + (m - m1)) * s1) / tf.math.exp((b_new - (m - m1)) **2 / (constant * (s1 **2))) + tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m1)) **2 + s1 ** 2) * tf.math.erf((-a_new + (m - m1)) / (tf.math.sqrt(constant) * s1)) - tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m1)) ** 2 + s1 **2) * tf.math.erf((-b_new + (m - m1)) / (tf.math.sqrt(constant) * s1))) / (constant * tf.math.sqrt(constant * _PI))
    secondG1 = (((2*s1*(a_new**2 + B_NEW**2 + a_new*(m-m1) + (m-m1)**2 - 2*B_NEW*(a_new+(m-m1)) + 2*s1**2 ))/tf.math.exp((a_new-(m-m1))**2 /(constant*(s1**2))) - (2*s1*(b_new**2 + B_NEW**2 + b_new*(m-m1) + (m-m1)**2 - 2*B_NEW*(b_new + (m-m1)) + 2*(s1**2)))/tf.math.exp((b_new - (m-m1))**2 /(constant*(s1**2))) - tf.math.sqrt(2*_PI)*(-((B_NEW - (m-m1))**constant *(m-m1)) + (2*B_NEW - 3*(m-m1))*(s1**2))*tf.math.erf((-a_new + (m-m1))/(tf.math.sqrt(constant)*s1)) + tf.math.sqrt(2*_PI)* (-((B_NEW - (m-m1))**constant *(m-m1)) + (2*B_NEW - 3*(m-m1))*(s1**2)) *tf.math.erf((-b_new + (m-m1))/(tf.math.sqrt(constant)*s1)))/(constant *tf.math.sqrt(2*_PI)))
    CURVEG1 = tf.math.abs((1-csi)*secondG1 + (b_new*csi - a_new)*firstG1)

    firstG2 = ((constant * (a_new - constant * B_NEW + (m - m2)) * s2) / tf.math.exp((a_new - (m - m2)) **2 / (constant * s2**2 )) - (constant * (b_new - constant * B_NEW + (m - m2)) * s2) / tf.math.exp((b_new - (m - m2)) **2 / (constant * (s2 **2))) + tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m2)) **2 + s2 ** 2) * tf.math.erf((-a_new + (m - m2)) / (tf.math.sqrt(constant) * s2)) - tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m2)) ** 2 + s2**2) * tf.math.erf((-b_new + (m - m2)) / (tf.math.sqrt(constant) * s2))) / (constant * tf.math.sqrt(constant * _PI))
    secondG2 = (((2*s2*(a_new**2 + B_NEW**2 + a_new*(m-m2) + (m-m2)**2 - 2*B_NEW*(a_new+(m-m2)) + 2*s2**2 ))/tf.math.exp((a_new-(m-m2))**2 /(constant*(s2**2))) - (2*s2*(b_new**2 + B_NEW**2 + b_new*(m-m2) + (m-m2)**2 - 2*B_NEW*(b_new + (m-m2)) + 2*(s2**2)))/tf.math.exp((b_new - (m-m2))**2 /(constant*(s2**2))) - tf.math.sqrt(2*_PI)*(-((B_NEW - (m-m2))**constant *(m-m2)) + (2*B_NEW - 3*(m-m2))*(s2**2))*tf.math.erf((-a_new + (m-m2))/(tf.math.sqrt(constant)*s2)) + tf.math.sqrt(2*_PI)* (-((B_NEW - (m-m2))**constant *(m-m2)) + (2*B_NEW - 3*(m-m2))*(s2**2)) *tf.math.erf((-b_new + (m-m2))/(tf.math.sqrt(constant)*s2)))/(constant *tf.math.sqrt(2*_PI)))
    CURVEG2 = tf.math.abs((1-csi)*secondG2 + (b_new*csi - a_new)*firstG2)

    firstG3 = ((constant * (a_new - constant * B_NEW + (m - m3)) * s3) / tf.math.exp((a_new - (m - m3)) **2 / (constant * s3**2 )) - (constant * (b_new - constant * B_NEW + (m - m3)) * s3) / tf.math.exp((b_new - (m - m3)) **2 / (constant * (s3 **2))) + tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m3)) **2 + s3 ** 2) * tf.math.erf((-a_new + (m - m3)) / (tf.math.sqrt(constant) * s3)) - tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m3)) ** 2 + s3 **2) * tf.math.erf((-b_new + (m - m3)) / (tf.math.sqrt(constant) * s3))) / (constant * tf.math.sqrt(constant * _PI))
    secondG3 = (((2*s3*(a_new**2 + B_NEW**2 + a_new*(m-m3) + (m-m3)**2 - 2*B_NEW*(a_new+(m-m3)) + 2*s3**2 ))/tf.math.exp((a_new-(m-m3))**2 /(constant*(s3**2))) - (2*s3*(b_new**2 + B_NEW**2 + b_new*(m-m3) + (m-m3)**2 - 2*B_NEW*(b_new + (m-m3)) + 2*(s3**2)))/tf.math.exp((b_new - (m-m3))**2 /(constant*(s3**2))) - tf.math.sqrt(2*_PI)*(-((B_NEW - (m-m3))**constant *(m-m3)) + (2*B_NEW - 3*(m-m3))*(s3**2))*tf.math.erf((-a_new + (m-m3))/(tf.math.sqrt(constant)*s3)) + tf.math.sqrt(2*_PI)* (-((B_NEW - (m-m3))**constant *(m-m3)) + (2*B_NEW - 3*(m-m3))*(s3**2)) *tf.math.erf((-b_new + (m-m3))/(tf.math.sqrt(constant)*s3)))/(constant *tf.math.sqrt(2*_PI)))
    CURVEG3 = tf.math.abs((1-csi)*secondG3 + (b_new*csi - a_new)*firstG3)

    firstG4 = ((constant * (a_new - constant * B_NEW + (m - m4)) * s4) / tf.math.exp((a_new - (m - m4)) **2 / (constant * s4**2 )) - (constant * (b_new - constant * B_NEW + (m - m4)) * s4) / tf.math.exp((b_new - (m - m4)) **2 / (constant * (s4 **2))) + tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m4)) **2 + s4 ** 2) * tf.math.erf((-a_new + (m - m4)) / (tf.math.sqrt(constant) * s4)) - tf.math.sqrt(constant * _PI) * ((B_NEW - (m - m4)) ** 2 + s4 **2) * tf.math.erf((-b_new + (m - m4)) / (tf.math.sqrt(constant) * s4))) / (constant * tf.math.sqrt(constant * _PI))
    secondG4 = (((2*s4*(a_new**2 + B_NEW**2 + a_new*(m-m4) + (m-m4)**2 - 2*B_NEW*(a_new+(m-m4)) + 2*s4**2 ))/tf.math.exp((a_new-(m-m4))**2 /(constant*(s4**2))) - (2*s4*(b_new**2 + B_NEW**2 + b_new*(m-m4) + (m-m4)**2 - 2*B_NEW*(b_new + (m-m4)) + 2*(s4**2)))/tf.math.exp((b_new - (m-m4))**2 /(constant*(s4**2))) - tf.math.sqrt(2*_PI)*(-((B_NEW - (m-m4))**constant *(m-m4)) + (2*B_NEW - 3*(m-m4))*(s4**2))*tf.math.erf((-a_new + (m-m4))/(tf.math.sqrt(constant)*s4)) + tf.math.sqrt(2*_PI)* (-((B_NEW - (m-m4))**constant *(m-m4)) + (2*B_NEW - 3*(m-m4))*(s4**2)) *tf.math.erf((-b_new + (m-m4))/(tf.math.sqrt(constant)*s4)))/(constant *tf.math.sqrt(2*_PI)))
    CURVEG4 = tf.math.abs((1-csi)*secondG4 + (b_new*csi - a_new)*firstG4)


    return tf.math.abs(f1*CURVEG1) + tf.math.abs(f2*CURVEG2) + tf.math.abs(f3*CURVEG3) + tf.math.abs((1-f1-f2-f3)*CURVEG4)

def CBShape(m,variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN CBShape")
    #    print("PRINTIN SOME STUFF IN CBShape")
    #    print(variables)
    m0    = tf.cast(variables[2],tf.float64)
    sigma = tf.cast(variables[3],tf.float64)
    alpha = tf.cast(variables[4],tf.float64)
    n     = tf.cast(variables[5],tf.float64)
    t =tf.cast((m-m0)/sigma, tf.float64)

    if alpha <0:
        t = -t
        pass
    else:
        t = t
        pass

    absAlpha = tf.cast(tf.math.abs(alpha) , tf.float64)
    val_a = tf.cast(tf.math.exp(-0.5*t*t), tf.float64)
   
    a =  tf.cast( tf.math.pow(n/absAlpha,n)*tf.math.exp(-0.5*absAlpha*absAlpha), tf.float64)
    b= tf.cast(n/absAlpha - absAlpha, tf.float64)
    val_b =tf.cast(a/tf.math.pow(b-t, n), tf.float64)
    
    val = tf.where(t >= -absAlpha, val_a, val_b)
    return val

def Exponential(m, variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN EXPONNTIAL")
    #    print("PRINTIN SOME STUFF IN EXPONNTIAL")
    #    print(variables)
    c = tf.cast(variables[2], tf.float64)
    return tf.math.exp(c*m)


def Gaussian(m, variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN Gaussian")
    #    print("PRINTIN SOME STUFF IN Gaussian")
    #    print(variables)
    mu    = tf.cast(variables[2], dtype=tf.float64)
    sigma = tf.cast(variables[3], dtype=tf.float64)
    return (tf.math.exp(-0.5*((m-mu)/sigma)**2))/(sigma*tf.math.sqrt(2.0*_PI))

def HILLdini(m,variables): #a,b,csi,shift,sigma,ratio_sigma,fraction_sigma
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN HILLdini")
    #    print("PRINTIN SOME STUFF IN HILLdini")
    #    print(variables)
    a_new = tf.cast(variables[2], tf.float64)
    b_new = tf.cast(variables[3], tf.float64)
    csi   = tf.cast(variables[4], tf.float64)
    shift = tf.cast(variables[5], tf.float64)
    sigma = tf.cast(variables[6], tf.float64)
    ratio_sigma = tf.cast(variables[7], tf.float64)
    fraction_sigma = tf.cast(variables[8], tf.float64)
    sigma2 = sigma * ratio_sigma
    constant = tf.constant(2.0, dtype=tf.float64)


    firstG1 = (constant*tf.math.exp(-((a_new-(m-shift))*(a_new-(m-shift))/(constant*(sigma*sigma))))*sigma*(b_new-(m-shift))+constant*tf.math.exp(-((b_new-(m-shift))*(b_new-(m-shift))/(constant*(sigma*sigma))))*sigma*(-a_new+(m-shift))-tf.math.sqrt(constant*_PI)*(a_new*b_new+(sigma*sigma)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*tf.math.erf((-a_new+(m-shift))/(tf.math.sqrt(constant)*sigma))+tf.math.sqrt(constant*_PI)*(a_new*b_new+(sigma*sigma)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*tf.math.erf((-b_new+(m-shift))/(tf.math.sqrt(constant)*sigma)))/(constant*tf.math.sqrt(constant*_PI))
    CURVEG1 = tf.math.abs((1-csi)/(b_new - a_new)*m + (b_new*csi - a_new)/(b_new-a_new))*tf.math.abs(firstG1)

    firstG2 = (constant*tf.math.exp(-((a_new-(m-shift))*(a_new-(m-shift))/(constant*(sigma2*sigma2))))*sigma2*(b_new-(m-shift))+constant*tf.math.exp(-((b_new-(m-shift))*(b_new-(m-shift))/(constant*(sigma2*sigma2))))*sigma2*(-a_new+(m-shift))-tf.math.sqrt(constant*_PI)*(a_new*b_new+(sigma2*sigma2)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*tf.math.erf((-a_new+(m-shift))/(tf.math.sqrt(constant)*sigma2))+tf.math.sqrt(constant*_PI)*(a_new*b_new+(sigma2*sigma2)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*tf.math.erf((-b_new+(m-shift))/(tf.math.sqrt(constant)*sigma2)))/(constant*tf.math.sqrt(constant*_PI))
    CURVEG2 = tf.math.abs((1-csi)/(b_new - a_new)*m + (b_new*csi - a_new)/(b_new-a_new))*tf.math.abs(firstG2)

    return tf.math.abs(fraction_sigma*CURVEG1) + tf.math.abs((1-fraction_sigma)*CURVEG2)


def HILLdini_misID(m,variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN HILLdini_misID")
    #    print("PRINTIN SOME STUFF IN HILLdini_misID")
    #    print(variables)
    a_new = tf.cast(variables[2], tf.float64)
    b_new = tf.cast(variables[3], tf.float64)
    csi   = tf.cast(variables[4], tf.float64)
    m1 = tf.cast(variables[ 5], tf.float64)
    s1 = tf.cast(variables[ 6], tf.float64)
    m2 = tf.cast(variables[ 7], tf.float64)
    s2 = tf.cast(variables[ 8], tf.float64)
    m3 = tf.cast(variables[ 9], tf.float64)
    s3 = tf.cast(variables[10], tf.float64)
    m4 = tf.cast(variables[11], tf.float64)
    s4 = tf.cast(variables[12], tf.float64)
    f1 = tf.cast(variables[13], tf.float64)
    f2 = tf.cast(variables[14], tf.float64)
    f3 = tf.cast(variables[15], tf.float64)
    constant = tf.constant(2.0, dtype=tf.float64)


    firstG1 = (constant*tf.math.exp(-((a_new-(m-m1))*(a_new-(m-m1))/(constant*(s1*s1))))*s1*(b_new-(m-m1))+constant*tf.math.exp(-((b_new-(m-m1))*(b_new-(m-m1))/(constant*(s1*s1))))*s1*(-a_new+(m-m1))-tf.math.sqrt(constant*_PI)*(a_new*b_new+(s1*s1)-(a_new+b_new)*(m-m1)+((m-m1)*(m-m1)))*tf.math.erf((-a_new+(m-m1))/(tf.math.sqrt(constant)*s1))+tf.math.sqrt(constant*_PI)*(a_new*b_new+(s1*s1)-(a_new+b_new)*(m-m1)+((m-m1)*(m-m1)))*tf.math.erf((-b_new+(m-m1))/(tf.math.sqrt(constant)*s1)))/(2*tf.math.sqrt(constant*_PI))
    CURVEG1 = tf.math.abs((1-csi)/(b_new-a_new)*(m-m1)  + (b_new*csi - a_new)/(b_new-a_new)  )*tf.math.abs(firstG1)

    firstG2 = (constant*tf.math.exp(-((a_new-(m-m2))*(a_new-(m-m2))/(constant*(s2*s2))))*s2*(b_new-(m-m2))+constant*tf.math.exp(-((b_new-(m-m2))*(b_new-(m-m2))/(constant*(s2*s2))))*s2*(-a_new+(m-m2))-tf.math.sqrt(constant*_PI)*(a_new*b_new+(s2*s2)-(a_new+b_new)*(m-m2)+((m-m2)*(m-m2)))*tf.math.erf((-a_new+(m-m2))/(tf.math.sqrt(constant)*s2))+tf.math.sqrt(constant*_PI)*(a_new*b_new+(s2*s2)-(a_new+b_new)*(m-m2)+((m-m2)*(m-m2)))*tf.math.erf((-b_new+(m-m2))/(tf.math.sqrt(constant)*s2)))/(2*tf.math.sqrt(constant*_PI))
    CURVEG2 = tf.math.abs((1-csi)/(b_new-a_new)*(m-m2)  + (b_new*csi - a_new)/(b_new-a_new)  )*tf.math.abs(firstG2)

    firstG3 = (constant*tf.math.exp(-((a_new-(m-m3))*(a_new-(m-m3))/(constant*(s3*s3))))*s3*(b_new-(m-m3))+constant*tf.math.exp(-((b_new-(m-m3))*(b_new-(m-m3))/(constant*(s3*s3))))*s3*(-a_new+(m-m3))-tf.math.sqrt(constant*_PI)*(a_new*b_new+(s3*s3)-(a_new+b_new)*(m-m3)+((m-m3)*(m-m3)))*tf.math.erf((-a_new+(m-m3))/(tf.math.sqrt(constant)*s3))+tf.math.sqrt(constant*_PI)*(a_new*b_new+(s3*s3)-(a_new+b_new)*(m-m3)+((m-m3)*(m-m3)))*tf.math.erf((-b_new+(m-m3))/(tf.math.sqrt(constant)*s3)))/(2*tf.math.sqrt(constant*_PI))
    CURVEG3 = tf.math.abs((1-csi)/(b_new-a_new)*(m-m3)  + (b_new*csi - a_new)/(b_new-a_new)  )*tf.math.abs(firstG3)

    firstG4 = (constant*tf.math.exp(-((a_new-(m-m4))*(a_new-(m-m4))/(constant*(s4*s4))))*s4*(b_new-(m-m4))+constant*tf.math.exp(-((b_new-(m-m4))*(b_new-(m-m4))/(constant*(s4*s4))))*s4*(-a_new+(m-m4))-tf.math.sqrt(constant*_PI)*(a_new*b_new+(s4*s4)-(a_new+b_new)*(m-m4)+((m-m4)*(m-m4)))*tf.math.erf((-a_new+(m-m4))/(tf.math.sqrt(constant)*s4))+tf.math.sqrt(constant*_PI)*(a_new*b_new+(s4*s4)-(a_new+b_new)*(m-m4)+((m-m4)*(m-m4)))*tf.math.erf((-b_new+(m-m4))/(tf.math.sqrt(constant)*s4)))/(2*tf.math.sqrt(constant*_PI))
    CURVEG4 = tf.math.abs((1-csi)/(b_new-a_new)*(m-m4)  + (b_new*csi - a_new)/(b_new-a_new)  )*tf.math.abs(firstG4)

    return tf.math.abs(f1*CURVEG1) + tf.math.abs(f2*CURVEG2) + tf.math.abs(f3*CURVEG3) + tf.math.abs((1-f1-f2-f3)*CURVEG4)


def HORNSdini_Gaussian(m, variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN HORNSdini_Gaussian")
    #    print("PRINTIN SOME STUFF IN HORNSdini_Gaussian")
    #    print(variables)
    ## HORNSdini
    var_HORNSdini = tf.convert_to_tensor([variables[0],variables[1],variables[2],variables[3],variables[4],variables[5],variables[6],variables[7],variables[8]])
    frac_HORNSdini = tf.cast(variables[9],tf.float64)
    pdf_HORNSdini  = lambda Bu_M: HORNSdini(m,var_HORNSdini)
    norm_HORNSdini = norm_pdf(m, pdf_HORNSdini)
    ### Gaussian
    var_Gaussian  = tf.convert_to_tensor([variables[0],variables[1],variables[10],variables[11]])
    pdf_Gaussian = lambda Bu_M: Gaussian(m,var_Gaussian)
    norm_Gaussian = norm_pdf(m, pdf_Gaussian)
    ## total
    res = frac_HORNSdini*norm_HORNSdini + (1-frac_HORNSdini)*norm_Gaussian
    return res

def HORNSdini_HORNSdini(m, variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN HORNSdini_HORNSdini")
    #    print("PRINTIN SOME STUFF IN HORNSdini_HORNSdini")
    #    print(variables)
    #### first 
    var_I  = tf.convert_to_tensor([variables[0],variables[1],variables[2],variables[3],variables[4],variables[5],variables[6],variables[7],variables[8]])
    frac_I = tf.cast(variables[9],tf.float64)
    pdf_I  = lambda Bu_M: HORNSdini(m,var_I)
    norm_I = norm_pdf(m, pdf_I)
    ### second
    var_II = tf.convert_to_tensor([variables[0],variables[1],variables[10],variables[11],variables[12],variables[13],variables[14],variables[15], variables[16]])
    pdf_II  = lambda Bu_M: HORNSdini(m,var_II)
    norm_II = norm_pdf(m, pdf_II)
    ## total
    res = frac_I*norm_I + (1-frac_I)*norm_II
    return res

def Cruijff_Gaussian(m, variables):
    # print(" ")
    # print(" ")
    # print("PRINTIN SOME STUFF IN Cruijff_Gaussian")
    # print("PRINTIN SOME STUFF IN Cruijff_Gaussian")
    # print(variables)
    ### Cruijff
    # var_Cruijff  = tf.convert_to_tensor([variables[0],variables[1],variables[2],variables[3],variables[4],variables[5],variables[6]])
    var_Cruijff  = [variables[0],variables[1],variables[2],variables[3],variables[4],variables[5],variables[6],variables[7]]
    frac_Cruijff = tf.cast(variables[8],tf.float64)
    pdf_Cruijff  = lambda Bu_M: CruijffExtended(m,var_Cruijff)
    norm_Cruijff = norm_pdf(m, pdf_Cruijff)
    ### Gaussian
    var_Gaussian = tf.convert_to_tensor([variables[0],variables[1],variables[9],variables[10]])
    pdf_Gaussian = lambda Bu_M: Gaussian(m,var_Gaussian)
    norm_Gaussian = norm_pdf(m, pdf_Gaussian)
    ### total
    res = frac_Cruijff*norm_Cruijff + (1-frac_Cruijff)*norm_Gaussian
    return res

def SumCBShape(m, variables):
    #    print(" ")
    #    print(" ")
    #    print("PRINTIN SOME STUFF IN SumCBShape")
    #    print("PRINTIN SOME STUFF IN SumCBShape")
    #    print(variables)
    ### first
    var_I  = tf.convert_to_tensor([variables[0],variables[1],variables[2],variables[3],variables[4],variables[5]])
    frac_I = tf.cast(variables[6],tf.float64)
    pdf_I  = lambda Bu_M: CBShape(m,var_I)
    norm_I = norm_pdf(m, pdf_I)
    ### second
    var_II  = tf.convert_to_tensor([variables[0],variables[1],variables[7],variables[8],variables[9],variables[10]])
    frac_II = tf.cast(variables[11],tf.float64)
    pdf_II  = lambda Bu_M: CBShape(m,var_II)
    norm_II = norm_pdf(m, pdf_II)
    ### total
    res = frac_I*norm_I + frac_II*norm_II
    return res


def preparePdf_data(varDict, mode='b2dk_LL'):
    """
    Import constructed data sets and construct PDFs with RooFit functions.
    PDFs and data sets are saved together in a new RooWorkspace

    Args:
        configDict: a dictionary containing the values of PDF shape parameters
        year: which subset of data to fit, can be any single year of data taking,
              or 'Run1', 'Run2', 'All'.
    """

    pdfList = {}

    print('--- Constructing signal pdfs...')
    varDict['DD_dk_Gauss_frac'] = 1 - varDict['DD_dk_Cruijff_frac']
    varDict['DD_dpi_Gauss_frac'] = 1- varDict['DD_dpi_Cruijff_frac']
    varDict['LL_dk_Gauss_frac'] = 1 - varDict['LL_dk_Cruijff_frac']
    varDict['LL_dpi_Gauss_frac'] = 1 - varDict['LL_dpi_Cruijff_frac']

    pdf_sig_Cruijff_DK_KsPiPi_DD  = lambda Bu_M: CruijffExtended(Bu_M, varDict['signal_mean'], varDict['sigma_dk_DD'], varDict['sigma_dk_DD'], varDict['DD_dk_Cruijff_alpha_L'], varDict['DD_dk_Cruijff_alpha_R'], varDict['Cruijff_beta'])
    pdf_sig_Gauss_DK_KsPiPi_DD    = lambda Bu_M: Gaussian(Bu_M, varDict['signal_mean'], varDict['sigma_dk_DD'])

    pdf_sig_Cruijff_DPi_KsPiPi_DD = lambda Bu_M: CruijffExtended(Bu_M, varDict['signal_mean'], varDict['sigma_dpi_DD'], varDict['sigma_dpi_DD'], varDict['DD_dpi_Cruijff_alpha_L'], varDict['DD_dpi_Cruijff_alpha_R'], varDict['Cruijff_beta'])
    pdf_sig_Gauss_DPi_KsPiPi_DD   = lambda Bu_M: Gaussian(Bu_M, varDict['signal_mean'], varDict['sigma_dpi_DD'])
    pdf_sig_Cruijff_DK_KsPiPi_LL  = lambda Bu_M: CruijffExtended(Bu_M, varDict['signal_mean'], varDict['sigma_dk_LL'], varDict['sigma_dk_LL'], varDict['LL_dk_Cruijff_alpha_L'], varDict['LL_dk_Cruijff_alpha_R'], varDict['Cruijff_beta'])
    pdf_sig_Gauss_DK_KsPiPi_LL    = lambda Bu_M: Gaussian(Bu_M, varDict['signal_mean'], varDict['sigma_dk_LL'])
    pdf_sig_Cruijff_DPi_KsPiPi_LL = lambda Bu_M: CruijffExtended(Bu_M, varDict['signal_mean'], varDict['sigma_dpi_LL'], varDict['sigma_dpi_LL'], varDict['LL_dpi_Cruijff_alpha_L'], varDict['LL_dpi_Cruijff_alpha_R'], varDict['Cruijff_beta'])
    pdf_sig_Gauss_DPi_KsPiPi_LL   = lambda Bu_M: Gaussian(Bu_M, varDict['signal_mean'], varDict['sigma_dpi_LL'])


    if mode == 'b2dk_LL':
        pdfList['sig'] = lambda Bu_M: (norm_pdf(Bu_M, pdf_sig_Cruijff_DK_KsPiPi_LL) * varDict['LL_dk_Cruijff_frac'] +  norm_pdf(Bu_M, pdf_sig_Gauss_DK_KsPiPi_LL) * varDict['LL_dk_Gauss_frac'])
    elif mode == 'b2dpi_LL':
        pdfList['sig'] = lambda Bu_M: (norm_pdf(Bu_M, pdf_sig_Cruijff_DPi_KsPiPi_LL) * varDict['LL_dpi_Cruijff_frac'] +   norm_pdf(Bu_M, pdf_sig_Gauss_DPi_KsPiPi_LL) *  varDict['LL_dpi_Gauss_frac'])
    elif mode == 'b2dk_DD':
        pdfList['sig'] = lambda Bu_M: (norm_pdf(Bu_M, pdf_sig_Cruijff_DK_KsPiPi_DD) * varDict['DD_dk_Cruijff_frac'] +   norm_pdf(Bu_M, pdf_sig_Gauss_DK_KsPiPi_DD) *  varDict['DD_dk_Gauss_frac'])
    elif mode == 'b2dpi_DD':
        pdfList['sig'] = lambda Bu_M: (norm_pdf(Bu_M, pdf_sig_Cruijff_DPi_KsPiPi_DD) * varDict['DD_dpi_Cruijff_frac'] +  norm_pdf(Bu_M, pdf_sig_Gauss_DPi_KsPiPi_DD) * varDict['DD_dpi_Gauss_frac'])    
 

    print('--- Constructing misID pdfs...')
    pdf_misid_CB1_DK_KsPiPi_LL = lambda Bu_M: CBShape(Bu_M, varDict['LL_d2kspp_dpi_to_dk_misID_mean1'], varDict['LL_d2kspp_dpi_to_dk_misID_width1'], varDict['LL_d2kspp_dpi_to_dk_misID_alpha1'], varDict['LL_d2kspp_dpi_to_dk_misID_n1'])
    pdf_misid_CB2_DK_KsPiPi_LL = lambda Bu_M: CBShape(Bu_M, varDict['LL_d2kspp_dpi_to_dk_misID_mean1'], varDict['LL_d2kspp_dpi_to_dk_misID_width2'], varDict['LL_d2kspp_dpi_to_dk_misID_alpha2'], varDict['LL_d2kspp_dpi_to_dk_misID_n1'])
    pdf_misid_CB1_DK_KsPiPi_DD = lambda Bu_M: CBShape(Bu_M, varDict['DD_d2kspp_dpi_to_dk_misID_mean1'], varDict['DD_d2kspp_dpi_to_dk_misID_width1'], varDict['DD_d2kspp_dpi_to_dk_misID_alpha1'], varDict['DD_d2kspp_dpi_to_dk_misID_n1'])
    pdf_misid_CB2_DK_KsPiPi_DD = lambda Bu_M: CBShape(Bu_M, varDict['DD_d2kspp_dpi_to_dk_misID_mean1'], varDict['DD_d2kspp_dpi_to_dk_misID_width2'], varDict['DD_d2kspp_dpi_to_dk_misID_alpha2'], varDict['DD_d2kspp_dpi_to_dk_misID_n1'])
    pdf_misid_CB_DPi_KsPiPi_DD = lambda Bu_M: CBShape(Bu_M, varDict['DD_dk_to_dpi_misID_mean1'], varDict['DD_dk_to_dpi_misID_width1'], varDict['DD_dk_to_dpi_misID_alpha1'], varDict['DD_dk_to_dpi_misID_n1'])
    pdf_misid_CB_DPi_KsPiPi_LL = lambda Bu_M: CBShape(Bu_M, varDict['LL_dk_to_dpi_misID_mean1'], varDict['LL_dk_to_dpi_misID_width1'], varDict['LL_dk_to_dpi_misID_alpha1'], varDict['LL_dk_to_dpi_misID_n1'])
    if mode == 'b2dk_LL':
        pdfList['misid'] = lambda Bu_M: (norm_pdf(Bu_M, pdf_misid_CB1_DK_KsPiPi_LL) * varDict['LL_d2kspp_dpi_to_dk_misID_frac1'] + norm_pdf(Bu_M, pdf_misid_CB2_DK_KsPiPi_LL) * varDict['LL_d2kspp_dpi_to_dk_misID_frac2'])
    elif mode == 'b2dk_DD':
        pdfList['misid'] = lambda Bu_M: (norm_pdf(Bu_M, pdf_misid_CB1_DK_KsPiPi_DD) * varDict['DD_d2kspp_dpi_to_dk_misID_frac1'] + norm_pdf(Bu_M, pdf_misid_CB2_DK_KsPiPi_DD) * varDict['DD_d2kspp_dpi_to_dk_misID_frac2'])
    elif mode == 'b2dpi_LL':
        pdfList['misid'] = lambda Bu_M: norm_pdf(Bu_M, pdf_misid_CB_DPi_KsPiPi_LL) 
    elif mode == 'b2dpi_DD':
        pdfList['misid'] = lambda Bu_M: norm_pdf(Bu_M, pdf_misid_CB_DPi_KsPiPi_DD)

    print('--- Constructing low-mass pdfs...')
    varDict['low_sigma_k_DD'] = varDict['low_sigma_pi_DD']/varDict['low_sigma_pi_over_k_ratio']
    varDict['low_sigma_k_LL'] = varDict['low_sigma_pi_LL']/varDict['low_sigma_pi_over_k_ratio']
    varDict['low_sigma_gamma_dk'] = varDict['low_sigma_gamma']/varDict['low_sigma_pi_over_k_ratio_gamma']

    # B2Dsth (missing pi)
    pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dpi'], varDict['low_b_Bu_Dstar0h_D0pi0_dpi'], varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_pi_LL'], varDict['low_ratio_pi'], varDict['low_f_pi'])
    pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dpi'], varDict['low_b_Bu_Dstar0h_D0pi0_dpi'], varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_pi_DD'], varDict['low_ratio_pi'], varDict['low_f_pi'])
    pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_LL  = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dpi'],  varDict['low_b_Bd_Dstarph_D0pi_dpi'],  varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_pi_LL'], varDict['low_ratio_pi'], varDict['low_f_pi'])
    pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_DD  = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dpi'],  varDict['low_b_Bd_Dstarph_D0pi_dpi'],  varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_pi_DD'], varDict['low_ratio_pi'], varDict['low_f_pi'])
    pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL  = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dk'],  varDict['low_b_Bu_Dstar0h_D0pi0_dk'],  varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_k_LL'],  varDict['low_ratio_pi'], varDict['low_f_pi'])
    pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD  = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dk'],  varDict['low_b_Bu_Dstar0h_D0pi0_dk'],  varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_k_DD'],  varDict['low_ratio_pi'], varDict['low_f_pi'])
    pdf_Bd_Dstarph_D0pi_DK_KsPiPi_LL   = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dk'],   varDict['low_b_Bd_Dstarph_D0pi_dk'],   varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_k_LL'],  varDict['low_ratio_pi'], varDict['low_f_pi'])
    pdf_Bd_Dstarph_D0pi_DK_KsPiPi_DD   = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dk'],   varDict['low_b_Bd_Dstarph_D0pi_dk'],   varDict['low_csi_pi'], varDict['low_global_shift'], varDict['low_sigma_k_DD'],  varDict['low_ratio_pi'], varDict['low_f_pi'])
    # DK misid
    pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dk'], varDict['low_b_Bu_Dstar0h_D0pi0_dk'], varDict['low_csi_pi'], varDict['m1pi_LL'], varDict['s1pi_LL'], varDict['m2pi_LL'], varDict['s2pi_LL'], varDict['m3pi_LL'], varDict['s3pi_LL'], varDict['m4pi_LL'], varDict['s4pi_LL'], varDict['f1pi_LL'], varDict['f2pi_LL'], varDict['f3pi_LL'])
    pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dk'], varDict['low_b_Bu_Dstar0h_D0pi0_dk'], varDict['low_csi_pi'], varDict['m1pi_DD'], varDict['s1pi_DD'], varDict['m2pi_DD'], varDict['s2pi_DD'], varDict['m3pi_DD'], varDict['s3pi_DD'], varDict['m4pi_DD'], varDict['s4pi_DD'], varDict['f1pi_DD'], varDict['f2pi_DD'], varDict['f3pi_DD'])
    pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_LL  = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dk'],  varDict['low_b_Bd_Dstarph_D0pi_dk'],  varDict['low_csi_pi'], varDict['m1pi_LL'], varDict['s1pi_LL'], varDict['m2pi_LL'], varDict['s2pi_LL'], varDict['m3pi_LL'], varDict['s3pi_LL'], varDict['m4pi_LL'], varDict['s4pi_LL'], varDict['f1pi_LL'], varDict['f2pi_LL'], varDict['f3pi_LL'])
    pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_DD  = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dk'],  varDict['low_b_Bd_Dstarph_D0pi_dk'],  varDict['low_csi_pi'], varDict['m1pi_DD'], varDict['s1pi_DD'], varDict['m2pi_DD'], varDict['s2pi_DD'], varDict['m3pi_DD'], varDict['s3pi_DD'], varDict['m4pi_DD'], varDict['s4pi_DD'], varDict['f1pi_DD'], varDict['f2pi_DD'], varDict['f3pi_DD'])
    # DPi misid
    pdf_low_misID_Bu_Dstar0h_D0pi0_DPi_KsPiPi_LL = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dpi'], varDict['low_b_Bu_Dstar0h_D0pi0_dpi'], varDict['low_csi_pi'], varDict['m1pi_pi_LL'], varDict['s1pi_pi_LL'], varDict['m2pi_pi_LL'], varDict['s2pi_pi_LL'], varDict['m3pi_pi_LL'], varDict['s3pi_pi_LL'], varDict['m4pi_pi_LL'], varDict['s4pi_pi_LL'], varDict['f1pi_pi_LL'], varDict['f2pi_pi_LL'], varDict['f3pi_pi_LL'])
    pdf_low_misID_Bu_Dstar0h_D0pi0_DPi_KsPiPi_DD = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0pi0_dpi'], varDict['low_b_Bu_Dstar0h_D0pi0_dpi'], varDict['low_csi_pi'], varDict['m1pi_pi_DD'], varDict['s1pi_pi_DD'], varDict['m2pi_pi_DD'], varDict['s2pi_pi_DD'], varDict['m3pi_pi_DD'], varDict['s3pi_pi_DD'], varDict['m4pi_pi_DD'], varDict['s4pi_pi_DD'], varDict['f1pi_pi_DD'], varDict['f2pi_pi_DD'], varDict['f3pi_pi_DD'])
    pdf_low_misID_Bd_Dstarph_D0pi_DPi_KsPiPi_LL  = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dpi'],  varDict['low_b_Bu_Dstar0h_D0pi0_dpi'], varDict['low_csi_pi'], varDict['m1pi_pi_LL'], varDict['s1pi_pi_LL'], varDict['m2pi_pi_LL'], varDict['s2pi_pi_LL'], varDict['m3pi_pi_LL'], varDict['s3pi_pi_LL'], varDict['m4pi_pi_LL'], varDict['s4pi_pi_LL'], varDict['f1pi_pi_LL'], varDict['f2pi_pi_LL'], varDict['f3pi_pi_LL'])
    pdf_low_misID_Bd_Dstarph_D0pi_DPi_KsPiPi_DD  = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_Bd_Dstarph_D0pi_dpi'],  varDict['low_b_Bu_Dstar0h_D0pi0_dpi'], varDict['low_csi_pi'], varDict['m1pi_pi_DD'], varDict['s1pi_pi_DD'], varDict['m2pi_pi_DD'], varDict['s2pi_pi_DD'], varDict['m3pi_pi_DD'], varDict['s3pi_pi_DD'], varDict['m4pi_pi_DD'], varDict['s4pi_pi_DD'], varDict['f1pi_pi_DD'], varDict['f2pi_pi_DD'], varDict['f3pi_pi_DD'])
    # B2Dsth (missing photon) 
    pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_LL = lambda Bu_M: HILLdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dpi'], varDict['low_b_Bu_Dstar0h_D0gamma_dpi'], varDict['low_csi_gamma'], varDict['low_global_shift'], varDict['low_sigma_gamma'],    varDict['low_ratio_gamma'], varDict['low_f_gamma'])
    pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_DD = lambda Bu_M: HILLdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dpi'], varDict['low_b_Bu_Dstar0h_D0gamma_dpi'], varDict['low_csi_gamma'], varDict['low_global_shift'], varDict['low_sigma_gamma'],    varDict['low_ratio_gamma'], varDict['low_f_gamma'])
    pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL  = lambda Bu_M: HILLdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dk'],  varDict['low_b_Bu_Dstar0h_D0gamma_dk'],  varDict['low_csi_gamma'], varDict['low_global_shift'], varDict['low_sigma_gamma_dk'], varDict['low_ratio_gamma'], varDict['low_f_gamma'])
    pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD  = lambda Bu_M: HILLdini(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dk'],  varDict['low_b_Bu_Dstar0h_D0gamma_dk'],  varDict['low_csi_gamma'], varDict['low_global_shift'], varDict['low_sigma_gamma_dk'], varDict['low_ratio_gamma'], varDict['low_f_gamma'])
    # DK misid
    pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL = lambda Bu_M: HILLdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dk'], varDict['low_b_Bu_Dstar0h_D0gamma_dk'], varDict['low_csi_gamma'], varDict['m1ga'], varDict['s1ga'], varDict['m2ga'], varDict['s2ga'], varDict['m3ga'], varDict['s3ga'], varDict['m4ga'], varDict['s4ga'], varDict['f1ga'], varDict['f2ga'], varDict['f3ga'])
    pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD = lambda Bu_M: HILLdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dk'], varDict['low_b_Bu_Dstar0h_D0gamma_dk'], varDict['low_csi_gamma'], varDict['m1ga'], varDict['s1ga'], varDict['m2ga'], varDict['s2ga'], varDict['m3ga'], varDict['s3ga'], varDict['m4ga'], varDict['s4ga'], varDict['f1ga'], varDict['f2ga'], varDict['f3ga'])
    # DPi misid
    pdf_low_misID_Bu_Dstar0h_D0gamma_DPi_KsPiPi_LL = lambda Bu_M: HILLdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dpi'], varDict['low_b_Bu_Dstar0h_D0gamma_dpi'], varDict['low_csi_gamma'], varDict['m1ga_pi'], varDict['s1ga_pi'], varDict['m2ga_pi'], varDict['s2ga_pi'], varDict['m3ga_pi'], varDict['s3ga_pi'], varDict['m4ga_pi'], varDict['s4ga_pi'], varDict['f1ga_pi'], varDict['f2ga_pi'], varDict['f3ga_pi'])
    pdf_low_misID_Bu_Dstar0h_D0gamma_DPi_KsPiPi_DD = lambda Bu_M: HILLdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dpi'], varDict['low_b_Bu_Dstar0h_D0gamma_dpi'], varDict['low_csi_gamma'], varDict['m1ga_pi'], varDict['s1ga_pi'], varDict['m2ga_pi'], varDict['s2ga_pi'], varDict['m3ga_pi'], varDict['s3ga_pi'], varDict['m4ga_pi'], varDict['s4ga_pi'], varDict['f1ga_pi'], varDict['f2ga_pi'], varDict['f3ga_pi'])

    # B2Dhpi
    # DPi
    pdf_B2Dpipi_1_DPi_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_I_B2Dpipi'],  varDict['low_b_I_B2Dpipi'],  varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_I_B2Dpipi'],  varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_1_DPi_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_I_B2Dpipi'],  varDict['low_b_I_B2Dpipi'],  varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_I_B2Dpipi'],  varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_2_DPi_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_II_B2Dpipi'], varDict['low_b_II_B2Dpipi'], varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_II_B2Dpipi'], varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_2_DPi_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_II_B2Dpipi'], varDict['low_b_II_B2Dpipi'], varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_II_B2Dpipi'], varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_DPi_KsPiPi_LL   = lambda Bu_M: norm_pdf(Bu_M, pdf_B2Dpipi_1_DPi_KsPiPi_LL) * varDict['low_frac_B2Dpipi'] + norm_pdf(Bu_M, pdf_B2Dpipi_2_DPi_KsPiPi_LL) * (1- varDict['low_frac_B2Dpipi'])
    pdf_B2Dpipi_DPi_KsPiPi_DD   = lambda Bu_M: norm_pdf(Bu_M, pdf_B2Dpipi_1_DPi_KsPiPi_DD) * varDict['low_frac_B2Dpipi'] + norm_pdf(Bu_M, pdf_B2Dpipi_2_DPi_KsPiPi_DD) * (1- varDict['low_frac_B2Dpipi'])
    # DK
    pdf_B2DKpi_1_DK_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_B2DKpi'], varDict['low_b_B2DKpi'], varDict['low_csi_B2DKpi'], varDict['low_global_shift'], varDict['low_sigma_B2DKpi'], varDict['low_ratio_B2DKpi'], varDict['low_f_B2DKpi'])
    pdf_B2DKpi_1_DK_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_B2DKpi'], varDict['low_b_B2DKpi'], varDict['low_csi_B2DKpi'], varDict['low_global_shift'], varDict['low_sigma_B2DKpi'], varDict['low_ratio_B2DKpi'], varDict['low_f_B2DKpi'])
    pdf_B2DKpi_2_DK_KsPiPi_LL = lambda Bu_M: Gaussian(Bu_M, varDict['low_mu_B2DKpi'], varDict['low_sigma_gaus_B2DKpi'])
    pdf_B2DKpi_2_DK_KsPiPi_DD = lambda Bu_M: Gaussian(Bu_M, varDict['low_mu_B2DKpi'], varDict['low_sigma_gaus_B2DKpi'])
    pdf_B2DKpi_DK_KsPiPi_LL   = lambda Bu_M: norm_pdf(Bu_M, pdf_B2DKpi_1_DK_KsPiPi_LL) * varDict['low_frac_B2DKpi'] + pdf_B2DKpi_2_DK_KsPiPi_LL(Bu_M) * (1- varDict['low_frac_B2DKpi'])
    pdf_B2DKpi_DK_KsPiPi_DD   = lambda Bu_M: norm_pdf(Bu_M, pdf_B2DKpi_1_DK_KsPiPi_DD) * varDict['low_frac_B2DKpi'] + pdf_B2DKpi_2_DK_KsPiPi_DD(Bu_M) * (1- varDict['low_frac_B2DKpi'])
    # DK misid
    pdf_low_misID_B2Dpipi_DK_KsPiPi_LL = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_B2Dpipi_misID'], varDict['low_b_B2Dpipi_misID'], varDict['low_csi_B2Dpipi'], varDict['low_m1_B2Dpipi_misID'], varDict['low_s1_B2Dpipi_misID'], varDict['low_m2_B2Dpipi_misID'], varDict['low_s2_B2Dpipi_misID'], varDict['low_m3_B2Dpipi_misID'], varDict['low_s3_B2Dpipi_misID'], varDict['low_m4_B2Dpipi_misID'], varDict['low_s4_B2Dpipi_misID'], varDict['low_f1_B2Dpipi_misID'], varDict['low_f2_B2Dpipi_misID'], varDict['low_f3_B2Dpipi_misID'])
    pdf_low_misID_B2Dpipi_DK_KsPiPi_DD = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_B2Dpipi_misID'], varDict['low_b_B2Dpipi_misID'], varDict['low_csi_B2Dpipi'], varDict['low_m1_B2Dpipi_misID'], varDict['low_s1_B2Dpipi_misID'], varDict['low_m2_B2Dpipi_misID'], varDict['low_s2_B2Dpipi_misID'], varDict['low_m3_B2Dpipi_misID'], varDict['low_s3_B2Dpipi_misID'], varDict['low_m4_B2Dpipi_misID'], varDict['low_s4_B2Dpipi_misID'], varDict['low_f1_B2Dpipi_misID'], varDict['low_f2_B2Dpipi_misID'], varDict['low_f3_B2Dpipi_misID'])

    # Bs pdf
    pdf_low_Bs2DKPi_DK_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bs2DKpi'], varDict['low_b_Bs2DKpi'], varDict['low_csi_Bs2DKpi'], varDict['low_global_shift'], varDict['low_sigma_Bs2DKpi'], varDict['low_ratio_Bs2DKpi'], varDict['low_f_Bs2DKpi'])
    pdf_low_Bs2DKPi_DK_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bs2DKpi'], varDict['low_b_Bs2DKpi'], varDict['low_csi_Bs2DKpi'], varDict['low_global_shift'], varDict['low_sigma_Bs2DKpi'], varDict['low_ratio_Bs2DKpi'], varDict['low_f_Bs2DKpi'])

    if mode == 'b2dk_LL':
        pdfList['low_Bs2DKPi'] = lambda Bu_M: norm_pdf(Bu_M,pdf_low_Bs2DKPi_DK_KsPiPi_LL)
    elif mode == 'b2dk_DD':
        pdfList['low_Bs2DKPi'] = lambda Bu_M: norm_pdf(Bu_M, pdf_low_Bs2DKPi_DK_KsPiPi_DD)


    # Combine: with fractions
    if 'frac_low_Bu_Dstar0h_D0pi0_DPi' in varDict.keys():
        # Combine: DPi lowmass
        pdf_low_dpi_DPi_KsPiPi_LL = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DPi']*norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_LL) + varDict['frac_low_Bd_Dstarph_D0pi_DPi']* norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_LL) + varDict['frac_low_Bu_Dstar0h_D0gamma_DPi'] * norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_LL) + varDict['frac_low_B2Dpipi_DPi'] * norm_pdf(Bu_M, pdf_B2Dpipi_DPi_KsPiPi_LL)
        pdf_low_dpi_DPi_KsPiPi_DD = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DPi']*norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_DD) + varDict['frac_low_Bd_Dstarph_D0pi_DPi']* norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_DD) + varDict['frac_low_Bu_Dstar0h_D0gamma_DPi'] * norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_DD) + varDict['frac_low_B2Dpipi_DPi'] * norm_pdf(Bu_M, pdf_B2Dpipi_DPi_KsPiPi_DD)
        # Combine: DK lowmass
        pdf_low_dk_DK_KsPiPi_LL = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DK']*norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL) + varDict['frac_low_Bd_Dstarph_D0pi_DK']* norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DK_KsPiPi_LL) + varDict['frac_low_Bu_Dstar0h_D0gamma_DK']* norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL) + varDict['frac_low_B2DKpi_DK']* norm_pdf(Bu_M, pdf_B2DKpi_DK_KsPiPi_LL)

        pdf_low_dk_DK_KsPiPi_DD = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DK']*norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD) + varDict['frac_low_Bd_Dstarph_D0pi_DK']* norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DK_KsPiPi_DD) + varDict['frac_low_Bu_Dstar0h_D0gamma_DK'] * norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD) + varDict['frac_low_B2DKpi_DK']* norm_pdf(Bu_M, pdf_B2DKpi_DK_KsPiPi_DD)
        # Combine: DK misid
        pdf_low_misID_DK_KsPiPi_LL = lambda Bu_M: varDict['frac_low_misID_Bu_Dstar0h_D0pi0_DK']* norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL) + varDict['frac_low_misID_Bd_Dstarph_D0pi_DK']* norm_pdf(Bu_M, pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_LL) + varDict['frac_low_misID_Bu_Dstar0h_D0gamma_DK']* norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL) + varDict['frac_low_misID_B2Dpipi_DK']* norm_pdf(Bu_M, pdf_low_misID_B2Dpipi_DK_KsPiPi_LL)
        pdf_low_misID_DK_KsPiPi_DD = lambda Bu_M: varDict['frac_low_misID_Bu_Dstar0h_D0pi0_DK']* norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD) + varDict['frac_low_misID_Bd_Dstarph_D0pi_DK']* norm_pdf(Bu_M, pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_DD) + varDict['frac_low_misID_Bu_Dstar0h_D0gamma_DK']* norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD) + varDict['frac_low_misID_B2Dpipi_DK']* norm_pdf(Bu_M, pdf_low_misID_B2Dpipi_DK_KsPiPi_DD)
    else:
        # Combine: DPi lowmass
        pdf_low_dst2dpi_DPi_KsPiPi_LL = lambda Bu_M: norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_LL) * varDict['low_dpi_ratio_Bd_dst_vs_Bu_dst'] + (1 - varDict['low_dpi_ratio_Bd_dst_vs_Bu_dst']) *norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_LL)
        pdf_low_dst2dpi_DPi_KsPiPi_DD = lambda Bu_M: norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_DD) * varDict['low_dpi_ratio_Bd_dst_vs_Bu_dst'] + (1 - varDict['low_dpi_ratio_Bd_dst_vs_Bu_dst']) *norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_DD)
        pdf_low_dpi_dst_DPi_KsPiPi_LL = lambda Bu_M: norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_LL) * varDict['low_dpi_ratio_dst2dgam_vs_dst2dpi'] + (1 - varDict['low_dpi_ratio_dst2dgam_vs_dst2dpi']) * norm_pdf(Bu_M, pdf_low_dst2dpi_DPi_KsPiPi_LL)
        pdf_low_dpi_dst_DPi_KsPiPi_DD = lambda Bu_M: norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_DD) * varDict['low_dpi_ratio_dst2dgam_vs_dst2dpi'] + (1 - varDict['low_dpi_ratio_dst2dgam_vs_dst2dpi']) * norm_pdf(Bu_M, pdf_low_dst2dpi_DPi_KsPiPi_DD)
        pdf_low_dpi_DPi_KsPiPi_LL     = lambda Bu_M: norm_pdf(Bu_M, pdf_B2Dpipi_DPi_KsPiPi_LL) * varDict['low_dpi_ratio_b2drho_vs_b2dstpi'] + (1 - varDict['low_dpi_ratio_b2drho_vs_b2dstpi']) * norm_pdf(Bu_M, pdf_low_dpi_dst_DPi_KsPiPi_LL)
        pdf_low_dpi_DPi_KsPiPi_DD     = lambda Bu_M: norm_pdf(Bu_M, pdf_B2Dpipi_DPi_KsPiPi_DD) * varDict['low_dpi_ratio_b2drho_vs_b2dstpi'] + (1 - varDict['low_dpi_ratio_b2drho_vs_b2dstpi']) * norm_pdf(Bu_M, pdf_low_dpi_dst_DPi_KsPiPi_DD)
        # Combine: DK lowmass
        pdf_low_dst2dpi_DK_KsPiPi_LL = lambda Bu_M: norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DK_KsPiPi_LL) * varDict['low_dk_ratio_Bd_dst_vs_Bu_dst'] + (1 - varDict['low_dk_ratio_Bd_dst_vs_Bu_dst']) * norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL) 
        pdf_low_dst2dpi_DK_KsPiPi_DD = lambda Bu_M: norm_pdf(Bu_M, pdf_Bd_Dstarph_D0pi_DK_KsPiPi_DD) * varDict['low_dk_ratio_Bd_dst_vs_Bu_dst'] + (1 - varDict['low_dk_ratio_Bd_dst_vs_Bu_dst']) * norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD)
        pdf_low_dk_dst_DK_KsPiPi_LL  = lambda Bu_M: norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL) * varDict['low_dk_ratio_dst2dgam_vs_dst2dk'] + (1 - varDict['low_dk_ratio_dst2dgam_vs_dst2dk']) * norm_pdf(Bu_M, pdf_low_dst2dpi_DK_KsPiPi_LL)
        pdf_low_dk_dst_DK_KsPiPi_DD  = lambda Bu_M: norm_pdf(Bu_M, pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD) * varDict['low_dk_ratio_dst2dgam_vs_dst2dk'] + (1 - varDict['low_dk_ratio_dst2dgam_vs_dst2dk']) * norm_pdf(Bu_M, pdf_low_dst2dpi_DK_KsPiPi_DD)
        pdf_low_dk_DK_KsPiPi_LL      = lambda Bu_M: norm_pdf(Bu_M, pdf_B2DKpi_DK_KsPiPi_LL) * varDict['low_dk_ratio_b2dkst_vs_b2dstk'] + (1- varDict['low_dk_ratio_b2dkst_vs_b2dstk']) *  norm_pdf(Bu_M, pdf_low_dk_dst_DK_KsPiPi_LL)
        pdf_low_dk_DK_KsPiPi_DD      = lambda Bu_M: norm_pdf(Bu_M, pdf_B2DKpi_DK_KsPiPi_DD) * varDict['low_dk_ratio_b2dkst_vs_b2dstk'] + (1- varDict['low_dk_ratio_b2dkst_vs_b2dstk']) *  norm_pdf(Bu_M, pdf_low_dk_dst_DK_KsPiPi_DD)
        # Combine: DK misid
        pdf_low_misID_dst2dpi_DK_KsPiPi_LL = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_LL) * varDict['low_dpi_to_dk_misID_ratio_Bd_dst_vs_Bu_dst'] + (1 - varDict['low_dpi_to_dk_misID_ratio_Bd_dst_vs_Bu_dst']) * norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL)
        pdf_low_misID_dst2dpi_DK_KsPiPi_DD = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_DD) * varDict['low_dpi_to_dk_misID_ratio_Bd_dst_vs_Bu_dst'] + (1 - varDict['low_dpi_to_dk_misID_ratio_Bd_dst_vs_Bu_dst']) * norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD)
        pdf_low_misID_dst_DK_KsPiPi_LL     = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL) * varDict['low_dpi_to_dk_misID_ratio_dst2dgam_vs_dst2dpi'] + (1 - varDict['low_dpi_to_dk_misID_ratio_dst2dgam_vs_dst2dpi']) * norm_pdf(Bu_M, pdf_low_misID_dst2dpi_DK_KsPiPi_LL)
        pdf_low_misID_dst_DK_KsPiPi_DD     = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD) * varDict['low_dpi_to_dk_misID_ratio_dst2dgam_vs_dst2dpi'] + (1 - varDict['low_dpi_to_dk_misID_ratio_dst2dgam_vs_dst2dpi']) * norm_pdf(Bu_M, pdf_low_misID_dst2dpi_DK_KsPiPi_DD)
        pdf_low_misID_DK_KsPiPi_LL         = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_B2Dpipi_DK_KsPiPi_LL) * varDict['low_dpi_to_dk_misID_ratio_b2drho_vs_b2dstpi'] + (1 - varDict['low_dpi_to_dk_misID_ratio_b2drho_vs_b2dstpi']) * norm_pdf(Bu_M, pdf_low_misID_dst_DK_KsPiPi_LL)
        pdf_low_misID_DK_KsPiPi_DD         = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_B2Dpipi_DK_KsPiPi_DD) * varDict['low_dpi_to_dk_misID_ratio_b2drho_vs_b2dstpi'] + (1 - varDict['low_dpi_to_dk_misID_ratio_b2drho_vs_b2dstpi']) * norm_pdf(Bu_M, pdf_low_misID_dst_DK_KsPiPi_DD)

    if mode == 'b2dk_LL':
        pdfList['low'] = lambda Bu_M: norm_pdf(Bu_M, pdf_low_dk_DK_KsPiPi_LL)
        pdfList['low_misID'] = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_DK_KsPiPi_LL)

    elif mode == 'b2dk_DD':
        pdfList['low'] = lambda Bu_M: norm_pdf(Bu_M, pdf_low_dk_DK_KsPiPi_DD)
        pdfList['low_misID'] = lambda Bu_M: norm_pdf(Bu_M, pdf_low_misID_DK_KsPiPi_DD)

    elif mode == 'b2dpi_LL':
        pdfList['low'] = lambda Bu_M: norm_pdf(Bu_M, pdf_low_dpi_DPi_KsPiPi_LL)

    elif mode == 'b2dpi_DD':
        pdfList['low'] = lambda Bu_M: norm_pdf(Bu_M, pdf_low_dpi_DPi_KsPiPi_DD)


    # combinatorial
    print('--- Constructing comb pdfs...')
    pdf_comb_DK_KsPiPi_LL = lambda Bu_M: Exponential(Bu_M, varDict['comb_const_dk_d2kspp_LL'])
    pdf_comb_DK_KsPiPi_DD = lambda Bu_M: Exponential(Bu_M, varDict['comb_const_dk_d2kspp_DD'])
    pdf_comb_DPi_KsPiPi_LL = lambda Bu_M: Exponential(Bu_M, varDict['comb_const_dpi_d2kspp_LL'])
    pdf_comb_DPi_KsPiPi_DD = lambda Bu_M: Exponential(Bu_M, varDict['comb_const_dpi_d2kspp_DD'])
    if mode == 'b2dk_LL':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Bu_M, pdf_comb_DK_KsPiPi_LL)
    elif mode == 'b2dk_DD':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Bu_M, pdf_comb_DK_KsPiPi_DD)
    elif mode == 'b2dpi_LL':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Bu_M, pdf_comb_DPi_KsPiPi_LL)
    elif mode == 'b2dpi_DD':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Bu_M, pdf_comb_DPi_KsPiPi_DD)


    print('--- INFO: Writing to file...')
    return pdfList




class MassPDF():
    def __init__(self, name_function, component, Bsign=None):
        self.name      = name_function
        self.component = component
        self.Bsign     = Bsign
        self.function  = self.get_function()

    def get_function(self):
        ## could be prettier if we change the functions to deal
        # with different variables "internally" so that they only take
        # Bu_M and variables as arguments
        if (self.name == "Exponential"):
            function = Exponential
        elif (self.name == "HORNSdini"):
            function = HORNSdini
        elif (self.name == "HORNSdini+Gaussian"):
            function = HORNSdini_Gaussian
        elif (self.name == "HORNSdini+HORNSdini"):
            function = HORNSdini_HORNSdini
        elif (self.name == "HORNSdini_misID"):
            function = HORNSdini_misID
        elif (self.name == "HILLdini"):
            function = HILLdini
        elif (self.name == "HILLdini_misID"):
            function = HILLdini_misID
        elif (self.name == "CBShape"):
            function = CBShape
        elif (self.name == "SumCBShape"):
            function = SumCBShape
        elif (self.name == "CruijffExtended"):
            function = CruijffExtended
        elif (self.name == "Gaussian"):
            function = Gaussian
        elif (self.name == "Cruijff+Gaussian"):
            function = Cruijff_Gaussian            ## Cruijff
        else:
            print("ERROR ---------- in MassPDF constructor")
            print("       PDF with self.name ",self.name," does not exist")
            print("            ---------------------  EXIT ")
            print("  ")
            pass
        return function

    # @tf.function
    def get_mass_pdf(self,variables, Bsign=None):
        # print(" PRINTING some stuff !")
        # print(self.name)
        # print(self.component)
        # print(variables)
        # if (Bsign==None):
        #     # print("Bsign not specified ------ summing B+ and B- samples")
        #     ######### explaining what's happening here:
        #     # we don't sum the two yields because they are just a single
        #     # parameter in the fit. That means that the free parameter is
        #     # called "{component}_yield_Bplus", but the actual result
        #     # will be the total yield
        #     comp_yield = variables[INDEX_YIELDS["Bplus"]] + variables[INDEX_YIELDS["Bminus"]]
        # else:
        #     # print(Bsign)
        #     # print(INDEX_YIELDS[Bsign])
        #     comp_yield = variables[INDEX_YIELDS[Bsign]]
        #     pass
        pdf = lambda Bu_M: self.function(Bu_M, variables)
        # print(" comp_yield: ",comp_yield)
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        # print(" between pdf and norm_pdf !")
        res = lambda Bu_M: norm_pdf(Bu_M,pdf) # comp_yield*
        # print(" after norm_pdf !")
        # print(" after norm_pdf !")
        # print(" after norm_pdf !")
        # print(" after norm_pdf !")
        self.pdf = res
        return res    

    def get_norm_pdf(self,variables):
        self.pdf = self.get_pdf(variables)
        res = lambda Bu_M: norm_pdf(Bu_M, self.pdf)
        self.norm_pdf = res
        return res
        
