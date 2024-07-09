import numpy as np
import tensorflow as tf

#def DeltadeltaD(A, Abar):
#    var = tf.math.angle(A*tf.math.conj(Abar))
#    return tf.cast(var, tf.float64)
_PI = tf.constant(np.pi, dtype=tf.float64)
def DeltadeltaD(A, Abar):
    var_a = tf.math.angle(A*np.conj(Abar))+_PI
    var_b = tf.where(var_a > _PI, var_a - 2*_PI, var_a)
    var = tf.where(var_b < -_PI, var_b + 2*_PI, var_b)

    return var

def totalAmplitudeSquared_Integrated_crossTerm(A, Abar):
    '''
    This function calculates the total amplitude squared for the integrated decay, v0.1 only for MD fitted, no correction yet
    |A||Abar|cos(deltaD)
    |A||Abar|sin(deltaD)
    '''
    phase = DeltadeltaD(A, Abar)
    AAbar = tf.cast(tf.abs(A)*tf.abs(Abar), tf.float64)
    real_part = tf.cast(tf.math.reduce_mean(AAbar*tf.cos(phase)), tf.float64)
    imag_part = tf.cast(tf.math.reduce_mean(AAbar*tf.sin(phase)), tf.float64)

    return (real_part, imag_part)


def totalAmplitudeSquared_Integrated_new(Bsign=1, A=[], Abar=[], x=(0,0,0,0)):

    '''
    A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)

    A^2 + Abar^2 * rb^2 + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)
    '''
    absA = tf.cast(tf.abs(A), tf.float64)
    absAbar = tf.cast(tf.abs(Abar), tf.float64)
    phase = DeltadeltaD(A, Abar)
    
    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)

        Gp = tf.cast(absA**2 * rB2 + absAbar**2 + 2.0 * (absA * absAbar) *(xPlus * tf.cos(phase) - yPlus * tf.sin(phase)), tf.float64)

        return tf.math.reduce_mean(Gp)
    
    else:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)
        Gm = tf.cast(absA**2 + absAbar**2  * rB2 + 2.0 * (absA * absAbar) *(xMinus * tf.cos(phase) + yMinus * tf.sin(phase)), tf.float64)

        return tf.math.reduce_mean(Gm)
    


def totalAmplitudeSquared_Integrated(Bsign=1, normA=1.1, normAbar=1.1, crossTerm=(0, 0), x=(0,0,0,0)):
    '''
    A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)

    A^2 + Abar^2 * rb^2 + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)
    '''
    normA = tf.cast(normA, tf.float64)
    normAbar = tf.cast(normAbar, tf.float64)
    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)

        return tf.cast(normA * rB2 + normAbar + 2.0 *(xPlus *crossTerm[0] - yPlus * crossTerm[1]), tf.float64)
    
    else:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)

        return tf.cast(normA + normAbar  * rB2 + 2.0 *(xMinus *crossTerm[0] + yMinus * crossTerm[1]), tf.float64)
    


    
def totalAmplitudeSquared_DPi_Integrated(Bsign=1, normA=1.1, normAbar=1.1, crossTerm=(0, 0), x=(0,0,0,0)):
    '''
    A^2 * rb^2 + Abar^2 + 2*|A||Abar| * rb * cos(deltaB + gamma + deltaD)

    A^2 + Abar^2 * rb^2 + 2*|A||Abar| * rb * cos(deltaB + gamma - deltaD)
    '''
    xXi = tf.cast(x[4], tf.float64)
    yXi = tf.cast(x[5], tf.float64)
    normA = tf.cast(normA, tf.float64)
    normAbar = tf.cast(normAbar, tf.float64)
    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        xPlus_DPi = tf.cast(xPlus * xXi - yPlus * yXi, tf.float64)
        yPlus_DPi = tf.cast(yPlus * xXi + xPlus * yXi, tf.float64)

        rB2 = tf.cast(xPlus_DPi**2 + yPlus_DPi**2, tf.float64)
        return tf.cast(normA * rB2 + normAbar + 2.0 *(xPlus_DPi *crossTerm[0] - yPlus_DPi * crossTerm[1]), tf.float64)
    
    else:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        xMinus_DPi = tf.cast(xMinus * xXi - yMinus * yXi, tf.float64)
        yMinus_DPi = tf.cast(yMinus * xXi + xMinus * yXi, tf.float64)

        rB2 = tf.cast(xMinus_DPi**2 + yMinus_DPi**2, tf.float64)

        return tf.cast(normA + normAbar  * rB2 + 2.0 *(xMinus_DPi *crossTerm[0] + yMinus_DPi * crossTerm[1]), tf.float64)
   
def prod_totalAmplitudeSquared_XY( Bsign=1, amp=[], ampbar=[], x=(0,0,0,0,0,0), pdfs1=[], pdfs2=[], B_M1=[], B_M2=[]):

    phase = DeltadeltaD(amp, ampbar)
    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)


    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        rB2 = tf.cast(xPlus**2 + yPlus**2, tf.float64)
        

        return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus * tf.cos(phase) - yPlus * tf.sin(phase)))
    
    elif Bsign == -1:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        rB2 = tf.cast(xMinus**2 + yMinus**2, tf.float64)

        return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus * tf.cos(phase) + yMinus * tf.sin(phase)))

def prod_totalAmplitudeSquared_DPi_XY( Bsign=1, amp=[], ampbar=[], x=(0,0,0,0,0,0,0,0,0,0,0,0,0,0), pdfs1=[], pdfs2=[], B_M1=[], B_M2=[]):

    phase = DeltadeltaD(amp, ampbar)
    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)

    xXi = tf.cast(x[4], tf.float64)
    yXi = tf.cast(x[5], tf.float64)

    if Bsign == 1:
        xPlus = tf.cast(x[0], tf.float64)
        yPlus = tf.cast(x[1], tf.float64)
        xPlus_DPi = tf.cast(xPlus * xXi - yPlus * yXi, tf.float64)
        yPlus_DPi = tf.cast(yPlus * xXi + xPlus * yXi, tf.float64)
        rB2 = tf.cast(xPlus_DPi**2 + yPlus_DPi**2, tf.float64)

        return (absA**2 * rB2  + absAbar **2  + 2.0 * (absA * absAbar) * (xPlus_DPi * tf.cos(phase) - yPlus_DPi * tf.sin(phase)))
    
    elif Bsign == -1:
        xMinus = tf.cast(x[2], tf.float64)
        yMinus = tf.cast(x[3], tf.float64)
        xMinus_DPi = tf.cast(xMinus * xXi - yMinus * yXi, tf.float64)
        yMinus_DPi = tf.cast(yMinus * xXi + xMinus * yXi, tf.float64)
        rB2 = tf.cast(xMinus_DPi**2 + yMinus_DPi**2, tf.float64)


        return (absA**2  + absAbar **2 * rB2 + 2.0 * (absA * absAbar) * (xMinus_DPi * tf.cos(phase) + yMinus_DPi * tf.sin(phase)))


def prod_comb(amp=[], ampbar=[], normA=1.2, normAbar=1.2, fracDD=0.82):

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)
    frac1 = fracDD/2.0
    frac2 = 1.0 - fracDD
    prob1 = absA**2/normA
    prob2 = absAbar**2/normAbar 
    prob3 = tf.ones_like(absA)

    #return ((absA**2/normA + absAbar**2/normAbar)*0.5 * fracDD + 1*(1.0-fracDD))
    return prob1*frac1 + prob2*frac1 + prob3/tf.reduce_sum(prob3) * frac2

def norm_comb(amp=[], ampbar=[], fracDD=0.82):

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)

    return tf.math.reduce_mean((absA**2 + absAbar**2)*0.5 * fracDD + (1.0-fracDD))


def prod_low(Bsign=1, amp=[], ampbar=[], x=(0,0,0,0,0), pdfs=[], B_M=[], type='low'):

    absA = tf.cast(tf.abs(amp), tf.float64)
    absAbar = tf.cast(tf.abs(ampbar), tf.float64)

    if Bsign ==1:
        return absAbar**2
    elif Bsign == -1:
        return absA**2


