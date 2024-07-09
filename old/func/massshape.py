import numpy as np
from scipy.special import erf

def norm_pdf(pdf):
    norm_term = np.mean(pdf)
    return np.array(pdf/norm_term)

def addPdf(pdflist=[], frac=[]):
    if isinstance(pdflist, np.ndarray):
        pdflist = [pdflist]  # Wrap single numpy array in a list
    pdflist = [np.array(pdf, dtype=float) for pdf in pdflist]  # Ensure all elements are numpy arrays

    if not isinstance(frac, list):
        frac = [frac]  # Convert single fraction to list

    # Initialize the PDF with zeros of the same shape as the first PDF in the list
    pdf = np.zeros_like(pdflist[0], dtype=float)

    # Normalize PDFs in pdflist if their sum is not exactly 1
    for i in range(len(pdflist)):
        if np.sum(pdflist[i]) != 1:
            pdflist[i] = norm_pdf(pdflist[i])


    if len(pdflist) != len(frac):
        if len(frac) ==1:
            pdf = frac[0]*pdflist[0] + (1-frac[0])*pdflist[1]

            return pdf
        else:
            return None and print('The length of the list of pdfs and the list of fractions must be the same')
    
    else:
        for i in range(len(pdflist)):
                pdf += frac[i]*pdflist[i]
        return pdf

def HORNSdini(m, a, b, csi, shift, sigma, ratio_sigma, fraction_sigma):

    a_new = a
    b_new = b
    B_NEW = (a_new + b_new) / 2.0
    sigma2 = sigma * ratio_sigma

    firstG1 = ((2*(a_new-2*B_NEW+(m-shift))*sigma)/np.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*sigma*sigma)) - (2*(b_new-2*B_NEW+(m-shift))*sigma)/np.exp((b_new-(m-shift))*(b_new-(m-shift))/(2*sigma*sigma))+ np.sqrt(2*np.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma)*erf((-a_new+(m-shift))/(np.sqrt(2)*sigma))  - np.sqrt(2*np.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma) * erf((-b_new+(m-shift))/(np.sqrt(2)*sigma)))/(2*np.sqrt(2*np.pi))
    secondG1 = (((2*sigma*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(a_new+(m-shift)) + 2*(sigma*sigma)))/np.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma*sigma))) - (2*sigma*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(b_new + (m-shift)) + 2*(sigma*sigma)))/np.exp((b_new - (m-shift))*(b_new - (m-shift))/(2*(sigma*sigma))) - np.sqrt(2*np.pi)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma*sigma))*erf((-a_new + (m-shift))/(np.sqrt(2)*sigma)) + np.sqrt(2*np.pi)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma*sigma)) *erf((-b_new + (m-shift))/(np.sqrt(2)*sigma)))/(2 *np.sqrt(2*np.pi)))

    CURVEG1 = np.abs((1-csi)*secondG1 + (b_new*csi - a_new)*firstG1)

    firstG2 = ((2*(a_new-2*B_NEW+(m-shift))*sigma2)/np.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*sigma2*sigma2)) - (2*(b_new-2*B_NEW+(m-shift))*sigma2)/np.exp((b_new-(m-shift))*(b_new-(m-shift))/(2*sigma2*sigma2))+ np.sqrt(2*np.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2*sigma2)*erf((-a_new+(m-shift))/(np.sqrt(2)*sigma2))  - np.sqrt(2*np.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2*sigma2) * erf((-b_new+(m-shift))/(np.sqrt(2)*sigma2)))/(2*np.sqrt(2*np.pi))
    secondG2 = (((2*sigma2*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(a_new+(m-shift)) + 2*(sigma2*sigma2)))/np.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma2*sigma2))) - (2*sigma2*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(b_new + (m-shift)) + 2*(sigma2*sigma2)))/np.exp((b_new - (m-shift))*(b_new - (m-shift))/(2*(sigma2*sigma2))) - np.sqrt(2*np.pi)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma2*sigma2))*erf((-a_new + (m-shift))/(np.sqrt(2)*sigma2)) + np.sqrt(2*np.pi)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma2*sigma2)) *erf((-b_new + (m-shift))/(np.sqrt(2)*sigma2)))/(2 *np.sqrt(2*np.pi)))

    CURVEG2 = np.abs((1-csi)*secondG2 + (b_new*csi - a_new)*firstG2)

    return (fraction_sigma*CURVEG1 + (1-fraction_sigma)*CURVEG2)


def CruijffExtended(m, m0, sigmaL, sigmaR, alphaL, alphaR, beta):
    sigma = 0.0
    alpha = 0.0
    dx = m - m0

    sigma = np.where(dx < 0, sigmaL, sigmaR)
    alpha = np.where(dx < 0, alphaL, alphaR)

    f = 2.0*sigma*sigma + alpha*dx*dx
    return np.exp(-dx**2 *(1 + beta * dx **2)/f)  


def CBShape(m,m0,sigma,alpha,n):
   t =(m-m0)/sigma

   if alpha <0:
      t = -t
   else:
      t = t

   absAlpha = np.abs(alpha) 
   val_a = np.exp(-0.5*t*t)
   
   a =  np.power(n/absAlpha,n)*np.exp(-0.5*absAlpha*absAlpha)
   b= n/absAlpha - absAlpha
   val_b = a/np.power(b-t, n)

   val = np.where(t >= -absAlpha, val_a, val_b)
   return val

def HORNSdini_misID(m,a,b,csi,m1,s1,m2,s2,m3,s3,m4,s4,f1,f2,f3):
    a_new = a
    b_new = b
    B_NEW = (a_new + b_new) / 2.0


    firstG1 = ((2 * (a_new - 2 * B_NEW + (m - m1)) * s1) / np.exp((a_new - (m - m1)) **2 / (2 * s1**2 )) - (2 * (b_new - 2 * B_NEW + (m - m1)) * s1) / np.exp((b_new - (m - m1)) **2 / (2 * (s1 **2))) + np.sqrt(2 * np.pi) * ((B_NEW - (m - m1)) **2 + s1 ** 2) * erf((-a_new + (m - m1)) / (np.sqrt(2) * s1)) - np.sqrt(2 * np.pi) * ((B_NEW - (m - m1)) ** 2 + s1 **2) * erf((-b_new + (m - m1)) / (np.sqrt(2) * s1))) / (2 * np.sqrt(2 * np.pi))
    secondG1 = (((2*s1*(a_new**2 + B_NEW**2 + a_new*(m-m1) + (m-m1)**2 - 2*B_NEW*(a_new+(m-m1)) + 2*s1**2 ))/np.exp((a_new-(m-m1))**2 /(2*(s1**2))) - (2*s1*(b_new**2 + B_NEW**2 + b_new*(m-m1) + (m-m1)**2 - 2*B_NEW*(b_new + (m-m1)) + 2*(s1**2)))/np.exp((b_new - (m-m1))**2 /(2*(s1**2))) - np.sqrt(2*np.pi)*(-((B_NEW - (m-m1))**2 *(m-m1)) + (2*B_NEW - 3*(m-m1))*(s1**2))*erf((-a_new + (m-m1))/(np.sqrt(2)*s1)) + np.sqrt(2*np.pi)* (-((B_NEW - (m-m1))**2 *(m-m1)) + (2*B_NEW - 3*(m-m1))*(s1**2)) *erf((-b_new + (m-m1))/(np.sqrt(2)*s1)))/(2 *np.sqrt(2*np.pi)))
    CURVEG1 = np.fabs((1-csi)*secondG1 + (b_new*csi - a_new)*firstG1)

    firstG2 = ((2 * (a_new - 2 * B_NEW + (m - m2)) * s2) / np.exp((a_new - (m - m2)) **2 / (2 * s2**2 )) - (2 * (b_new - 2 * B_NEW + (m - m2)) * s2) / np.exp((b_new - (m - m2)) **2 / (2 * (s2 **2))) + np.sqrt(2 * np.pi) * ((B_NEW - (m - m2)) **2 + s2 ** 2) * erf((-a_new + (m - m2)) / (np.sqrt(2) * s2)) - np.sqrt(2 * np.pi) * ((B_NEW - (m - m2)) ** 2 + s2 **2) * erf((-b_new + (m - m2)) / (np.sqrt(2) * s2))) / (2 * np.sqrt(2 * np.pi))
    secondG2 = (((2*s2*(a_new**2 + B_NEW**2 + a_new*(m-m2) + (m-m2)**2 - 2*B_NEW*(a_new+(m-m2)) + 2*s2**2 ))/np.exp((a_new-(m-m2))**2 /(2*(s2**2))) - (2*s2*(b_new**2 + B_NEW**2 + b_new*(m-m2) + (m-m2)**2 - 2*B_NEW*(b_new + (m-m2)) + 2*(s2**2)))/np.exp((b_new - (m-m2))**2 /(2*(s2**2))) - np.sqrt(2*np.pi)*(-((B_NEW - (m-m2))**2 *(m-m2)) + (2*B_NEW - 3*(m-m2))*(s2**2))*erf((-a_new + (m-m2))/(np.sqrt(2)*s2)) + np.sqrt(2*np.pi)* (-((B_NEW - (m-m2))**2 *(m-m2)) + (2*B_NEW - 3*(m-m2))*(s2**2)) *erf((-b_new + (m-m2))/(np.sqrt(2)*s2)))/(2 *np.sqrt(2*np.pi)))
    CURVEG2 = np.fabs((1-csi)*secondG2 + (b_new*csi - a_new)*firstG2)

    firstG3 = ((2 * (a_new - 2 * B_NEW + (m - m3)) * s3) / np.exp((a_new - (m - m3)) **2 / (2 * s3**2 )) - (2 * (b_new - 2 * B_NEW + (m - m3)) * s3) / np.exp((b_new - (m - m3)) **2 / (2 * (s3 **2))) + np.sqrt(2 * np.pi) * ((B_NEW - (m - m3)) **2 + s3 ** 2) * erf((-a_new + (m - m3)) / (np.sqrt(2) * s3)) - np.sqrt(2 * np.pi) * ((B_NEW - (m - m3)) ** 2 + s3 **2) * erf((-b_new + (m - m3)) / (np.sqrt(2) * s3))) / (2 * np.sqrt(2 * np.pi))
    secondG3 = (((2*s3*(a_new**2 + B_NEW**2 + a_new*(m-m3) + (m-m3)**2 - 2*B_NEW*(a_new+(m-m3)) + 2*s3**2 ))/np.exp((a_new-(m-m3))**2 /(2*(s3**2))) - (2*s3*(b_new**2 + B_NEW**2 + b_new*(m-m3) + (m-m3)**2 - 2*B_NEW*(b_new + (m-m3)) + 2*(s3**2)))/np.exp((b_new - (m-m3))**2 /(2*(s3**2))) - np.sqrt(2*np.pi)*(-((B_NEW - (m-m3))**2 *(m-m3)) + (2*B_NEW - 3*(m-m3))*(s3**2))*erf((-a_new + (m-m3))/(np.sqrt(2)*s3)) + np.sqrt(2*np.pi)* (-((B_NEW - (m-m3))**2 *(m-m3)) + (2*B_NEW - 3*(m-m3))*(s3**2)) *erf((-b_new + (m-m3))/(np.sqrt(2)*s3)))/(2 *np.sqrt(2*np.pi)))
    CURVEG3 = np.fabs((1-csi)*secondG3 + (b_new*csi - a_new)*firstG3)

    firstG4 = ((2 * (a_new - 2 * B_NEW + (m - m4)) * s4) / np.exp((a_new - (m - m4)) **2 / (2 * s4**2 )) - (2 * (b_new - 2 * B_NEW + (m - m4)) * s4) / np.exp((b_new - (m - m4)) **2 / (2 * (s4 **2))) + np.sqrt(2 * np.pi) * ((B_NEW - (m - m4)) **2 + s4 ** 2) * erf((-a_new + (m - m4)) / (np.sqrt(2) * s4)) - np.sqrt(2 * np.pi) * ((B_NEW - (m - m4)) ** 2 + s4 **2) * erf((-b_new + (m - m4)) / (np.sqrt(2) * s4))) / (2 * np.sqrt(2 * np.pi))
    secondG4 = (((2*s4*(a_new**2 + B_NEW**2 + a_new*(m-m4) + (m-m4)**2 - 2*B_NEW*(a_new+(m-m4)) + 2*s4**2 ))/np.exp((a_new-(m-m4))**2 /(2*(s4**2))) - (2*s4*(b_new**2 + B_NEW**2 + b_new*(m-m4) + (m-m4)**2 - 2*B_NEW*(b_new + (m-m4)) + 2*(s4**2)))/np.exp((b_new - (m-m4))**2 /(2*(s4**2))) - np.sqrt(2*np.pi)*(-((B_NEW - (m-m4))**2 *(m-m4)) + (2*B_NEW - 3*(m-m4))*(s4**2))*erf((-a_new + (m-m4))/(np.sqrt(2)*s4)) + np.sqrt(2*np.pi)* (-((B_NEW - (m-m4))**2 *(m-m4)) + (2*B_NEW - 3*(m-m4))*(s4**2)) *erf((-b_new + (m-m4))/(np.sqrt(2)*s4)))/(2 *np.sqrt(2*np.pi)))
    CURVEG4 = np.fabs((1-csi)*secondG4 + (b_new*csi - a_new)*firstG4)


    return np.fabs(f1*CURVEG1) + np.fabs(f2*CURVEG2) + np.fabs(f3*CURVEG3) + np.fabs((1-f1-f2-f3)*CURVEG4)

    
def HILLdini(m,a,b,csi,shift,sigma,ratio_sigma,fraction_sigma):
    a_new = a
    b_new = b
    sigma2 = sigma * ratio_sigma

    firstG1 = (2*np.exp(-((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma*sigma))))*sigma*(b_new-(m-shift))+2*np.exp(-((b_new-(m-shift))*(b_new-(m-shift))/(2*(sigma*sigma))))*sigma*(-a_new+(m-shift))-np.sqrt(2*np.pi)*(a_new*b_new+(sigma*sigma)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-a_new+(m-shift))/(np.sqrt(2)*sigma))+np.sqrt(2*np.pi)*(a_new*b_new+(sigma*sigma)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-b_new+(m-shift))/(np.sqrt(2)*sigma)))/(2*np.sqrt(2*np.pi))
    CURVEG1 = np.fabs((1-csi)/(b_new - a_new)*m + (b_new*csi - a_new)/(b_new-a_new))*np.fabs(firstG1)

    firstG2 = (2*np.exp(-((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma2*sigma2))))*sigma2*(b_new-(m-shift))+2*np.exp(-((b_new-(m-shift))*(b_new-(m-shift))/(2*(sigma2*sigma2))))*sigma2*(-a_new+(m-shift))-np.sqrt(2*np.pi)*(a_new*b_new+(sigma2*sigma2)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-a_new+(m-shift))/(np.sqrt(2)*sigma2))+np.sqrt(2*np.pi)*(a_new*b_new+(sigma2*sigma2)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-b_new+(m-shift))/(np.sqrt(2)*sigma2)))/(2*np.sqrt(2*np.pi))
    CURVEG2 = np.fabs((1-csi)/(b_new - a_new)*m + (b_new*csi - a_new)/(b_new-a_new))*np.fabs(firstG2)

    return np.fabs(fraction_sigma*CURVEG1) + np.fabs((1-fraction_sigma)*CURVEG2)

def HILLdini_misID(m,a,b,csi,m1,s1,m2,s2,m3,s3,m4,s4,f1,f2,f3):
    a_new = a
    b_new = b
    B_NEW = (a_new + b_new) / 2.0

    firstG1 = (2*np.exp(-((a_new-(m-m1))*(a_new-(m-m1))/(2*(s1*s1))))*s1*(b_new-(m-m1))+2*np.exp(-((b_new-(m-m1))*(b_new-(m-m1))/(2*(s1*s1))))*s1*(-a_new+(m-m1))-np.sqrt(2*np.pi)*(a_new*b_new+(s1*s1)-(a_new+b_new)*(m-m1)+((m-m1)*(m-m1)))*erf((-a_new+(m-m1))/(np.sqrt(2)*s1))+np.sqrt(2*np.pi)*(a_new*b_new+(s1*s1)-(a_new+b_new)*(m-m1)+((m-m1)*(m-m1)))*erf((-b_new+(m-m1))/(np.sqrt(2)*s1)))/(2*np.sqrt(2*np.pi))
    CURVEG1 = np.fabs((1-csi)/(b_new-a_new)*(m-m1)  + (b_new*csi - a_new)/(b_new-a_new)  )*np.fabs(firstG1)

    firstG2 = (2*np.exp(-((a_new-(m-m2))*(a_new-(m-m2))/(2*(s2*s2))))*s2*(b_new-(m-m2))+2*np.exp(-((b_new-(m-m2))*(b_new-(m-m2))/(2*(s2*s2))))*s2*(-a_new+(m-m2))-np.sqrt(2*np.pi)*(a_new*b_new+(s2*s2)-(a_new+b_new)*(m-m2)+((m-m2)*(m-m2)))*erf((-a_new+(m-m2))/(np.sqrt(2)*s2))+np.sqrt(2*np.pi)*(a_new*b_new+(s2*s2)-(a_new+b_new)*(m-m2)+((m-m2)*(m-m2)))*erf((-b_new+(m-m2))/(np.sqrt(2)*s2)))/(2*np.sqrt(2*np.pi))
    CURVEG2 = np.fabs((1-csi)/(b_new-a_new)*(m-m2)  + (b_new*csi - a_new)/(b_new-a_new)  )*np.fabs(firstG2)

    firstG3 = (2*np.exp(-((a_new-(m-m3))*(a_new-(m-m3))/(2*(s3*s3))))*s3*(b_new-(m-m3))+2*np.exp(-((b_new-(m-m3))*(b_new-(m-m3))/(2*(s3*s3))))*s3*(-a_new+(m-m3))-np.sqrt(2*np.pi)*(a_new*b_new+(s3*s3)-(a_new+b_new)*(m-m3)+((m-m3)*(m-m3)))*erf((-a_new+(m-m3))/(np.sqrt(2)*s3))+np.sqrt(2*np.pi)*(a_new*b_new+(s3*s3)-(a_new+b_new)*(m-m3)+((m-m3)*(m-m3)))*erf((-b_new+(m-m3))/(np.sqrt(2)*s3)))/(2*np.sqrt(2*np.pi))
    CURVEG3 = np.fabs((1-csi)/(b_new-a_new)*(m-m3)  + (b_new*csi - a_new)/(b_new-a_new)  )*np.fabs(firstG3)

    firstG4 = (2*np.exp(-((a_new-(m-m4))*(a_new-(m-m4))/(2*(s4*s4))))*s4*(b_new-(m-m4))+2*np.exp(-((b_new-(m-m4))*(b_new-(m-m4))/(2*(s4*s4))))*s4*(-a_new+(m-m4))-np.sqrt(2*np.pi)*(a_new*b_new+(s4*s4)-(a_new+b_new)*(m-m4)+((m-m4)*(m-m4)))*erf((-a_new+(m-m4))/(np.sqrt(2)*s4))+np.sqrt(2*np.pi)*(a_new*b_new+(s4*s4)-(a_new+b_new)*(m-m4)+((m-m4)*(m-m4)))*erf((-b_new+(m-m4))/(np.sqrt(2)*s4)))/(2*np.sqrt(2*np.pi))
    CURVEG4 = np.fabs((1-csi)/(b_new-a_new)*(m-m4)  + (b_new*csi - a_new)/(b_new-a_new)  )*np.fabs(firstG4)

    return np.fabs(f1*CURVEG1) + np.fabs(f2*CURVEG2) + np.fabs(f3*CURVEG3) + np.fabs((1-f1-f2-f3)*CURVEG4)

def Gaussian(m, mu, sigma):
    return (np.exp(-0.5*((m-mu)/sigma)**2))/(sigma*np.sqrt(2*np.pi))

def Exponential(m, c):
    return np.exp(c*m)



def preparePdf_data(Bu_M, varDict, mode='b2dk_LL'):
    '''
    Import constructed data sets and construct PDFs with RooFit functions.
    PDFs and data sets are saved together in a new RooWorkspace

    Args:
        configDict: a dictionary containing the values of PDF shape parameters
        year: which subset of data to fit, can be any single year of data taking,
              or 'Run1', 'Run2', 'All'.
    ''' 

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
        pdfList['sig'] = lambda Bu_M: (norm_pdf(pdf_sig_Cruijff_DK_KsPiPi_LL(Bu_M)) * varDict['LL_dk_Cruijff_frac'] +  norm_pdf(pdf_sig_Gauss_DK_KsPiPi_LL(Bu_M)) * varDict['LL_dk_Gauss_frac'])
    elif mode == 'b2dpi_LL':
        pdfList['sig'] = lambda Bu_M: (norm_pdf(pdf_sig_Cruijff_DPi_KsPiPi_LL(Bu_M)) * varDict['LL_dpi_Cruijff_frac'] +  norm_pdf(pdf_sig_Gauss_DPi_KsPiPi_LL(Bu_M)) *  varDict['LL_dpi_Gauss_frac'])
    elif mode == 'b2dk_DD':
        pdfList['sig'] = lambda Bu_M: (norm_pdf(pdf_sig_Cruijff_DK_KsPiPi_DD(Bu_M)) * varDict['DD_dk_Cruijff_frac'] +  norm_pdf(pdf_sig_Gauss_DK_KsPiPi_DD(Bu_M)) *  varDict['DD_dk_Gauss_frac'])
    elif mode == 'b2dpi_DD':
        pdfList['sig'] = lambda Bu_M: (norm_pdf(pdf_sig_Cruijff_DPi_KsPiPi_DD(Bu_M)) * varDict['DD_dpi_Cruijff_frac'] + norm_pdf(pdf_sig_Gauss_DPi_KsPiPi_DD(Bu_M)) * varDict['DD_dpi_Gauss_frac'])    
 

    print('--- Constructing misID pdfs...')
    pdf_misid_CB1_DK_KsPiPi_LL = lambda Bu_M: CBShape(Bu_M, varDict['LL_d2kspp_dpi_to_dk_misID_mean1'], varDict['LL_d2kspp_dpi_to_dk_misID_width1'], varDict['LL_d2kspp_dpi_to_dk_misID_alpha1'], varDict['LL_d2kspp_dpi_to_dk_misID_n1'])
    pdf_misid_CB2_DK_KsPiPi_LL = lambda Bu_M: CBShape(Bu_M, varDict['LL_d2kspp_dpi_to_dk_misID_mean1'], varDict['LL_d2kspp_dpi_to_dk_misID_width2'], varDict['LL_d2kspp_dpi_to_dk_misID_alpha2'], varDict['LL_d2kspp_dpi_to_dk_misID_n1'])
    pdf_misid_CB1_DK_KsPiPi_DD = lambda Bu_M: CBShape(Bu_M, varDict['DD_d2kspp_dpi_to_dk_misID_mean1'], varDict['DD_d2kspp_dpi_to_dk_misID_width1'], varDict['DD_d2kspp_dpi_to_dk_misID_alpha1'], varDict['DD_d2kspp_dpi_to_dk_misID_n1'])
    pdf_misid_CB2_DK_KsPiPi_DD = lambda Bu_M: CBShape(Bu_M, varDict['DD_d2kspp_dpi_to_dk_misID_mean1'], varDict['DD_d2kspp_dpi_to_dk_misID_width2'], varDict['DD_d2kspp_dpi_to_dk_misID_alpha2'], varDict['DD_d2kspp_dpi_to_dk_misID_n1'])
    if mode == 'b2dk_LL':
        pdfList['misid'] = lambda Bu_M: (norm_pdf(pdf_misid_CB1_DK_KsPiPi_LL(Bu_M)) * varDict['LL_d2kspp_dpi_to_dk_misID_frac1'] + norm_pdf(pdf_misid_CB2_DK_KsPiPi_LL(Bu_M)) * varDict['LL_d2kspp_dpi_to_dk_misID_frac2'])
    elif mode == 'b2dk_DD':
        pdfList['misid'] = lambda Bu_M: (norm_pdf(pdf_misid_CB1_DK_KsPiPi_DD(Bu_M)) * varDict['DD_d2kspp_dpi_to_dk_misID_frac1'] + norm_pdf(pdf_misid_CB2_DK_KsPiPi_DD(Bu_M)) * varDict['DD_d2kspp_dpi_to_dk_misID_frac2'])
    elif mode == 'b2dpi_LL':
        pdfList['misid'] = lambda Bu_M: norm_pdf(CBShape(Bu_M, varDict['LL_dk_to_dpi_misID_mean1'], varDict['LL_dk_to_dpi_misID_width1'], varDict['LL_dk_to_dpi_misID_alpha1'], varDict['LL_dk_to_dpi_misID_n1']))
    elif mode == 'b2dpi_DD':
        pdfList['misid'] = lambda Bu_M: norm_pdf(CBShape(Bu_M, varDict['DD_dk_to_dpi_misID_mean1'], varDict['DD_dk_to_dpi_misID_width1'], varDict['DD_dk_to_dpi_misID_alpha1'], varDict['DD_dk_to_dpi_misID_n1']))

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
    pdf_low_misID_Bu_Dstar0h_D0gamma_DPi_KsPiPi_LL = lambda Bu_M: ßHILLdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dpi'], varDict['low_b_Bu_Dstar0h_D0gamma_dpi'], varDict['low_csi_gamma'], varDict['m1ga_pi'], varDict['s1ga_pi'], varDict['m2ga_pi'], varDict['s2ga_pi'], varDict['m3ga_pi'], varDict['s3ga_pi'], varDict['m4ga_pi'], varDict['s4ga_pi'], varDict['f1ga_pi'], varDict['f2ga_pi'], varDict['f3ga_pi'])
    pdf_low_misID_Bu_Dstar0h_D0gamma_DPi_KsPiPi_DD = lambda Bu_M: ßHILLdini_misID(Bu_M, varDict['low_a_Bu_Dstar0h_D0gamma_dpi'], varDict['low_b_Bu_Dstar0h_D0gamma_dpi'], varDict['low_csi_gamma'], varDict['m1ga_pi'], varDict['s1ga_pi'], varDict['m2ga_pi'], varDict['s2ga_pi'], varDict['m3ga_pi'], varDict['s3ga_pi'], varDict['m4ga_pi'], varDict['s4ga_pi'], varDict['f1ga_pi'], varDict['f2ga_pi'], varDict['f3ga_pi'])

    # B2Dhpi
    # DPi
    pdf_B2Dpipi_1_DPi_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_I_B2Dpipi'],  varDict['low_b_I_B2Dpipi'],  varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_I_B2Dpipi'],  varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_1_DPi_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_I_B2Dpipi'],  varDict['low_b_I_B2Dpipi'],  varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_I_B2Dpipi'],  varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_2_DPi_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_II_B2Dpipi'], varDict['low_b_II_B2Dpipi'], varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_II_B2Dpipi'], varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_2_DPi_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_II_B2Dpipi'], varDict['low_b_II_B2Dpipi'], varDict['low_csi_B2Dpipi'], varDict['low_global_shift'], varDict['low_sigma_II_B2Dpipi'], varDict['low_ratio_B2Dpipi'], varDict['low_f_B2Dpipi'])
    pdf_B2Dpipi_DPi_KsPiPi_LL   = lambda Bu_M: pdf_B2Dpipi_1_DPi_KsPiPi_LL(Bu_M) * varDict['low_frac_B2Dpipi'] + pdf_B2Dpipi_2_DPi_KsPiPi_LL(Bu_M) * (1- varDict['low_frac_B2Dpipi'])
    pdf_B2Dpipi_DPi_KsPiPi_DD   = lambda Bu_M: pdf_B2Dpipi_1_DPi_KsPiPi_DD(Bu_M) * varDict['low_frac_B2Dpipi'] + pdf_B2Dpipi_2_DPi_KsPiPi_DD(Bu_M) * (1- varDict['low_frac_B2Dpipi'])
    # DK
    pdf_B2DKpi_1_DK_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_B2DKpi'], varDict['low_b_B2DKpi'], varDict['low_csi_B2DKpi'], varDict['low_global_shift'], varDict['low_sigma_B2DKpi'], varDict['low_ratio_B2DKpi'], varDict['low_f_B2DKpi'])
    pdf_B2DKpi_1_DK_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_B2DKpi'], varDict['low_b_B2DKpi'], varDict['low_csi_B2DKpi'], varDict['low_global_shift'], varDict['low_sigma_B2DKpi'], varDict['low_ratio_B2DKpi'], varDict['low_f_B2DKpi'])
    pdf_B2DKpi_2_DK_KsPiPi_LL = lambda Bu_M: Gaussian(Bu_M, varDict['low_mu_B2DKpi'], varDict['low_sigma_gaus_B2DKpi'])
    pdf_B2DKpi_2_DK_KsPiPi_DD = lambda Bu_M: Gaussian(Bu_M, varDict['low_mu_B2DKpi'], varDict['low_sigma_gaus_B2DKpi'])
    pdf_B2DKpi_DK_KsPiPi_LL   = lambda Bu_M: pdf_B2DKpi_1_DK_KsPiPi_LL(Bu_M) * varDict['low_frac_B2DKpi'] + pdf_B2DKpi_2_DK_KsPiPi_LL(Bu_M) * (1- varDict['low_frac_B2DKpi'])
    pdf_B2DKpi_DK_KsPiPi_DD   = lambda Bu_M: pdf_B2DKpi_1_DK_KsPiPi_DD(Bu_M) * varDict['low_frac_B2DKpi'] + pdf_B2DKpi_2_DK_KsPiPi_DD(Bu_M) * (1- varDict['low_frac_B2DKpi'])
    # DK misid
    pdf_low_misID_B2Dpipi_DK_KsPiPi_LL = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_B2Dpipi_misID'], varDict['low_b_B2Dpipi_misID'], varDict['low_csi_B2Dpipi'], varDict['low_m1_B2Dpipi_misID'], varDict['low_s1_B2Dpipi_misID'], varDict['low_m2_B2Dpipi_misID'], varDict['low_s2_B2Dpipi_misID'], varDict['low_m3_B2Dpipi_misID'], varDict['low_s3_B2Dpipi_misID'], varDict['low_m4_B2Dpipi_misID'], varDict['low_s4_B2Dpipi_misID'], varDict['low_f1_B2Dpipi_misID'], varDict['low_f2_B2Dpipi_misID'], varDict['low_f3_B2Dpipi_misID'])
    pdf_low_misID_B2Dpipi_DK_KsPiPi_DD = lambda Bu_M: HORNSdini_misID(Bu_M, varDict['low_a_B2Dpipi_misID'], varDict['low_b_B2Dpipi_misID'], varDict['low_csi_B2Dpipi'], varDict['low_m1_B2Dpipi_misID'], varDict['low_s1_B2Dpipi_misID'], varDict['low_m2_B2Dpipi_misID'], varDict['low_s2_B2Dpipi_misID'], varDict['low_m3_B2Dpipi_misID'], varDict['low_s3_B2Dpipi_misID'], varDict['low_m4_B2Dpipi_misID'], varDict['low_s4_B2Dpipi_misID'], varDict['low_f1_B2Dpipi_misID'], varDict['low_f2_B2Dpipi_misID'], varDict['low_f3_B2Dpipi_misID'])

    # Bs pdf
    pdf_low_Bs2DKPi_DK_KsPiPi_LL = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bs2DKpi'], varDict['low_b_Bs2DKpi'], varDict['low_csi_Bs2DKpi'], varDict['low_global_shift'], varDict['low_sigma_Bs2DKpi'], varDict['low_ratio_Bs2DKpi'], varDict['low_f_Bs2DKpi'])
    pdf_low_Bs2DKPi_DK_KsPiPi_DD = lambda Bu_M: HORNSdini(Bu_M, varDict['low_a_Bs2DKpi'], varDict['low_b_Bs2DKpi'], varDict['low_csi_Bs2DKpi'], varDict['low_global_shift'], varDict['low_sigma_Bs2DKpi'], varDict['low_ratio_Bs2DKpi'], varDict['low_f_Bs2DKpi'])

    if mode == 'b2dk_LL':
        pdfList['low_Bs2DKPi'] = lambda Bu_M: norm_pdf(pdf_low_Bs2DKPi_DK_KsPiPi_LL(Bu_M))
    elif mode == 'b2dk_DD':
        pdfList['low_Bs2DKPi'] = lambda Bu_M: norm_pdf(pdf_low_Bs2DKPi_DK_KsPiPi_DD(Bu_M))


    # Combine: with fractions
    if 'frac_low_Bu_Dstar0h_D0pi0_DPi' in varDict.keys():
        # Combine: DPi lowmass
        pdf_low_dpi_DPi_KsPiPi_LL = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DPi']*norm_pdf(pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_LL(Bu_M)) + varDict['frac_low_Bd_Dstarph_D0pi_DPi']* norm_pdf(pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_LL(Bu_M)) + varDict['frac_low_Bu_Dstar0h_D0gamma_DPi'] * norm_pdf(pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_LL(Bu_M)) + varDict['frac_low_B2Dpipi_DPi'] * norm_pdf(pdf_B2Dpipi_DPi_KsPiPi_LL(Bu_M))
        pdf_low_dpi_DPi_KsPiPi_DD = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DPi']*norm_pdf(pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_DD(Bu_M)) + varDict['frac_low_Bd_Dstarph_D0pi_DPi']* norm_pdf(pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_DD(Bu_M)) + varDict['frac_low_Bu_Dstar0h_D0gamma_DPi'] * norm_pdf(pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_DD(Bu_M)) + varDict['frac_low_B2Dpipi_DPi'] * norm_pdf(pdf_B2Dpipi_DPi_KsPiPi_DD(Bu_M))
        # Combine: DK lowmass
        pdf_low_dk_DK_KsPiPi_LL = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DK']*norm_pdf(pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL(Bu_M)) + varDict['frac_low_Bd_Dstarph_D0pi_DK']* norm_pdf(pdf_Bd_Dstarph_D0pi_DK_KsPiPi_LL(Bu_M)) + varDict['frac_low_Bu_Dstar0h_D0gamma_DK']* norm_pdf(pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL(Bu_M)) + varDict['frac_low_B2DKpi_DK']* norm_pdf(pdf_B2DKpi_DK_KsPiPi_LL(Bu_M))

        pdf_low_dk_DK_KsPiPi_DD = lambda Bu_M: varDict['frac_low_Bu_Dstar0h_D0pi0_DK']*norm_pdf(pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD(Bu_M)) + varDict['frac_low_Bd_Dstarph_D0pi_DK']* norm_pdf(pdf_Bd_Dstarph_D0pi_DK_KsPiPi_DD(Bu_M)) + varDict['frac_low_Bu_Dstar0h_D0gamma_DK'] * norm_pdf(pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD(Bu_M)) + varDict['frac_low_B2DKpi_DK']* norm_pdf(pdf_B2DKpi_DK_KsPiPi_DD(Bu_M))
        # Combine: DK misid
        pdf_low_misID_DK_KsPiPi_LL = lambda Bu_M: varDict['frac_low_misID_Bu_Dstar0h_D0pi0_DK']* norm_pdf(pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL(Bu_M)) + varDict['frac_low_misID_Bd_Dstarph_D0pi_DK']* norm_pdf(pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_LL(Bu_M)) + varDict['frac_low_misID_Bu_Dstar0h_D0gamma_DK']* norm_pdf(pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL(Bu_M)) + varDict['frac_low_misID_B2Dpipi_DK']* norm_pdf(pdf_low_misID_B2Dpipi_DK_KsPiPi_LL(Bu_M))
        pdf_low_misID_DK_KsPiPi_DD = lambda Bu_M: varDict['frac_low_misID_Bu_Dstar0h_D0pi0_DK']* norm_pdf(pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD(Bu_M)) + varDict['frac_low_misID_Bd_Dstarph_D0pi_DK']* norm_pdf(pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_DD(Bu_M)) + varDict['frac_low_misID_Bu_Dstar0h_D0gamma_DK']* norm_pdf(pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD(Bu_M)) + varDict['frac_low_misID_B2Dpipi_DK']* norm_pdf(pdf_low_misID_B2Dpipi_DK_KsPiPi_DD(Bu_M))
    # Combine: with ratios
    else:
        # Combine: DPi lowmass
        pdf_low_dst2dpi_DPi_KsPiPi_LL = addPdf([pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_LL, pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_LL], varDict['low_dpi_ratio_Bd_dst_vs_Bu_dst'])
        pdf_low_dst2dpi_DPi_KsPiPi_DD = addPdf([pdf_Bd_Dstarph_D0pi_DPi_KsPiPi_DD, pdf_Bu_Dstar0h_D0pi0_DPi_KsPiPi_DD], varDict['low_dpi_ratio_Bd_dst_vs_Bu_dst'])
        pdf_low_dpi_dst_DPi_KsPiPi_LL = addPdf([pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_LL, pdf_low_dst2dpi_DPi_KsPiPi_LL], varDict['low_dpi_ratio_dst2dgam_vs_dst2dpi'])
        pdf_low_dpi_dst_DPi_KsPiPi_DD = addPdf([pdf_Bu_Dstar0h_D0gamma_DPi_KsPiPi_DD, pdf_low_dst2dpi_DPi_KsPiPi_DD], varDict['low_dpi_ratio_dst2dgam_vs_dst2dpi'])
        pdf_low_dpi_DPi_KsPiPi_LL     = addPdf([pdf_B2Dpipi_DPi_KsPiPi_LL, pdf_low_dpi_dst_DPi_KsPiPi_LL], varDict['low_dpi_ratio_b2drho_vs_b2dstpi'])
        pdf_low_dpi_DPi_KsPiPi_DD     = addPdf([pdf_B2Dpipi_DPi_KsPiPi_DD, pdf_low_dpi_dst_DPi_KsPiPi_DD], varDict['low_dpi_ratio_b2drho_vs_b2dstpi'])
        # Combine: DK lowmass
        pdf_low_dst2dpi_DK_KsPiPi_LL = addPdf([pdf_Bd_Dstarph_D0pi_DK_KsPiPi_LL, pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL], varDict['low_dk_ratio_Bd_dst_vs_Bu_dst']) 
        pdf_low_dst2dpi_DK_KsPiPi_DD = addPdf([pdf_Bd_Dstarph_D0pi_DK_KsPiPi_DD, pdf_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD], varDict['low_dk_ratio_Bd_dst_vs_Bu_dst']) 
        pdf_low_dk_dst_DK_KsPiPi_LL  = addPdf([pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL, pdf_low_dst2dpi_DK_KsPiPi_LL], varDict['low_dk_ratio_dst2dgam_vs_dst2dk'])
        pdf_low_dk_dst_DK_KsPiPi_DD  = addPdf([pdf_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD, pdf_low_dst2dpi_DK_KsPiPi_DD], varDict['low_dk_ratio_dst2dgam_vs_dst2dk'])
        pdf_low_dk_DK_KsPiPi_LL      = addPdf([pdf_B2DKpi_DK_KsPiPi_LL, pdf_low_dk_dst_DK_KsPiPi_LL], varDict['low_dk_ratio_b2dkst_vs_b2dstk'])
        pdf_low_dk_DK_KsPiPi_DD      = addPdf([pdf_B2DKpi_DK_KsPiPi_DD, pdf_low_dk_dst_DK_KsPiPi_DD], varDict['low_dk_ratio_b2dkst_vs_b2dstk'])
        # Combine: DK misid
        pdf_low_misID_dst2dpi_DK_KsPiPi_LL = addPdf([pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_LL, pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_LL], varDict['low_dpi_to_dk_misID_ratio_Bd_dst_vs_Bu_dst'])
        pdf_low_misID_dst2dpi_DK_KsPiPi_DD = addPdf([pdf_low_misID_Bd_Dstarph_D0pi_DK_KsPiPi_DD, pdf_low_misID_Bu_Dstar0h_D0pi0_DK_KsPiPi_DD], varDict['low_dpi_to_dk_misID_ratio_Bd_dst_vs_Bu_dst'])
        pdf_low_misID_dst_DK_KsPiPi_LL     = addPdf([pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_LL, pdf_low_misID_dst2dpi_DK_KsPiPi_LL], varDict['low_dpi_to_dk_misID_ratio_dst2dgam_vs_dst2dpi'])
        pdf_low_misID_dst_DK_KsPiPi_DD     = addPdf([pdf_low_misID_Bu_Dstar0h_D0gamma_DK_KsPiPi_DD, pdf_low_misID_dst2dpi_DK_KsPiPi_DD], varDict['low_dpi_to_dk_misID_ratio_dst2dgam_vs_dst2dpi'])
        pdf_low_misID_DK_KsPiPi_LL         = addPdf([pdf_low_misID_B2Dpipi_DK_KsPiPi_LL, pdf_low_misID_dst_DK_KsPiPi_LL], varDict['low_dpi_to_dk_misID_ratio_b2drho_vs_b2dstpi'])
        pdf_low_misID_DK_KsPiPi_DD         = addPdf([pdf_low_misID_B2Dpipi_DK_KsPiPi_DD, pdf_low_misID_dst_DK_KsPiPi_DD], varDict['low_dpi_to_dk_misID_ratio_b2drho_vs_b2dstpi'])

    if mode == 'b2dk_LL':
        pdfList['low'] = lambda Bu_M: (pdf_low_dk_DK_KsPiPi_LL(Bu_M))
        pdfList['low_misID'] = lambda Bu_M: (pdf_low_misID_DK_KsPiPi_LL(Bu_M))

    elif mode == 'b2dk_DD':
        pdfList['low'] = lambda Bu_M: (pdf_low_dk_DK_KsPiPi_DD(Bu_M))
        pdfList['low_misID'] = lambda Bu_M: (pdf_low_misID_DK_KsPiPi_DD(Bu_M))

    elif mode == 'b2dpi_LL':
        pdfList['low'] = lambda Bu_M: (pdf_low_dpi_DPi_KsPiPi_LL(Bu_M))

    elif mode == 'b2dpi_DD':
        pdfList['low'] = lambda Bu_M: (pdf_low_dpi_DPi_KsPiPi_DD(Bu_M))


    # combinatorial
    print('--- Constructing comb pdfs...')
    if mode == 'b2dk_LL':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Exponential(Bu_M, varDict['comb_const_dk_d2kspp_LL']))
    elif mode == 'b2dk_DD':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Exponential(Bu_M, varDict['comb_const_dk_d2kspp_DD']))
    elif mode == 'b2dpi_LL':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Exponential(Bu_M, varDict['comb_const_dpi_d2kspp_LL']))
    elif mode == 'b2dpi_DD':
        pdfList['comb'] = lambda Bu_M: norm_pdf(Exponential(Bu_M, varDict['comb_const_dpi_d2kspp_DD']))


    print('--- INFO: Writing to file...')
    return pdfList    