import cupy as cp
from scipy.special import erf

def norm_pdf(pdf):
    norm_term = cp.mean(pdf)
    return cp.array(pdf/norm_term/len(pdf))

def addPdf(pdflist=[], frac=[]):
    if isinstance(pdflist, cp.ndarray):
        pdflist = [pdflist]  # Wrap single numpy array in a list
    pdflist = [cp.array(pdf, dtype=float) for pdf in pdflist]  # Ensure all elements are numpy arrays

    if not isinstance(frac, list):
        frac = [frac]  # Convert single fraction to list

    # Initialize the PDF with zeros of the same shape as the first PDF in the list
    pdf = cp.zeros_like(pdflist[0], dtype=float)

    # Normalize PDFs in pdflist if their sum is not exactly 1
    for i in range(len(pdflist)):
        if cp.sum(pdflist[i]) != 1:
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

    firstG1 = ((2*(a_new-2*B_NEW+(m-shift))*sigma)/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*sigma*sigma)) - (2*(b_new-2*B_NEW+(m-shift))*sigma)/cp.exp((b_new-(m-shift))*(b_new-(m-shift))/(2*sigma*sigma))+ cp.sqrt(2*cp.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma)*erf((-a_new+(m-shift))/(cp.sqrt(2)*sigma))  - cp.sqrt(2*cp.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma*sigma) * erf((-b_new+(m-shift))/(cp.sqrt(2)*sigma)))/(2*cp.sqrt(2*cp.pi))
    secondG1 = (((2*sigma*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(a_new+(m-shift)) + 2*(sigma*sigma)))/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma*sigma))) - (2*sigma*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(b_new + (m-shift)) + 2*(sigma*sigma)))/cp.exp((b_new - (m-shift))*(b_new - (m-shift))/(2*(sigma*sigma))) - cp.sqrt(2*cp.pi)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma*sigma))*erf((-a_new + (m-shift))/(cp.sqrt(2)*sigma)) + cp.sqrt(2*cp.pi)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma*sigma)) *erf((-b_new + (m-shift))/(cp.sqrt(2)*sigma)))/(2 *cp.sqrt(2*cp.pi)))

    CURVEG1 = cp.abs((1-csi)*secondG1 + (b_new*csi - a_new)*firstG1)

    firstG2 = ((2*(a_new-2*B_NEW+(m-shift))*sigma2)/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*sigma2*sigma2)) - (2*(b_new-2*B_NEW+(m-shift))*sigma2)/cp.exp((b_new-(m-shift))*(b_new-(m-shift))/(2*sigma2*sigma2))+ cp.sqrt(2*cp.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2*sigma2)*erf((-a_new+(m-shift))/(cp.sqrt(2)*sigma2))  - cp.sqrt(2*cp.pi)*((B_NEW-(m-shift))*(B_NEW-(m-shift)) + sigma2*sigma2) * erf((-b_new+(m-shift))/(cp.sqrt(2)*sigma2)))/(2*cp.sqrt(2*cp.pi))
    secondG2 = (((2*sigma2*(a_new*a_new + B_NEW*B_NEW + a_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(a_new+(m-shift)) + 2*(sigma2*sigma2)))/cp.exp((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma2*sigma2))) - (2*sigma2*(b_new*b_new + B_NEW*B_NEW + b_new*(m-shift) + (m-shift)*(m-shift) - 2*B_NEW*(b_new + (m-shift)) + 2*(sigma2*sigma2)))/cp.exp((b_new - (m-shift))*(b_new - (m-shift))/(2*(sigma2*sigma2))) - cp.sqrt(2*cp.pi)*(-((B_NEW - (m-shift))*(B_NEW - (m-shift)) *(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma2*sigma2))*erf((-a_new + (m-shift))/(cp.sqrt(2)*sigma2)) + cp.sqrt(2*cp.pi)* (-((B_NEW - (m-shift))*(B_NEW - (m-shift))*(m-shift)) + (2*B_NEW - 3*(m-shift))*(sigma2*sigma2)) *erf((-b_new + (m-shift))/(cp.sqrt(2)*sigma2)))/(2 *cp.sqrt(2*cp.pi)))

    CURVEG2 = cp.abs((1-csi)*secondG2 + (b_new*csi - a_new)*firstG2)

    return (fraction_sigma*CURVEG1 + (1-fraction_sigma)*CURVEG2)


def CruijffExtended(m, m0, sigmaL, sigmaR, alphaL, alphaR, beta):
    sigma = 0.0
    alpha = 0.0
    dx = m - m0

    sigma = cp.where(dx < 0, sigmaL, sigmaR)
    alpha = cp.where(dx < 0, alphaL, alphaR)

    f = 2.0*sigma*sigma + alpha*dx*dx
    return cp.exp(-dx**2 *(1 + beta * dx **2)/f)  


def CBShape(m,m0,sigma,alpha,n):
   t =(m-m0)/sigma

   if alpha <0:
      t = -t
   else:
      t = t

   absAlpha = cp.abs(alpha) 
   val_a = cp.exp(-0.5*t*t)
   
   a =  cp.power(n/absAlpha,n)*cp.exp(-0.5*absAlpha*absAlpha)
   b= n/absAlpha - absAlpha
   val_b = a/cp.power(b-t, n)

   val = cp.where(t >= -absAlpha, val_a, val_b)
   return val

def HORNSdini_misID(m,a,b,csi,m1,s1,m2,s2,m3,s3,m4,s4,f1,f2,f3):
    a_new = a
    b_new = b
    B_NEW = (a_new + b_new) / 2.0


    firstG1 = ((2 * (a_new - 2 * B_NEW + (m - m1)) * s1) / cp.exp((a_new - (m - m1)) **2 / (2 * s1**2 )) - (2 * (b_new - 2 * B_NEW + (m - m1)) * s1) / cp.exp((b_new - (m - m1)) **2 / (2 * (s1 **2))) + cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m1)) **2 + s1 ** 2) * erf((-a_new + (m - m1)) / (cp.sqrt(2) * s1)) - cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m1)) ** 2 + s1 **2) * erf((-b_new + (m - m1)) / (cp.sqrt(2) * s1))) / (2 * cp.sqrt(2 * cp.pi))
    secondG1 = (((2*s1*(a_new**2 + B_NEW**2 + a_new*(m-m1) + (m-m1)**2 - 2*B_NEW*(a_new+(m-m1)) + 2*s1**2 ))/cp.exp((a_new-(m-m1))**2 /(2*(s1**2))) - (2*s1*(b_new**2 + B_NEW**2 + b_new*(m-m1) + (m-m1)**2 - 2*B_NEW*(b_new + (m-m1)) + 2*(s1**2)))/cp.exp((b_new - (m-m1))**2 /(2*(s1**2))) - cp.sqrt(2*cp.pi)*(-((B_NEW - (m-m1))**2 *(m-m1)) + (2*B_NEW - 3*(m-m1))*(s1**2))*erf((-a_new + (m-m1))/(cp.sqrt(2)*s1)) + cp.sqrt(2*cp.pi)* (-((B_NEW - (m-m1))**2 *(m-m1)) + (2*B_NEW - 3*(m-m1))*(s1**2)) *erf((-b_new + (m-m1))/(cp.sqrt(2)*s1)))/(2 *cp.sqrt(2*cp.pi)))
    CURVEG1 = cp.fabs((1-csi)*secondG1 + (b_new*csi - a_new)*firstG1)

    firstG2 = ((2 * (a_new - 2 * B_NEW + (m - m2)) * s2) / cp.exp((a_new - (m - m2)) **2 / (2 * s2**2 )) - (2 * (b_new - 2 * B_NEW + (m - m2)) * s2) / cp.exp((b_new - (m - m2)) **2 / (2 * (s2 **2))) + cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m2)) **2 + s2 ** 2) * erf((-a_new + (m - m2)) / (cp.sqrt(2) * s2)) - cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m2)) ** 2 + s2 **2) * erf((-b_new + (m - m2)) / (cp.sqrt(2) * s2))) / (2 * cp.sqrt(2 * cp.pi))
    secondG2 = (((2*s2*(a_new**2 + B_NEW**2 + a_new*(m-m2) + (m-m2)**2 - 2*B_NEW*(a_new+(m-m2)) + 2*s2**2 ))/cp.exp((a_new-(m-m2))**2 /(2*(s2**2))) - (2*s2*(b_new**2 + B_NEW**2 + b_new*(m-m2) + (m-m2)**2 - 2*B_NEW*(b_new + (m-m2)) + 2*(s2**2)))/cp.exp((b_new - (m-m2))**2 /(2*(s2**2))) - cp.sqrt(2*cp.pi)*(-((B_NEW - (m-m2))**2 *(m-m2)) + (2*B_NEW - 3*(m-m2))*(s2**2))*erf((-a_new + (m-m2))/(cp.sqrt(2)*s2)) + cp.sqrt(2*cp.pi)* (-((B_NEW - (m-m2))**2 *(m-m2)) + (2*B_NEW - 3*(m-m2))*(s2**2)) *erf((-b_new + (m-m2))/(cp.sqrt(2)*s2)))/(2 *cp.sqrt(2*cp.pi)))
    CURVEG2 = cp.fabs((1-csi)*secondG2 + (b_new*csi - a_new)*firstG2)

    firstG3 = ((2 * (a_new - 2 * B_NEW + (m - m3)) * s3) / cp.exp((a_new - (m - m3)) **2 / (2 * s3**2 )) - (2 * (b_new - 2 * B_NEW + (m - m3)) * s3) / cp.exp((b_new - (m - m3)) **2 / (2 * (s3 **2))) + cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m3)) **2 + s3 ** 2) * erf((-a_new + (m - m3)) / (cp.sqrt(2) * s3)) - cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m3)) ** 2 + s3 **2) * erf((-b_new + (m - m3)) / (cp.sqrt(2) * s3))) / (2 * cp.sqrt(2 * cp.pi))
    secondG3 = (((2*s3*(a_new**2 + B_NEW**2 + a_new*(m-m3) + (m-m3)**2 - 2*B_NEW*(a_new+(m-m3)) + 2*s3**2 ))/cp.exp((a_new-(m-m3))**2 /(2*(s3**2))) - (2*s3*(b_new**2 + B_NEW**2 + b_new*(m-m3) + (m-m3)**2 - 2*B_NEW*(b_new + (m-m3)) + 2*(s3**2)))/cp.exp((b_new - (m-m3))**2 /(2*(s3**2))) - cp.sqrt(2*cp.pi)*(-((B_NEW - (m-m3))**2 *(m-m3)) + (2*B_NEW - 3*(m-m3))*(s3**2))*erf((-a_new + (m-m3))/(cp.sqrt(2)*s3)) + cp.sqrt(2*cp.pi)* (-((B_NEW - (m-m3))**2 *(m-m3)) + (2*B_NEW - 3*(m-m3))*(s3**2)) *erf((-b_new + (m-m3))/(cp.sqrt(2)*s3)))/(2 *cp.sqrt(2*cp.pi)))
    CURVEG3 = cp.fabs((1-csi)*secondG3 + (b_new*csi - a_new)*firstG3)

    firstG4 = ((2 * (a_new - 2 * B_NEW + (m - m4)) * s4) / cp.exp((a_new - (m - m4)) **2 / (2 * s4**2 )) - (2 * (b_new - 2 * B_NEW + (m - m4)) * s4) / cp.exp((b_new - (m - m4)) **2 / (2 * (s4 **2))) + cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m4)) **2 + s4 ** 2) * erf((-a_new + (m - m4)) / (cp.sqrt(2) * s4)) - cp.sqrt(2 * cp.pi) * ((B_NEW - (m - m4)) ** 2 + s4 **2) * erf((-b_new + (m - m4)) / (cp.sqrt(2) * s4))) / (2 * cp.sqrt(2 * cp.pi))
    secondG4 = (((2*s4*(a_new**2 + B_NEW**2 + a_new*(m-m4) + (m-m4)**2 - 2*B_NEW*(a_new+(m-m4)) + 2*s4**2 ))/cp.exp((a_new-(m-m4))**2 /(2*(s4**2))) - (2*s4*(b_new**2 + B_NEW**2 + b_new*(m-m4) + (m-m4)**2 - 2*B_NEW*(b_new + (m-m4)) + 2*(s4**2)))/cp.exp((b_new - (m-m4))**2 /(2*(s4**2))) - cp.sqrt(2*cp.pi)*(-((B_NEW - (m-m4))**2 *(m-m4)) + (2*B_NEW - 3*(m-m4))*(s4**2))*erf((-a_new + (m-m4))/(cp.sqrt(2)*s4)) + cp.sqrt(2*cp.pi)* (-((B_NEW - (m-m4))**2 *(m-m4)) + (2*B_NEW - 3*(m-m4))*(s4**2)) *erf((-b_new + (m-m4))/(cp.sqrt(2)*s4)))/(2 *cp.sqrt(2*cp.pi)))
    CURVEG4 = cp.fabs((1-csi)*secondG4 + (b_new*csi - a_new)*firstG4)


    return cp.fabs(f1*CURVEG1) + cp.fabs(f2*CURVEG2) + cp.fabs(f3*CURVEG3) + cp.fabs((1-f1-f2-f3)*CURVEG4)

    
def HILLdini(m,a,b,csi,shift,sigma,ratio_sigma,fraction_sigma):
    a_new = a
    b_new = b
    sigma2 = sigma * ratio_sigma

    firstG1 = (2*cp.exp(-((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma*sigma))))*sigma*(b_new-(m-shift))+2*cp.exp(-((b_new-(m-shift))*(b_new-(m-shift))/(2*(sigma*sigma))))*sigma*(-a_new+(m-shift))-cp.sqrt(2*cp.pi)*(a_new*b_new+(sigma*sigma)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-a_new+(m-shift))/(cp.sqrt(2)*sigma))+cp.sqrt(2*cp.pi)*(a_new*b_new+(sigma*sigma)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-b_new+(m-shift))/(cp.sqrt(2)*sigma)))/(2*cp.sqrt(2*cp.pi))
    CURVEG1 = cp.fabs((1-csi)/(b_new - a_new)*m + (b_new*csi - a_new)/(b_new-a_new))*cp.fabs(firstG1)

    firstG2 = (2*cp.exp(-((a_new-(m-shift))*(a_new-(m-shift))/(2*(sigma2*sigma2))))*sigma2*(b_new-(m-shift))+2*cp.exp(-((b_new-(m-shift))*(b_new-(m-shift))/(2*(sigma2*sigma2))))*sigma2*(-a_new+(m-shift))-cp.sqrt(2*cp.pi)*(a_new*b_new+(sigma2*sigma2)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-a_new+(m-shift))/(cp.sqrt(2)*sigma2))+cp.sqrt(2*cp.pi)*(a_new*b_new+(sigma2*sigma2)-(a_new+b_new)*(m-shift)+((m-shift)*(m-shift)))*erf((-b_new+(m-shift))/(cp.sqrt(2)*sigma2)))/(2*cp.sqrt(2*cp.pi))
    CURVEG2 = cp.fabs((1-csi)/(b_new - a_new)*m + (b_new*csi - a_new)/(b_new-a_new))*cp.fabs(firstG2)

    return cp.fabs(fraction_sigma*CURVEG1) + cp.fabs((1-fraction_sigma)*CURVEG2)

def HILLdini_misID(m,a,b,csi,m1,s1,m2,s2,m3,s3,m4,s4,f1,f2,f3):
    a_new = a
    b_new = b
    B_NEW = (a_new + b_new) / 2.0

    firstG1 = (2*cp.exp(-((a_new-(m-m1))*(a_new-(m-m1))/(2*(s1*s1))))*s1*(b_new-(m-m1))+2*cp.exp(-((b_new-(m-m1))*(b_new-(m-m1))/(2*(s1*s1))))*s1*(-a_new+(m-m1))-cp.sqrt(2*cp.pi)*(a_new*b_new+(s1*s1)-(a_new+b_new)*(m-m1)+((m-m1)*(m-m1)))*erf((-a_new+(m-m1))/(cp.sqrt(2)*s1))+cp.sqrt(2*cp.pi)*(a_new*b_new+(s1*s1)-(a_new+b_new)*(m-m1)+((m-m1)*(m-m1)))*erf((-b_new+(m-m1))/(cp.sqrt(2)*s1)))/(2*cp.sqrt(2*cp.pi))
    CURVEG1 = cp.fabs((1-csi)/(b_new-a_new)*(m-m1)  + (b_new*csi - a_new)/(b_new-a_new)  )*cp.fabs(firstG1)

    firstG2 = (2*cp.exp(-((a_new-(m-m2))*(a_new-(m-m2))/(2*(s2*s2))))*s2*(b_new-(m-m2))+2*cp.exp(-((b_new-(m-m2))*(b_new-(m-m2))/(2*(s2*s2))))*s2*(-a_new+(m-m2))-cp.sqrt(2*cp.pi)*(a_new*b_new+(s2*s2)-(a_new+b_new)*(m-m2)+((m-m2)*(m-m2)))*erf((-a_new+(m-m2))/(cp.sqrt(2)*s2))+cp.sqrt(2*cp.pi)*(a_new*b_new+(s2*s2)-(a_new+b_new)*(m-m2)+((m-m2)*(m-m2)))*erf((-b_new+(m-m2))/(cp.sqrt(2)*s2)))/(2*cp.sqrt(2*cp.pi))
    CURVEG2 = cp.fabs((1-csi)/(b_new-a_new)*(m-m2)  + (b_new*csi - a_new)/(b_new-a_new)  )*cp.fabs(firstG2)

    firstG3 = (2*cp.exp(-((a_new-(m-m3))*(a_new-(m-m3))/(2*(s3*s3))))*s3*(b_new-(m-m3))+2*cp.exp(-((b_new-(m-m3))*(b_new-(m-m3))/(2*(s3*s3))))*s3*(-a_new+(m-m3))-cp.sqrt(2*cp.pi)*(a_new*b_new+(s3*s3)-(a_new+b_new)*(m-m3)+((m-m3)*(m-m3)))*erf((-a_new+(m-m3))/(cp.sqrt(2)*s3))+cp.sqrt(2*cp.pi)*(a_new*b_new+(s3*s3)-(a_new+b_new)*(m-m3)+((m-m3)*(m-m3)))*erf((-b_new+(m-m3))/(cp.sqrt(2)*s3)))/(2*cp.sqrt(2*cp.pi))
    CURVEG3 = cp.fabs((1-csi)/(b_new-a_new)*(m-m3)  + (b_new*csi - a_new)/(b_new-a_new)  )*cp.fabs(firstG3)

    firstG4 = (2*cp.exp(-((a_new-(m-m4))*(a_new-(m-m4))/(2*(s4*s4))))*s4*(b_new-(m-m4))+2*cp.exp(-((b_new-(m-m4))*(b_new-(m-m4))/(2*(s4*s4))))*s4*(-a_new+(m-m4))-cp.sqrt(2*cp.pi)*(a_new*b_new+(s4*s4)-(a_new+b_new)*(m-m4)+((m-m4)*(m-m4)))*erf((-a_new+(m-m4))/(cp.sqrt(2)*s4))+cp.sqrt(2*cp.pi)*(a_new*b_new+(s4*s4)-(a_new+b_new)*(m-m4)+((m-m4)*(m-m4)))*erf((-b_new+(m-m4))/(cp.sqrt(2)*s4)))/(2*cp.sqrt(2*cp.pi))
    CURVEG4 = cp.fabs((1-csi)/(b_new-a_new)*(m-m4)  + (b_new*csi - a_new)/(b_new-a_new)  )*cp.fabs(firstG4)

    return cp.fabs(f1*CURVEG1) + cp.fabs(f2*CURVEG2) + cp.fabs(f3*CURVEG3) + cp.fabs((1-f1-f2-f3)*CURVEG4)

def Gaussian(m, mu, sigma):
    return (cp.exp(-0.5*((m-mu)/sigma)**2))/(sigma*cp.sqrt(2*cp.pi))

def Exponential(m, c):
    return cp.exp(c*m)