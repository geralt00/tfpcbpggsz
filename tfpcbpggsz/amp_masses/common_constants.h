#ifndef INCLUDES_COMMON_CONSTANTS_H 
#define INCLUDES_COMMON_CONSTANTS_H 1

#include "Measurement.h"
#include "Measurement.cpp"

// PDF constants
Measurement PDG_m_pi = Measurement(139.57039,0.00018);
Measurement PDG_m_Ks = Measurement(497.611,0.013);
Measurement PDG_m_Dz = Measurement(1864.84,0.05);

// QMI method
Measurement QMI_smax_Kspi = (PDG_m_Dz-PDG_m_pi).power(2);
Measurement QMI_smin_Kspi = (PDG_m_Ks+PDG_m_pi).power(2);
double QMI_zpmax_Kspi =  3686290; // MeV2
double QMI_zpmin_Kspi =  1894890;// MeV2
double QMI_zmmax_Kspi =  2485740;// MeV2
double QMI_zmmin_Kspi = -2485740;// MeV2


#endif 
