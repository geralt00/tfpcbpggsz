#ifndef D0TOKSPIPI2018_H
#define D0TOKSPIPI2018_H

//#include "EvtGenBase/EvtComplex.hh"
#include "QCMCYinghaoChange/common_definitions.h"

#include <vector>
#include <complex>
#include "TComplex.h"
#include <regex.h>

using namespace std;

class D0ToKspipi2018{

public:
    D0ToKspipi2018(){};
    virtual ~D0ToKspipi2018();
    
    void init();

    bool inDalitz(Double_t x, Double_t y);
    //bool inDalitz_01_02(Double_t x, Double_t y);
    bool inDalitz_01_02(Double_t x, Double_t y);

    double computeNormalization(Int_t npar, Double_t *par);

    // setup Dalitz model START
    string m_Dalitzmodel;

    // arrays for the Dalitz model results without systematics

    double _DalitzNormalization;
    double computeDalitzAmpSquaredNormalizationIntegral();
    TComplex Amp(double *x, double *par);
    TComplex Amp_PFT(double *x);
    double GetStrongPhase(double *x);
    int getBin(double *x);
    
    int _nd;
    
    int _parameter_number[NUMBER_PARAMETERS];
    double _parameter_value[NUMBER_PARAMETERS];
    double _parameter_error[NUMBER_PARAMETERS];
};
#endif
