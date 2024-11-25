#include <complex>
#include <iostream>
using namespace std;

#include "D0ToKspipi2018.h"

// using namespace std;
// const double mD0 = 1.8648399;
// const double mKs = 0.49761401;
// const double mPi = 0.13957017;

    // Default constructor
D0ToKspipi2018::D0ToKspipi2018() {}

D0ToKspipi2018::~D0ToKspipi2018(){}


void D0ToKspipi2018::init(){
  cout << "INITIALISATION OF THE TEST !" << endl;
}

complex<double> D0ToKspipi2018::get_amp()
{

  // init();
  cout << "I'm trying something here" << endl;
  complex<double> ampD0  = complex<double>(-11.5075,-6.47542);
  complex<double> ampD0b = complex<double>(14.7697,-7.15437);
  complex<double> result = ampD0; //, ampD0b};
  return result;
}
