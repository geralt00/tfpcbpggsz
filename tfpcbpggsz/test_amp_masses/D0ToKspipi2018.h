#ifndef D0TOKSPIPI2018_H
#define D0TOKSPIPI2018_H

#include <complex>

class D0ToKspipi2018 {
public:
  D0ToKspipi2018();
  ~D0ToKspipi2018();
  void init();  
  std::complex<double> get_amp();  
};
#endif
