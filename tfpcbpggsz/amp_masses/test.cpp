#include "D0ToKspipi2018.h"
#include "D0ToKspipi2018.cxx"
#include <complex>
#include <vector>

int main(void) 
{
  std::cout << "Let's test this thing" << std::endl;;
  D0ToKspipi2018* Kspipi = new D0ToKspipi2018();
  Kspipi->init();
  std::vector<std::complex<double>> res = Kspipi->get_amp(0.528122750730856, 0.4777242753240625);
  std::cout << res[0] << std::endl;
  std::cout << res[1] << std::endl;
  return 0;
}
