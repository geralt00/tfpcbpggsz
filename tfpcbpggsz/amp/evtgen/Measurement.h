#ifndef MEASUREMENT_H 
#define MEASUREMENT_H 1


// Include files
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>

// Definitions 
#define MAXERRORS 4

// Usage
using namespace std;

/** @class Measurement Measurement.h 
 *  A class to handle authomatic error propagation in normal operations 
 *
 *  @author Francesco Dettori
 *  @date   2013-03-24,
  *  
 TODO list:
 - generic math functions 
 - Correct implementation of the asymmetric errors propagation
*/
class Measurement {
public: 
  /// Standard constructor
  Measurement( ){
    m_value = 0;
    for(int i =0; i<MAXERRORS; i++) m_errors[i]=0.;
    
  }; 
  
  Measurement(double value, double error);
  
  Measurement(double value, double first_error, double second_error, int scheme=1);
  
  Measurement(double value, 
	      double first_error, double second_error, 
	      double third_error, double fourth_error);

  static Measurement AMeasurement(double value, double upper_error, double lower_error);
  
  static Measurement SMeasurement(double value, double stat_error, double syst_error);

  static Measurement Efficiency(double eff, double n_tot);  
  
  static Measurement GetEfficiency(double n_sel, double n_tot);  
  
  static Measurement GetEfficiency(Measurement n_sel, Measurement n_tot);
  
  virtual ~Measurement( ); ///< Destructor

  // Setters and getters 
  void set_error(double err);
  void set_errors(double first_err, double second_err, int to_scheme =1);
  double *errors(){
    return m_errors;
  }

  double error(){ return total_error(); };
    
  double total_error() const;
  double relative_error() const { return total_error()/m_value;};
  double rel_error()const   {return relative_error();  };
  
  double get_min_error() const;
  
  
  double value(){ return m_value; }
  void set_value(double value){m_value = value; };
  
  // Operators with measurements
  Measurement operator+(const Measurement &a) const;
  Measurement operator-(const Measurement &a) const;
  Measurement operator*(const Measurement &a) const;
  Measurement operator/(const Measurement &a) const;
  //  Measurement operator=(const Measurement &a){ return Measurement(*this);};

  Measurement operator+=(const Measurement &a) {(*this = *this+a); return *this;}
  Measurement operator-=(const Measurement &a) {(*this = *this-a); return *this;}
  Measurement operator*=(const Measurement &a) {(*this = *this*a); return *this;}
  Measurement operator/=(const Measurement &a) {(*this = *this/a); return *this;}
  bool operator>(Measurement b) const { return m_value > b.m_value; }
  bool operator<(Measurement b) const { return m_value < b.m_value; }
  

  // Operators with numbers
  Measurement operator+(double b) const ;
  Measurement operator-(double b) const ;
  Measurement operator*(double b) const ;
  Measurement operator/(double b) const ;
  Measurement operator+=(double b){ *this = *this+b; return *this;}
  Measurement operator-=(double b) { *this = *this-b; return *this;}
  Measurement operator*=(double b) { *this = *this*b; return *this;}
  Measurement operator/=(double b) { *this = *this/b; return *this;}
    
  bool operator==(double b) const;
  bool operator!=(double b) const;
  bool operator>(double b) const;
  bool operator<(double b) const;
  

  
  friend Measurement operator+(double b, const Measurement &a);
  friend Measurement operator-(double b, const Measurement &a);
  friend Measurement operator*(double b, const Measurement &a);
  friend Measurement operator/(double b, const Measurement &a);

  // Operators with ostream
  friend ostream& operator<<(ostream& os, const Measurement& dt);
  operator const char*() const
  { return str().data(); };
  
  operator char*() const
  { return (char*)str().data(); };
  
  
  string str() const;
  string tex() const;
  string stex(bool print_separated = true) const;
 
  const char* getLaTeX() const;
  
  // Math functions 
  Measurement inverse() const;
  Measurement average(Measurement b) const;
  Measurement power(double power) const;
  Measurement ln() const;

  void add_to_errors(double err, int to_scheme=0);
  void add_to_errors(double first_error, double second_error, int to_scheme =1);
  void add_to_syst_errors(double err);
  void add_to_syst_errors(double first_error, double second_error);

  
  
  void symmetrize();
  
  void compress();

  void debug();

protected:
  
  void sum_err_quad(Measurement a, Measurement b,  Measurement &result) const;

  void sum_err_log(Measurement a, Measurement b,  Measurement &result) const;

  void product_err(Measurement a, Measurement b,  Measurement &result) const;

  void division_err(Measurement a, Measurement b,  Measurement &result) const;

  void scale_error(double scale);

  double * get_errors(int scheme);



  int common_scheme(Measurement a); 

  stringstream& _sstr(stringstream &s) const ;
  stringstream& _stex(stringstream &s) const ;
  stringstream& _stex_sci(stringstream &s, bool print_separated = true) const ;

private:
  double m_value;
  double m_errors[MAXERRORS];

  int m_scheme; // Change this to an enum ?
  /* one error: 
     - 0 = all in one error 
     two errors: 
     - 1 = asymmetric error 
     - 2 = symmetric (stat + syst) 
     four errors: 
     - 3 = asymmetric (stat and syst)
  */
  
  
};




#endif // MEASUREMENT_H
