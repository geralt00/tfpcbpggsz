##### python translation from Francesco Dettori's cpp package
# -----------------------------------------------------------------------------
#  Measurement
#  A class to handle authomatic error propagation with normal operations
#  Assumes gaussian approximation
#  2013-03-24 : Francesco Dettori
# -----------------------------------------------------------------------------
import math
import numpy as np

def sum_err_quad(err1,err2):
    _res = 0
    _res = math.sqrt(err1**2+err2**2)
    return _res

def product_err(value1, err1, value2, err2):
    _res = 0
    _res = math.sqrt( (err1*value2)**2 + (err2*value1)**2 )
    return _res

def division_err(value1, err1, value2, err2):
    _res = 0
    _res = value1/value2*math.sqrt( (err1/value1)**2 + (err2/value2)**2 )
    return _res

##=============================================================================
## Standard constructor, initializes variables
##=============================================================================
class Measurement:
    def __init__(self, value, error, isEff=False):
        if (isEff==True):
            ## value = N_pass, error = N_total = (N_pass + N_fail)
            if (value > error):
                print("ERROR IN MEASUREMENT - trying to get an efficiency with pass > total")
                print("REMINDER, 1st arg=pass, 2nd arg=total in the constructor")
                self.value = -1
                self.error = -1
                pass
            else:
                self.value = float(value) / float(error)
                self.error = self.get_error_efficiency(value, error)
                pass
            pass
        else:
            self.value = value
            self.error = abs(error)
        pass

    def get_error_efficiency(self, N_pass, N_total):
        _eff = float(N_pass) / float(N_total)
        _res = math.sqrt(_eff * (1 - _eff) / N_total )
        return _res
    
    def __repr__(self):
        return "Measurement"
    
    def __str__(self):
        _value = str(self.value)
        _error = str(self.error)
        _res = _value+ " +- "+_error
        return _res
     
    def set_error(self,error):
        self.error = error

    def set_value(self,value):
        self.value = value

    def scale_error(self,scale):
        ## Scales all errors of a given amount
        self.error = self.error*scale


    #### Common operators with other measurement
    def __add__(self, meas):
        _res = Measurement(0,0)
        if (isinstance(meas, Measurement) ):
            _res.set_value(self.value + meas.value)
            _res.set_error(sum_err_quad(self.error,meas.error))
        elif ( isinstance(meas,float) or isinstance(meas,int)):
            _res.set_value(self.value + meas)
            _res.set_error(self.error)
        return _res
 
    def __sub__(self, meas):
        if (isinstance(meas, Measurement) ):
            _res = Measurement(0,0)
            _res.set_value(self.value - meas.value)
            _res.set_error(sum_err_quad(self.error,meas.error))
        elif ( isinstance(meas,float) or isinstance(meas,int)):
            _res = Measurement(0,0)
            _res.set_value(self.value - meas)
            _res.set_error(self.error)
        return _res

    def __mul__(self,meas):
        _res = 0
        if (isinstance(meas, Measurement) ):
            _res = Measurement(0,0)
            _res.set_value(self.value * meas.value)
            _res.set_error(product_err(self.value,self.error,meas.value,meas.error))
        elif ( isinstance(meas,float) or isinstance(meas,int)):
            _res = Measurement(0,0)
            _res.set_value(self.value * meas)
            _res.set_error(self.error)
            _res.scale_error(abs(meas))            
        return _res

    def __truediv__(self,meas):
        _res = Measurement(0,0)
        if (isinstance(meas, Measurement) ):
            _res.set_value(self.value / meas.value)
            _res.set_error(division_err(self.value,self.error, meas.value, meas.error))
        elif ( isinstance(meas,float) or isinstance(meas,int)):
            _res.set_value(self.value / meas)
            _res.set_error(self.error)
            _res.scale_error(abs(1/meas))
        return _res

    def inverse(self):
        _res = Measurement(0,0)
        _res.set_value(1/self.value)
        _res.set_error(self.error/(self.value**2) )
        return _res
        
    #### Common swapped operators with other measurement
    def __radd__(self, meas):
        return self+meas

    def __rmul__(self,meas):
        return self*meas

    def __rsub__(self, meas):
        return -1.*self+meas

    def __rtruediv__(self,meas):
        return meas*self.inverse()

    
    def __pow__(self,power):
        _res = Measurement(0,0)
        if ( isinstance(power,float) ):
            _res.set_value(self.value**power)
            _res.set_error(self.error/self.value * power * _res.value)
        elif ( isinstance(power,int) ):
            _res.set_value(self.value**power)
            _res.set_error(self.error/self.value * power * _res.value)
        return _res

    ### Comparator
    def __eq__(self,meas):
        if (isinstance(meas, Measurement) ):
            if (self.value == meas.value):
                return True
            else:
                return False
        elif (type(meas) == float):
            if (self.value == meas):
                return True
            else:
                return False
            
    def __ne__(self,meas):
        if (isinstance(meas, Measurement) ):
            if (self.value == meas.value):
                return False
            else:
                return True
        elif (type(meas) == float):
            if (self.value == meas):
                return False
            else:
                return True

    def sample(self, distribution, interval=None):
        _float = 0
        if (self.error == 0):
            return self.value
        if (distribution == 'flat'):
            _float = np.random.uniform(interval[0],interval[1])
        elif (distribution == 'gaussian'):
            _float = np.random.normal(self.value,self.error)
        return _float


if __name__=='__main__':

    # these values are chosen so that
    # whatever we do we should get around
    # 10% error on the output
    test1 = Measurement(10.,1.)
    test2 = Measurement(20.,.1)
    print("References: test1  ")
    print(test1)
    print("            test2  ")
    print(test2)
    test1.scale_error(2)
    print("scale: should be 2.  ", test1.error)
    test1.scale_error(0.5)
    print("scale: should be 1.  ", test1.error)
    print(" ")
    #sum
    test3=test1+test2
    print(test3.value)
    print(test3.error)
    #mult
    test4=test1*test2
    print(test4.value)
    print(test4.error)
    #sub
    test5=test1-test2
    print(test5.value)
    print(test5.error)
    # div
    test6=test1/test2
    print(test6.value)
    print(test6.error)

    #test with numbers:
    test7=test1+20.
    print(test7.value)
    print(test7.error)

    test8=test1-20.
    print(test8.value)
    print(test8.error)

    test9=test1*20.
    print(test9.value)
    print(test9.error)

    test10=test1/20.
    print(test10.value)
    print(test10.error)

    print(" ")
    print(" Test power")
    test_pow=test1**3
    print(test_pow.value)
    print(test_pow.error)

    print(" ")
    print(" Test inverse")
    test_inv=test1.inverse()
    print(test_inv.value)
    print(test_inv.error)

    print(" ")
    print(" Test inverse")
    test_inv=test1.inverse()
    print(test_inv)
    test_inv2=1/test1
    print(test_inv2)

    print(" ")
    print(" Test inverse + product")
    test_inv=test1.inverse()*test2
    print(test_inv)
    test_inv5=test2*test1.inverse()
    print(test_inv)
    test_inv2=1/test1*test2
    print(test_inv2)
    test_inv4=1/(test1*test2)
    print(test_inv4)
    test_inv3=test2/test1
    print(test_inv3)

    
    #     bool Measurement::operator==(double b) const 
# {
#   return (m_value == b);
# }

# bool Measurement::operator!=(double b) const 
# {
#   return (m_value != b);
# }

# bool Measurement::operator>(double b) const 
# {
#   return (m_value > b);
# }

# bool Measurement::operator<(double b) const 
# {
#   return (m_value < b);
# }

# //=============================================================================
# // Mathematical functions
# //=============================================================================
# // Inverse function 
# Measurement Measurement::inverse() const
# {
#   Measurement _result(*this);
#   _result.m_value = 1/m_value;
#   for(int i=0;i<MAXERRORS; i++){
#     _result.m_errors[i] = m_errors[i]/pow(m_value,2);
#   }
#   return _result;
# }

# //=============================================================================
# // power function
# Measurement Measurement::power(double power) const
# {
#   Measurement _result(*this);
#   _result.m_value = pow(m_value,power);
#   double scale = fabs(power*_result.m_value / m_value);
#   for(int i=0;i<MAXERRORS; i++){
#     _result.m_errors[i] = m_errors[i]*scale;
#   }
#   return _result;
# }
# //=============================================================================
# // Natural logarithm 
# Measurement Measurement::ln() const
# {
#   Measurement _result(*this);
#   _result.m_value = log(m_value);
#   double scale = fabs(1/(m_value*m_value));
#   for(int i=0;i<MAXERRORS; i++){
#     _result.m_errors[i] = m_errors[i]*scale;
#   }
#   return _result;
# }




# //=============================================================================
# // Ostream operators 
# //=============================================================================
# ostream& operator<<(ostream& os, const Measurement& a)
# {
#   stringstream ss;
#   os <<   a._sstr(ss).str();
#   return os;
# }

# stringstream& Measurement::_sstr(stringstream &out) const
# {
 
#   if(m_scheme == 0){
#     out << m_value << " +- " << m_errors[0];
#   }else if(m_scheme == 1){
#     out << m_value << " +" << m_errors[0] << " -" << m_errors[1];
#   }else if(m_scheme == 2){
#     out << m_value << " +- " << m_errors[0] << " +- " << m_errors[2];
#   }else if(m_scheme == 3){
#     out << m_value << " +" << m_errors[0] << " -" << m_errors[1] 
#        << " +" << m_errors[2] << " -" << m_errors[3];  
#   }else{
#     cerr << " Something bad with error scheme..." << m_scheme << endl;
#     out << m_value << " + " << m_errors[0] << " - " << m_errors[1] 
#         << " + " << m_errors[2] << " - " << m_errors[3];  
#   }    
#   return out;
# }

# string Measurement::str() const
# {
#   stringstream ss;
#   return _sstr(ss).str();
# }


# stringstream& Measurement::_stex(stringstream &out) const
# {
#   // TODO: implement the number of significant digits
#   // - the number of the smallest exponent within the errors?  
#   if(m_scheme == 0){
#     out << "$" << m_value << " \\pm " << m_errors[0] << "$";
#   }else if(m_scheme == 1){
#     out << "$" << m_value << "^{+" << m_errors[0] << "}_{-" 
#         << m_errors[1] <<"}$";
#   }else if(m_scheme == 2){
#     out << "$" << m_value << " \\pm " << m_errors[0] 
#         << " \\pm " << m_errors[2] << "$" ;
#   }else if(m_scheme == 3){
#     out << "$" << m_value << "^{+" << m_errors[0] << "}_{-" << m_errors[1] 
#         << "}^{+" << m_errors[2] << "}_{-" << m_errors[3] << "}$";  
#   }else{
#     cerr << " Something bad with error scheme..." << m_scheme << endl;
#     out << m_value << " + " << m_errors[0] << " - " << m_errors[1] 
#         << " + " << m_errors[2] << " - " << m_errors[3];  
#   }    
#   return out;
# }

# string Measurement::tex() const
# {
#   stringstream ss;
#   return _stex(ss).str();
# }



# double Measurement::get_min_error() const 
# {
#   if(m_scheme == 0) return m_errors[0];
#   else if(m_scheme == 1) return min(m_errors[0], m_errors[1]);
#   else if(m_scheme == 2) return min(m_errors[0], m_errors[2]);
#   else if(m_scheme == 3) return min(min(m_errors[0], m_errors[1]),min(m_errors[2], m_errors[3]));
#   else return m_errors[0]; // This should not happen. 
# }


# stringstream& Measurement::_stex_sci(stringstream &out, bool print_separated) const
#  {
#    // Convert value to scientific notation for latex
#    // Get exponent and argument for the value 
#    int exp = (m_value ==0 ? 0 :  floor( log10(abs(m_value))));
#    double arg = m_value / pow(10, exp);
   
#    // Get at least one digit before the dot 
#    if(arg<1){
#      arg *= 10;
#      exp -= 1;
#    }
#    // Get the minimum of the uncertainties
#    double error = get_min_error();
#    if(!print_separated && m_scheme == 2) error = total_error();
#    int err_exp = floor(log10( abs(error)));
#    int err_residual_exp = floor(log10( abs(error) / pow(10, exp) ));

#    // Here we apply the PDG 354 rule, quoting: 
#    // If the error The basic rule states that if the three highest order digits
#    // of the error lie between 100 and 354, we round to two significant digits.
#    // If they lie between 355 and 949, we round to one significant digit.
#    // Finally, if they lie between 950 and 999, we round up to 1000
#    // and keep two significant digits.
#    // Get number of significant digits from uncertainty 

#    int pdg_corr = 0;
#    double err_residual = 100*error / pow(10, std::floor( log10(error)));
#    if(err_residual <= 354 ) pdg_corr = 1 ;
#    else if(err_residual <= 950 ) pdg_corr = 0;
#    else pdg_corr  = 0; /// But rounded upwards1
#    int prec = ( pdg_corr+ abs(err_residual_exp));
   
#    int prec_plain = (err_exp >0  ? 0 : (pdg_corr + abs(err_exp)));
   
#    // If uncertainty is larger than value, set precision to 2 just in case 
#    int old_precision = out.precision();

#    if(m_scheme == 0){
     
#      // If exponent is larger than 2 use powers of ten 
#      if(abs(exp)>2)
#      {
#        out.precision(prec);
#        out <<  fixed << "$(" <<  arg <<  " \\pm " 
#            << m_errors[0] / pow(10, exp)  << ")\\times 10^{"<< exp << "}$";
#      }else{
#        // If exponent is smaller than 2 use plain form 
#        out.precision(prec_plain);
#        out <<  fixed << "$" <<  m_value  <<  " \\pm " << m_errors[0] << "$";

#      }
#    }else if(m_scheme == 1){
#      if(abs(exp)>2){
#        out.precision(prec);
#        out <<  fixed << "$" <<  arg <<  "^{+" 
#                           << m_errors[0] / pow(10, exp) << "}_{-"  <<  m_errors[1] / pow(10, exp)  
#                           << "} \\times 10^{"<< exp << "}$" ;
#      }
#      else {
#        // If exponent is smaller than 2 use plain form 

#        out.precision(prec_plain);
#        out <<  fixed << "$" <<  m_value  
#            <<  "^{+" << m_errors[0] << "}_{-" << m_errors[1] << "}$";
#      }
#    }else if(m_scheme == 2){
#      if(print_separated){
#        if(abs(exp)>2){
# 	 out.precision(prec);
# 	 out <<  fixed << "$(" <<  arg <<  " \\pm " 
# 	     << m_errors[0] / pow(10, exp) <<  " \\pm " 
# 	     << m_errors[2] / pow(10, exp)  << ")\\times 10^{"<< exp << "}$" ;
#        }
#        else     {
# 	 // If exponent is smaller than 2 use plain form 
# 	 out.precision(prec_plain);
# 	 out <<  fixed << "$" <<  m_value  
# 	     <<  " \\pm " << m_errors[0] << " \\pm " << m_errors[2] << "$";
#        }
#      }else{
#        if(abs(exp)>2){
# 	 out.precision(prec);
# 	   out <<  fixed << "$(" <<  arg <<  " \\pm " 
# 	       << total_error() / pow(10, exp) << ")\\times 10^{"<< exp << "}$" ;
#        }
#        else     {
# 	 // If exponent is smaller than 2 use plain form w
# 	 out.precision(prec_plain);
# 	 out <<  fixed << "$" <<  m_value  
# 	     <<  " \\pm " << total_error() << "$";
	 
#        }
#      }
#    }else if(m_scheme == 3){
#      if(abs(exp)>2){
#        out.precision(prec);
#        out <<  fixed << "$" <<  arg <<  "^{+" 
# 	   << m_errors[0] / pow(10, exp) << "}_{-"  <<  m_errors[1] / pow(10, exp)  
# 	   << "}^{+" << m_errors[2] / pow(10, exp) << "}_{-"  <<  m_errors[3] / pow(10, exp)  
# 	   << "}\\times 10^{"<< exp << "}$" ;
#      }
#      else {
#        // If exponent is smaller than 2 use plain form w
#        out.precision(prec_plain);
#        out <<  fixed << "$" <<  m_value  
#            << "^{+" << m_errors[0]  << "}_{-"  <<  m_errors[1]
#            << "}^{+" << m_errors[2] << "}_{-"  <<  m_errors[3] << "}$";
#      }
     
#    }else{
#      cerr << " Something bad with error scheme..." << m_scheme << endl;
#      out.precision(old_precision); // Restore precision;
#      out << m_value << " + " << m_errors[0] << " - " << m_errors[1] 
#          << " + " << m_errors[2] << " - " << m_errors[3];  
#    }    
#    out.precision(old_precision); // Restore precision;
#    return out;
#  }


# string Measurement::stex(bool print_separated) const
# {
#   stringstream ss;
#   return _stex_sci(ss, print_separated).str();
# }



# //=============================================================================
# void Measurement::add_to_errors(double err, int to_scheme){
#   if(m_scheme & 2){// If systematic error is present add this to it (symmetrically) 
#     m_errors[2] = sqrt(pow(m_errors[2],2) + pow(err,2));
#     m_errors[3] = sqrt(pow(m_errors[3],2) + pow(err,2));
#   }else{
#     if(to_scheme==0){ // Add it to stat error 
#       m_errors[0] = sqrt(pow(m_errors[0],2) + pow(err,2));
#       m_errors[1] = sqrt(pow(m_errors[1],2) + pow(err,2));
#     }else{
#       m_errors[2] = err;
#       m_errors[3] = err;
#       m_scheme = 2;
#     }
#   }
# }
# void Measurement::add_to_errors(double first_error, double second_error, int to_scheme){
#   if(m_scheme & 2){// If systematic error is present add them to it  
#     m_errors[2] = sqrt(pow(m_errors[2],2) + pow(first_error,2));
#     m_errors[3] = sqrt(pow(m_errors[3],2) + pow(second_error,2));
#   }else{ 
#     if(to_scheme == 1){ // Add it to stat error 
#       m_errors[0] = sqrt(pow(m_errors[0],2) + pow(first_error,2));
#       m_errors[1] = sqrt(pow(m_errors[1],2) + pow(second_error,2));
#     }else{
#       m_errors[2] = sqrt(pow(m_errors[2],2) + pow(first_error,2));
#       m_errors[3] = sqrt(pow(m_errors[3],2) + pow(second_error,2));
#       m_scheme = 3;
#     }
#   }

# }


# void Measurement::add_to_syst_errors(double err){
#   add_to_errors(err,2);
# }
# void Measurement::add_to_syst_errors(double first_error, double second_error){
#   add_to_errors(first_error, second_error, 3);
# }

# //=============================================================================
# void Measurement::symmetrize(){
#   if(m_scheme & 1){ // check is indeed asymmetric
#     if(m_scheme ==3 ){ // check stat and syst are separated
#       m_errors[0] = (m_errors[0]+m_errors[1])/2; // Stat average
#       m_errors[1] = (m_errors[2]+m_errors[3])/2; // Syst average
#       m_scheme = 2;
#     }else{
#       m_errors[0] = (m_errors[0]+m_errors[1])/2;
#       m_scheme = 1;
#     }
#   }
#   return;
# }

# //=============================================================================
# void Measurement::compress(){
#   if(m_scheme & 2){ // check it was splitted in the first place
#     if(m_scheme & 1){ // check we have asymmetric errors
#       m_errors[0] = sqrt(pow(m_errors[0],2)+pow(m_errors[2],2)); // Upper stat+syst
#       m_errors[1] = sqrt(pow(m_errors[1],2)+pow(m_errors[3],2)); // Lower stat+syst
#       m_scheme = 1;
#     }else{
#       m_errors[0] = sqrt(pow(m_errors[0],2)+pow(m_errors[1],2)); 
#       m_scheme = 0;
#     }
#   }
#   return;
# }


# int Measurement::common_scheme(Measurement a){
#   // Function to evaluate the common scheme between the element
#   // and a second element 
#   // -  -  -  -  - 
#   // Finds the more general one between the two
#   return (this->m_scheme | a.m_scheme) ; // bit OR to get common scheme 
# }

# void Measurement::debug(){
#   // Debug function 
#   // Prints all the member variables of the element 
#   cout << "Meas debug  " << m_value << " + " <<  m_errors[0] 
#        << " - " << m_errors[1] 
#        << " + " << m_errors[2] 
#        << " - " << m_errors[3] 
#        << " Scheme " << m_scheme << endl;  

# }

# //=============================================================================
# // Sum errors in quadrature 

# void Measurement::sum_err_quad(Measurement a, Measurement b, Measurement &result) const 
# {
#   for(int i=0;i<MAXERRORS; i++){
#     result.m_errors[i] = sqrt(pow(a.m_errors[i],2)+pow(b.m_errors[i],2));
#     result.m_scheme = a.common_scheme(b);
#   }
# }

# //=============================================================================
# // Propagation of errors "log" formula 
# void Measurement::sum_err_log(Measurement a, Measurement b, Measurement &result)const
# {
#   for(int i=0;i<MAXERRORS; i++){
#     result.m_errors[i] = result.m_value * sqrt(pow(a.m_errors[i] / a.m_value ,2)
# 					       +pow(b.m_errors[i] / b.m_value ,2));
#     result.m_scheme = a.common_scheme(b);
#   }
# }

# void Measurement::product_err(Measurement a, Measurement b, Measurement &result)const
# {
#   for(int i=0;i<MAXERRORS; i++){
#     result.m_errors[i] = sqrt(pow(a.m_errors[i] * b.m_value ,2)
#                               +pow(b.m_errors[i] * a.m_value ,2));
#     result.m_scheme = a.common_scheme(b);
#   }
# }

# void Measurement::division_err(Measurement a, Measurement b, Measurement &result)const
# {
#   for(int i=0;i<MAXERRORS; i++){
#     result.m_errors[i] = sqrt(pow(a.m_errors[i] / b.m_value ,2)
#                               +pow(b.m_errors[i] * a.m_value/pow(b.m_value,2) ,2));
#     result.m_scheme = a.common_scheme(b);
#   }
# }




# void Measurement::scale_error(double scale){
#   // Scales all errors of a given amount
#   for(int i=0;i<MAXERRORS; i++){
#     m_errors[i] *= scale;
#   }
# }


# double Measurement::total_error() const 
# {
#   // Calculates total error without storing the result 
#   if(m_scheme ==0){ 
#     return m_errors[0];
#   }else if(m_scheme == 1){
#     // If asymmetric return the average
#     return (m_errors[0] + m_errors[1]) / 2; 
#   }else if(m_scheme == 2){
#     // Sum in quadrature stat and syst 
#     return sqrt(pow(m_errors[0],2) + pow(m_errors[2],2));
#   }else {
#     return   sqrt( pow ( (m_errors[0] + m_errors[1]) / 2, 2)+
# 		   pow ( (m_errors[2] + m_errors[3]) / 2, 2));
		   
#   }
  
# }


# const char* Measurement::getLaTeX() const {
#   return tex().c_str();
# }
# ///> function to print latex number, like: a^{+x}_{-y} or a\pm b
# /*char* Measurement::getLaTeX(){
#   char* latex = new char[100];
#   if(m_scheme == 0){
#       sprintf(latex,"$%.3e \\pm %.3e$",m_value,m_errors[0]);
#     }else if(m_scheme == 1){
#       sprintf(latex,"$%.3e^{+%.3e}_{-%.3e}$",m_value,m_errors[0],m_errors[1]);
#     }else if(m_scheme == 2){
#       sprintf(latex,"$%.3e \\pm %.3e (\\text{stat}) \\pm %.3e (\\text{sys})$",m_value,m_errors[0],m_errors[1]);
#     }else if(m_scheme == 3){
#       sprintf(latex,"$%.3e^{%.3e}_{%.3e} (\\text{stat})^{%.3e}_{%.3e} (\\text{sys})$",
#               m_value,m_errors[0],m_errors[1],m_errors[2],m_errors[3]);
#     }else{
#       cerr << " Something bad with error scheme..." << m_scheme << endl;
#     }    
#     return latex;
#     };*/


# Measurement Measurement::average(Measurement b) const
# {
#   // Weighted average between two elements
#   // Only total errors are considered here 
#   // The returned average has only one symmetric total error
#   double wa = 1/pow(total_error(),2);
#   double wb = 1/pow(b.total_error(),2);
#   double sumw = wa + wb;
#   double val = (m_value*wa + b.value()*wb)/sumw;
#   double err = sqrt(1/sumw);
  
#   return Measurement(val, err);
# }
