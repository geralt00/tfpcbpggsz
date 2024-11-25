#include "Resonance.hh"
#include "DalitzPoint.hh"
#include <math.h>
#include <complex>
#include <stdlib.h>

#define PI 3.14159265


// generic resonance contructor
Resonance::Resonance(std::string RName, double RAmp, double RPhase, double RMass,
		double RWidth, int RSpin, double RRadii, double DRadii, std::string axis) {
	_name = RName;
	_amp  = RAmp;
	_phase  = RPhase;
	_massR = RMass;
	_gammaR = RWidth;
	_spinR = RSpin;
	_radiusR = RRadii;
	_radiusD = DRadii;
	_axislabel = axis;

	// Create proper orientation for a,b,c
	setAxis(axis);

	_xmin = pow(_ma + _mb,2);
	_xmax = pow(mD0 - _mc,2);



} // end constructor()

// K-matrix constructor
Resonance::Resonance( std::string RName,
		std::string axis,
		double Kmatrix_beta1_Amplitude,
		double Kmatrix_beta1_Phase,
		double Kmatrix_beta2_Amplitude,
		double Kmatrix_beta2_Phase,
		double Kmatrix_beta3_Amplitude,
		double Kmatrix_beta3_Phase,
		double Kmatrix_beta4_Amplitude,
		double Kmatrix_beta4_Phase,
		double Kmatrix_beta5_Amplitude,
		double Kmatrix_beta5_Phase,
		double Kmatrix_f_prod_11_Amplitude,
		double Kmatrix_f_prod_11_Phase,
		double Kmatrix_f_prod_12_Amplitude,
		double Kmatrix_f_prod_12_Phase,
		double Kmatrix_f_prod_13_Amplitude,
		double Kmatrix_f_prod_13_Phase,
		double Kmatrix_f_prod_14_Amplitude,
		double Kmatrix_f_prod_14_Phase,
		double Kmatrix_f_prod_15_Amplitude,
		double Kmatrix_f_prod_15_Phase,
		double Kmatrix_s_prod_0 ) {
	_name = RName;
	_amp  = -999.;
	_phase  = -999.;
	_massR = -999.;
	_gammaR = -999.;
	_spinR = -999.;
	_radiusR = -999.;
	_radiusD = -999.;
	_axislabel = axis;

	// K matrix parameters
	_Kmatrix_beta1_Amplitude = Kmatrix_beta1_Amplitude;
	_Kmatrix_beta1_Phase = Kmatrix_beta1_Phase;
	_Kmatrix_beta2_Amplitude = Kmatrix_beta2_Amplitude;
	_Kmatrix_beta2_Phase = Kmatrix_beta2_Phase;
	_Kmatrix_beta3_Amplitude = Kmatrix_beta3_Amplitude;
	_Kmatrix_beta3_Phase = Kmatrix_beta3_Phase;
	_Kmatrix_beta4_Amplitude = Kmatrix_beta4_Amplitude;
	_Kmatrix_beta4_Phase = Kmatrix_beta4_Phase;
	_Kmatrix_beta5_Amplitude = Kmatrix_beta5_Amplitude;
	_Kmatrix_beta5_Phase = Kmatrix_beta5_Phase;
	_Kmatrix_f_prod_11_Amplitude = Kmatrix_f_prod_11_Amplitude;
	_Kmatrix_f_prod_11_Phase = Kmatrix_f_prod_11_Phase;
	_Kmatrix_f_prod_12_Amplitude = Kmatrix_f_prod_12_Amplitude;
	_Kmatrix_f_prod_12_Phase = Kmatrix_f_prod_12_Phase;
	_Kmatrix_f_prod_13_Amplitude = Kmatrix_f_prod_13_Amplitude;
	_Kmatrix_f_prod_13_Phase = Kmatrix_f_prod_13_Phase;
	_Kmatrix_f_prod_14_Amplitude = Kmatrix_f_prod_14_Amplitude;
	_Kmatrix_f_prod_14_Phase = Kmatrix_f_prod_14_Phase;
	_Kmatrix_f_prod_15_Amplitude = Kmatrix_f_prod_15_Amplitude;
	_Kmatrix_f_prod_15_Phase = Kmatrix_f_prod_15_Phase;

	_Kmatrix_s_prod_0 = Kmatrix_s_prod_0;

	// Create proper orientation for a,b,c
	setAxis(axis);

	// _ma,_mb,_mc have all been permutated according to "axis"

	_xmin = pow(_ma + _mb,2);
	_xmax = pow(mD0 - _mc,2);

	

}

// LASS constructor
Resonance::Resonance(std::string RName, double RAmp, double RPhase, double RMass,
		double RWidth, int RSpin, double RRadii, double DRadii, std::string axis,
		double LASS_F,
		double LASS_phi_F,
		double LASS_R,
		double LASS_phi_R,
		double LASS_a,
		double LASS_r) {

	_name = RName;
	_amp  = RAmp;
	_phase  = RPhase;
	_massR = RMass;
	_gammaR = RWidth;
	_spinR = RSpin;
	_radiusR = RRadii;
	_radiusD = DRadii;
	_axislabel = axis;

	// LASS parameters
	_LASS_F = LASS_F;
	_LASS_phi_F = LASS_phi_F;
	_LASS_R = LASS_R;
	_LASS_phi_R = LASS_phi_R;
	_LASS_a = LASS_a;
	_LASS_r = LASS_r;

	// Create proper orientation for a,b,c
	setAxis(axis);

	// _ma,_mb,_mc have all been permutated according to "axis"

	_xmin = pow(_ma + _mb,2);
	_xmax = pow(mD0 - _mc,2);


}

///////////////////////////////////////////////////////////////////
void Resonance::setAxis(std::string axisLabel) {

	// Initialize for troubleshooting
	_axis = 0;

	// Set internal value of axis
	if (axisLabel == "X") {
		_axis = 1;
	} else if (axisLabel == "Y") {
		_axis = 2;
	} else if (axisLabel == "Z") {
		_axis = 3;
	}

	// Set internal value of axis
	if (axisLabel == "-X") {
		_axis = -1;
	} else if (axisLabel == "-Y") {
		_axis = -2;
	} else if (axisLabel == "-Z") {
		_axis = -3;
	}

	// Set internal orientation of final state particles
	switch (abs(_axis)) {

	case 1: {
		_ma = mKs;
		_mb = mPi;
		_mc = mPi;
	} break;

	case 2: {
		_ma = mPi;
		_mb = mPi;
		_mc = mKs;
	} break;

	case 3: {
		_ma = mKs;
		_mb = mPi;
		_mc = mPi;
	} break;

	default: {
		_ma = 0;
		_mb = 0;
		_mc = 0;

		cout << "\n!!!! Bad axis! Select X Y or Z\n\n";
	}
	}
}


////////////////////////////////////////////////////////////////
std::complex<double> Resonance::contribution(const DalitzPoint& point) const {


  // check if in Dalitz plot
  bool flag_inDalitz=false;
  if (point.isValid() == true) {
    flag_inDalitz=true;
  }
  
  const double x = point.q("0p"); // x[0]; // is either msquared01 for D0 or msquared02 for D0bar
  const double z = point.q("0n"); // x[1]; // is either msquared02 for D0 or msquared01 for D0bar
  const double y = mD0 * mD0 + mKs * mKs + 2 * mPi * mPi - x - z; // compute msquared12 to be used in amplitude calculation below

  // Variables (v1,v2) are used for PDF(x,y)
  double v1, v2, v3;

  // Determine orientation of (v1,v2) <- (x,y,z)
  if (abs(_axis) == 1) {
    v1 = x;
    v2 = y;
    v3 = z;
  } else if (abs(_axis) == 2) {
    v1 = y;
    v2 = x;
    v3 = z;
  } else if (abs(_axis) == 3) {
    v1 = z;
    v2 = y;
    v3 = x;
  } else {
    v1 = v2 = v3 = 0;
    cout << "No axis defined in contribution!!!\n";
  }

  // Variable to store the complex number for the resonance amplitude/composite matrix element
  std::complex<double> matrixEl;

  const double degToRad = PI/180.0;
  const double rads = _phase*degToRad;
  const std::complex<double> decay_amp(_amp*cos(rads),_amp*sin(rads));

  // Compute the resonances amplitude/composite matrix element
  if (flag_inDalitz==true) {
    if (_name=="KMatrix") {
      matrixEl = amplitude_KMatrix( v1 ); // Final K-matrix implementation.
    } else if (_name=="BW") {
      matrixEl = amplitude_BW( v1, v2, v3 ); // Final Breit-Wigner model for the isobars.
    } else if (_name=="LASS") {
      matrixEl = amplitude_LASS( v1 ); // Final LASS implementation.
    } else {
      std::cout << "Resonance warning: _name == " << _name << " not found in Resonance.cc" << std::endl;
    }
  } else {
    if ( _verbose > 0) {cout << "Point not in Dalitz plot phase space " << endl;}

    return std::complex<double>(0., 0.);
  }

  // Return the resonance amplitude/composite matrix element.
  return matrixEl;
}

std::complex<double> Resonance::amplitude_BW(double v1, double v2, double v3) const {

	std::complex<double> matrixEl;

	const double pi180inv = PI / 180.;

	const double ampl=_amp;
	const double theta=_phase;

	const double gamma = _gammaR;
	const double bwm = _massR;
	const int spin = _spinR;

	const double R_r = 1.5; // "resonance radius" for Blatt-Weisskoper barrier factors.
	const double R_D = 5.0; // "D meson radius" for Blatt-Weisskoper barrier factors.

	const double mAB=sqrt(v1); // = (_p4_d1+_p4_d2).mass();
	const double mBC=sqrt(v2); // = (_p4_d2+p4_d3).mass();
	const double mAC=sqrt(v3); // = (_p4_d1+p4_d3).mass();

	const double mA=_ma; // = _p4_d1.mass();
	const double mB=_mb; // = _p4_d2.mass();
	const double mC=_mc; // = _p4_d3.mass();
	const double mD=mD0; // = _p4_p.mass();

	const double mR=bwm;
	const double gammaR=gamma;

	const double pAB=sqrt((((mAB*mAB-mA*mA-mB*mB)*(mAB*mAB-mA*mA-mB*mB)/4.0) - mA*mA*mB*mB)/(mAB*mAB));
	const double pR=sqrt((((mR*mR-mA*mA-mB*mB)*(mR*mR-mA*mA-mB*mB)/4.0) - mA*mA*mB*mB)/(mR*mR));

        double pD=(((mD*mD-mR*mR-mC*mC)*(mD*mD-mR*mR-mC*mC)/4.0) - mR*mR*mC*mC)/(mD*mD);
	if ( pD>0 ) { pD=sqrt(pD); } else {pD=0;}
	const double pDAB=sqrt( (((mD*mD-mAB*mAB-mC*mC)*(mD*mD-mAB*mAB-mC*mC)/4.0) - mAB*mAB*mC*mC)/(mD*mD));

	// Compute Blatt-Weisskopf barrier factors.
	double fR=1;
	double fD=1;
	double num;
	int power;

	switch (spin) {
	case 0:
		fR=1.0;
		fD=1.0;
		power=1;
		break;
	case 1:
		fR=sqrt(1.0+R_r*R_r*pR*pR)/sqrt(1.0+R_r*R_r*pAB*pAB);
		fD=sqrt(1.0+R_D*R_D*pD*pD)/sqrt(1.0+R_D*R_D*pDAB*pDAB);
		power=3;
		break;
	case 2:
		fR=sqrt(9.0+3.0*R_r*R_r*pR*pR+R_r*R_r*pR*pR*R_r*R_r*pR*pR)/sqrt(9.0+3.0*R_r*R_r*pAB*pAB+R_r*R_r*pAB*pAB*R_r*R_r*pAB*pAB);
		fD=sqrt(9.0+3.0*R_D*R_D*pD*pD+R_D*R_D*pD*pD*R_D*R_D*pD*pD)/sqrt(9.0+3.0*R_D*R_D*pDAB*pDAB+R_D*R_D*pDAB*pDAB*R_D*R_D*pDAB*pDAB);
		power=5;
		break;
	default:
		std::cout << "Incorrect spin in Resonance.cc" << std::endl;
	}

	// Compute the running width.
	const double gammaAB= gammaR*pow(pAB/pR,power)*(mR/mAB)*fR*fR;

	// Compute the amplitude.
	switch (spin) {
	case 0:
		matrixEl = ampl*std::complex<double>(cos(theta*pi180inv),sin(theta*pi180inv))*fR*fD/(mR*mR-mAB*mAB-std::complex<double>(0.0,mR*gammaAB));
		break;
	case 1:
		matrixEl = ampl*std::complex<double>(cos(theta*pi180inv),sin(theta*pi180inv))*(fR*fD*(mAC*mAC-mBC*mBC+((mD*mD-mC*mC)*(mB*mB-mA*mA)/(mAB*mAB)))/(mR*mR-mAB*mAB-std::complex<double>(0.0,mR*gammaAB)));
		break;
	case 2:
		// corrected denominator for angular term [PRD 83:052001(2011)]
		num = pow(mBC*mBC-mAC*mAC+(mD*mD-mC*mC)*(mA*mA-mB*mB)/(mAB*mAB),2)-
		1./3.*(mAB*mAB-2.*(mD*mD+mC*mC)+pow(mD*mD-mC*mC,2)/(mAB*mAB))*
		(mAB*mAB-2.*(mA*mA+mB*mB)+pow(mA*mA-mB*mB,2)/(mAB*mAB));
		matrixEl = ampl*std::complex<double>(cos(theta*pi180inv),sin(theta*pi180inv))*(fR*fD*num/(mR*mR-mAB*mAB-std::complex<double>(0.0,mR*gammaAB)));
		break;
	default:
		std::cout << "Incorrect spin in Resonance.cc" << std::endl;
	}

	return matrixEl;
}

std::complex<double> Resonance::amplitude_LASS(double mab2) const {

	const double pi180inv = PI / 180.;

	const double ampl=_amp;
	const double theta=_phase;

	const double s = mab2;

	const double _a = _LASS_a;
	const double _r = _LASS_r;
	const double _R = _LASS_R; // Amplitude magnitude of the resonant term
	const double _phiR = _LASS_phi_R; // Phase of the resonant term
	const double _F = _LASS_F; // Amplitude magnitude of the non-resonant term
	const double _phiF = _LASS_phi_F; // Phase of the non-resonant term

	const double mR=_massR;
	const double gammaR=_gammaR;

	const double fR=1.0; // K*0(1430) has spin zero
	const int power=1; // Power is 1 for spin zero

	const double mAB=sqrt(mab2); // (_p4_d1+_p4_d2).mass();

	const double mA=_ma; // _p4_d1.mass();
	const double mB=_mb; // _p4_d2.mass();

	const double pAB=sqrt( (((mAB*mAB-mA*mA-mB*mB)*(mAB*mAB-mA*mA-mB*mB)/4.0) - mA*mA*mB*mB)/(mAB*mAB));
	const double q=pAB;

	const double pR=sqrt( (((mR*mR-mA*mA-mB*mB)*(mR*mR-mA*mA-mB*mB)/4.0) - mA*mA*mB*mB)/(mR*mR));

	// Running width.
	const double g = gammaR*pow(pAB/pR,power)*(mR/mAB)*fR*fR;

	const std::complex<double> propagator_relativistic_BreitWigner = 1./(mR*mR - mAB*mAB - std::complex<double>(0.,mR*g));

	// Non-resonant phase shift
	const double cot_deltaF = 1.0/(_a*q) + 0.5*_r*q;
	const double qcot_deltaF = 1.0/_a + 0.5*_r*q*q;

	// Compute resonant part
	const std::complex<double> expi2deltaF = std::complex<double>(qcot_deltaF, q)/ std::complex<double>(qcot_deltaF, -q);

	const std::complex<double> resonant_term_T = _R * std::complex<double>(cos(_phiR + 2 * _phiF), sin(_phiR + 2 * _phiF)) * propagator_relativistic_BreitWigner * mR * gammaR * mR / pR * expi2deltaF;

	// Compute non-resonant part
	const std::complex<double> non_resonant_term_F = _F * std::complex<double>(cos(_phiF), sin(_phiF)) * (cos(_phiF) + cot_deltaF * sin(_phiF)) * sqrt(s) / std::complex<double>(qcot_deltaF, -q);

	// Add non-resonant and resonant terms
	const std::complex<double> LASS_contribution = non_resonant_term_F + resonant_term_T;

	std::complex<double> matrixEl = ampl * std::complex<double>(cos(theta*pi180inv), sin(theta*pi180inv)) * LASS_contribution;

	return matrixEl;
}

std::complex<double> Resonance::amplitude_KMatrix(double mab2) const {
	const double s = mab2;

	// Define the complex coupling constants
	double g[5][5]; // g[Physical pole]Decay channel]

	// pi+pi- channel
	g[0][0]=0.22889;
	g[1][0]=0.94128;
	g[2][0]=0.36856;
	g[3][0]=0.33650;
	g[4][0]=0.18171;

	// K+K- channel
	g[0][1]=-0.55377;
	g[1][1]=0.55095;
	g[2][1]=0.23888;
	g[3][1]=0.40907;
	g[4][1]=-0.17558;

	// 4pi channel
	g[0][2]=0;
	g[1][2]=0;
	g[2][2]=0.55639;
	g[3][2]=0.85679;
	g[4][2]=-0.79658;

	// eta eta channel
	g[0][3]=-0.39899;
	g[1][3]=0.39065;
	g[2][3]=0.18340;
	g[3][3]=0.19906;
	g[4][3]=-0.00355;

	//eta eta' channel
	g[0][4]=-0.34639;
	g[1][4]=0.31503;
	g[2][4]=0.18681;
	g[3][4]=-0.00984;
	g[4][4]=0.22358;

	// Define masses of the physical poles (in GeV)
        double ma[5];

	ma[0]=0.651;
	ma[1]=1.20360;
	ma[2]=1.55817;
	ma[3]=1.21000;
	ma[4]=1.82206;

	// Define variables
	std::complex<double> n11,n12,n13,n14,n15,n21,n22,n23,n24,n25,n31,n32,n33,n34,n35,n41,n42,n43,n44,n45,n51,n52,n53,n54,n55;
	double rho1sq,rho2sq,rho4sq,rho5sq;
	std::complex<double> rho1,rho2,rho3,rho4,rho5;
	std::complex<double> rho[5];
	std::complex<double> pole,SVT,Adler;
	std::complex<double> det;
	std::complex<double> i[5][5];
	double f[5][5];

	// pi+, K+, eta, and eta' PDG masses
	const double mpi=0.13957;
	const double mK=0.493677;
	const double meta=0.54775;
	const double metap=0.95778;

	// Init matrices and vectors with zeros
	std::complex<double> K[5][5];
	for(int k=0;k<5;k++) {
		for(int l=0;l<5;l++) {
			i[k][l]=std::complex<double>(0,0);
			K[k][l]=std::complex<double>(0,0);
			f[k][l]=0;
		}
		rho[k]=0;
	}

	// Fill scattering data values
	const double s_scatt=-3.92637;
	const double sa=1.0;
	const double sa_0=-0.15;

	// f_scattering
	f[0][0]=0.23399;
	f[0][1]=0.15044;
	f[0][2]=-0.20545;
	f[0][3]=0.32825;
	f[0][4]=0.35412;

	f[1][0]=f[0][1];
	f[2][0]=f[0][2];
	f[3][0]=f[0][3];
	f[4][0]=f[0][4];

	// Compute phase space factors
	rho1sq=(1.0-(pow((mpi+mpi),2)/s));
	if(rho1sq >=0) {
		rho1=std::complex<double>(sqrt(rho1sq),0);
	}
	else{
		rho1=std::complex<double>(0,sqrt(-rho1sq));
	}
	rho[0]=rho1;

	rho2sq=(1.0-(pow((mK+mK),2)/s));
	if(rho2sq >=0) {
		rho2=std::complex<double>(sqrt(rho2sq),0);
	}
	else{
		rho2=std::complex<double>(0,sqrt(-rho2sq));
	}

	rho[1]=rho2;

	rho3=std::complex<double>(0,0);

	if(s<=1) {
		const double real = 1.2274+0.00370909/(s*s) - (0.111203)/(s) - 6.39017*s +16.8358*s*s - 21.8845*s*s*s + 11.3153*s*s*s*s;
		const double cont32=sqrt(1.0-(16.0*mpi*mpi));
		rho3=std::complex<double>(cont32*real,0);
	}
	else{
		rho3=std::complex<double>(sqrt(1.0-(16.0*mpi*mpi/s)),0);
	}
	rho[2]=rho3;

	rho4sq=(1.0-(pow((meta+meta),2)/s));
	if(rho4sq>=0) {
		rho4=std::complex<double>(sqrt(rho4sq),0);
	}
	else{
		rho4=std::complex<double>(0,sqrt(-rho4sq));
	}
	rho[3]=rho4;

	rho5sq=(1.0-(pow((meta+metap),2)/s));
	if(rho5sq >=0) {
		rho5=std::complex<double>(sqrt(rho5sq),0);
	}
	else{
		rho5=std::complex<double>(0,sqrt(-rho5sq));
	}
	rho[4]=rho5;

	// Sum over the poles
	for(int k=0;k<5;k++) {
		for(int l=0;l<5;l++) {
			for (int pole_index=0;pole_index<5;pole_index++) {
				const double A=g[pole_index][k]*g[pole_index][l];
				const double B=ma[pole_index]*ma[pole_index]-s;
				K[k][l]=K[k][l]+std::complex<double>(A/B,0);
			}
		}
	}

	for(int k=0;k<5;k++) {
		for(int l=0;l<5;l++) {
			const double C=f[k][l]*(1.0-s_scatt);
			const double D=(s-s_scatt);
			K[k][l]=K[k][l]+std::complex<double>(C/D,0);
		}
	}

	for(int k=0;k<5;k++) {
		for(int l=0;l<5;l++) {
			const double E=(s-(sa*mpi*mpi*0.5))*(1.0-sa_0);
			const double F=(s-sa_0);
			K[k][l]=K[k][l]*std::complex<double>(E/F,0);
		}
	}

	n11=std::complex<double>(1,0)-std::complex<double>(0,1)*K[0][0]*rho[0];
	n12=std::complex<double>(0,0)-std::complex<double>(0,1)*K[0][1]*rho[1];
	n13=std::complex<double>(0,0)-std::complex<double>(0,1)*K[0][2]*rho[2];
	n14=std::complex<double>(0,0)-std::complex<double>(0,1)*K[0][3]*rho[3];
	n15=std::complex<double>(0,0)-std::complex<double>(0,1)*K[0][4]*rho[4];

	n21=std::complex<double>(0,0)-std::complex<double>(0,1)*K[1][0]*rho[0];
	n22=std::complex<double>(1,0)-std::complex<double>(0,1)*K[1][1]*rho[1];
	n23=std::complex<double>(0,0)-std::complex<double>(0,1)*K[1][2]*rho[2];
	n24=std::complex<double>(0,0)-std::complex<double>(0,1)*K[1][3]*rho[3];
	n25=std::complex<double>(0,0)-std::complex<double>(0,1)*K[1][4]*rho[4];

	n31=std::complex<double>(0,0)-std::complex<double>(0,1)*K[2][0]*rho[0];
	n32=std::complex<double>(0,0)-std::complex<double>(0,1)*K[2][1]*rho[1];
	n33=std::complex<double>(1,0)-std::complex<double>(0,1)*K[2][2]*rho[2];
	n34=std::complex<double>(0,0)-std::complex<double>(0,1)*K[2][3]*rho[3];
	n35=std::complex<double>(0,0)-std::complex<double>(0,1)*K[2][4]*rho[4];

	n41=std::complex<double>(0,0)-std::complex<double>(0,1)*K[3][0]*rho[0];
	n42=std::complex<double>(0,0)-std::complex<double>(0,1)*K[3][1]*rho[1];
	n43=std::complex<double>(0,0)-std::complex<double>(0,1)*K[3][2]*rho[2];
	n44=std::complex<double>(1,0)-std::complex<double>(0,1)*K[3][3]*rho[3];
	n45=std::complex<double>(0,0)-std::complex<double>(0,1)*K[3][4]*rho[4];

	n51=std::complex<double>(0,0)-std::complex<double>(0,1)*K[4][0]*rho[0];
	n52=std::complex<double>(0,0)-std::complex<double>(0,1)*K[4][1]*rho[1];
	n53=std::complex<double>(0,0)-std::complex<double>(0,1)*K[4][2]*rho[2];
	n54=std::complex<double>(0,0)-std::complex<double>(0,1)*K[4][3]*rho[3];
	n55=std::complex<double>(1,0)-std::complex<double>(0,1)*K[4][4]*rho[4];

	// Compute the determinant
	det = (n15*n24*n33*n42*n51 - n14*n25*n33*n42*n51 - n15*n23*n34*n42*n51 +
			n13*n25*n34*n42*n51 + n14*n23*n35*n42*n51 - n13*n24*n35*n42*n51 -
			n15*n24*n32*n43*n51 + n14*n25*n32*n43*n51 + n15*n22*n34*n43*n51 -
			n12*n25*n34*n43*n51 - n14*n22*n35*n43*n51 + n12*n24*n35*n43*n51 +
			n15*n23*n32*n44*n51 - n13*n25*n32*n44*n51 - n15*n22*n33*n44*n51 +
			n12*n25*n33*n44*n51 + n13*n22*n35*n44*n51 - n12*n23*n35*n44*n51 -
			n14*n23*n32*n45*n51 + n13*n24*n32*n45*n51 + n14*n22*n33*n45*n51 -
			n12*n24*n33*n45*n51 - n13*n22*n34*n45*n51 + n12*n23*n34*n45*n51 -
			n15*n24*n33*n41*n52 + n14*n25*n33*n41*n52 + n15*n23*n34*n41*n52 -
			n13*n25*n34*n41*n52 - n14*n23*n35*n41*n52 + n13*n24*n35*n41*n52 +
			n15*n24*n31*n43*n52 - n14*n25*n31*n43*n52 - n15*n21*n34*n43*n52 +
			n11*n25*n34*n43*n52 + n14*n21*n35*n43*n52 - n11*n24*n35*n43*n52 -
			n15*n23*n31*n44*n52 + n13*n25*n31*n44*n52 + n15*n21*n33*n44*n52 -
			n11*n25*n33*n44*n52 - n13*n21*n35*n44*n52 + n11*n23*n35*n44*n52 +
			n14*n23*n31*n45*n52 - n13*n24*n31*n45*n52 - n14*n21*n33*n45*n52 +
			n11*n24*n33*n45*n52 + n13*n21*n34*n45*n52 - n11*n23*n34*n45*n52 +
			n15*n24*n32*n41*n53 - n14*n25*n32*n41*n53 - n15*n22*n34*n41*n53 +
			n12*n25*n34*n41*n53 + n14*n22*n35*n41*n53 - n12*n24*n35*n41*n53 -
			n15*n24*n31*n42*n53 + n14*n25*n31*n42*n53 + n15*n21*n34*n42*n53 -
			n11*n25*n34*n42*n53 - n14*n21*n35*n42*n53 + n11*n24*n35*n42*n53 +
			n15*n22*n31*n44*n53 - n12*n25*n31*n44*n53 - n15*n21*n32*n44*n53 +
			n11*n25*n32*n44*n53 + n12*n21*n35*n44*n53 - n11*n22*n35*n44*n53 -
			n14*n22*n31*n45*n53 + n12*n24*n31*n45*n53 + n14*n21*n32*n45*n53 -
			n11*n24*n32*n45*n53 - n12*n21*n34*n45*n53 + n11*n22*n34*n45*n53 -
			n15*n23*n32*n41*n54 + n13*n25*n32*n41*n54 + n15*n22*n33*n41*n54 -
			n12*n25*n33*n41*n54 - n13*n22*n35*n41*n54 + n12*n23*n35*n41*n54 +
			n15*n23*n31*n42*n54 - n13*n25*n31*n42*n54 - n15*n21*n33*n42*n54 +
			n11*n25*n33*n42*n54 + n13*n21*n35*n42*n54 - n11*n23*n35*n42*n54 -
			n15*n22*n31*n43*n54 + n12*n25*n31*n43*n54 + n15*n21*n32*n43*n54 -
			n11*n25*n32*n43*n54 - n12*n21*n35*n43*n54 + n11*n22*n35*n43*n54 +
			n13*n22*n31*n45*n54 - n12*n23*n31*n45*n54 - n13*n21*n32*n45*n54 +
			n11*n23*n32*n45*n54 + n12*n21*n33*n45*n54 - n11*n22*n33*n45*n54 +
			n14*n23*n32*n41*n55 - n13*n24*n32*n41*n55 - n14*n22*n33*n41*n55 +
			n12*n24*n33*n41*n55 + n13*n22*n34*n41*n55 - n12*n23*n34*n41*n55 -
			n14*n23*n31*n42*n55 + n13*n24*n31*n42*n55 + n14*n21*n33*n42*n55 -
			n11*n24*n33*n42*n55 - n13*n21*n34*n42*n55 + n11*n23*n34*n42*n55 +
			n14*n22*n31*n43*n55 - n12*n24*n31*n43*n55 - n14*n21*n32*n43*n55 +
			n11*n24*n32*n43*n55 + n12*n21*n34*n43*n55 - n11*n22*n34*n43*n55 -
			n13*n22*n31*n44*n55 + n12*n23*n31*n44*n55 + n13*n21*n32*n44*n55 -
			n11*n23*n32*n44*n55 - n12*n21*n33*n44*n55 + n11*n22*n33*n44*n55);

	// The 1st row of the inverse matrix {(I-iKp)^-1}_0j
	i[0][0] = (n25*n34*n43*n52 -
			n24*n35*n43*n52 - n25*n33*n44*n52 + n23*n35*n44*n52 +
			n24*n33*n45*n52 - n23*n34*n45*n52 - n25*n34*n42*n53 +
			n24*n35*n42*n53 + n25*n32*n44*n53 - n22*n35*n44*n53 -
			n24*n32*n45*n53 + n22*n34*n45*n53 + n25*n33*n42*n54 -
			n23*n35*n42*n54 - n25*n32*n43*n54 + n22*n35*n43*n54 +
			n23*n32*n45*n54 - n22*n33*n45*n54 - n24*n33*n42*n55 +
			n23*n34*n42*n55 + n24*n32*n43*n55 - n22*n34*n43*n55 -
			n23*n32*n44*n55 + n22*n33*n44*n55)/det;

	i[0][1] = (-n15*n34*n43*n52 +
			n14*n35*n43*n52 + n15*n33*n44*n52 - n13*n35*n44*n52 -
			n14*n33*n45*n52 + n13*n34*n45*n52 + n15*n34*n42*n53 -
			n14*n35*n42*n53 - n15*n32*n44*n53 + n12*n35*n44*n53 +
			n14*n32*n45*n53 - n12*n34*n45*n53 - n15*n33*n42*n54 +
			n13*n35*n42*n54 + n15*n32*n43*n54 - n12*n35*n43*n54 -
			n13*n32*n45*n54 + n12*n33*n45*n54 + n14*n33*n42*n55 -
			n13*n34*n42*n55 - n14*n32*n43*n55 + n12*n34*n43*n55 +
			n13*n32*n44*n55 - n12*n33*n44*n55)/det;

	i[0][2] = (n15*n24*n43*n52 -
			n14*n25*n43*n52 - n15*n23*n44*n52 + n13*n25*n44*n52 +
			n14*n23*n45*n52 - n13*n24*n45*n52 - n15*n24*n42*n53 +
			n14*n25*n42*n53 + n15*n22*n44*n53 - n12*n25*n44*n53 -
			n14*n22*n45*n53 + n12*n24*n45*n53 + n15*n23*n42*n54 -
			n13*n25*n42*n54 - n15*n22*n43*n54 + n12*n25*n43*n54 +
			n13*n22*n45*n54 - n12*n23*n45*n54 - n14*n23*n42*n55 +
			n13*n24*n42*n55 + n14*n22*n43*n55 - n12*n24*n43*n55 -
			n13*n22*n44*n55 + n12*n23*n44*n55)/det;

	i[0][3] = (-n15*n24*n33*n52 +
			n14*n25*n33*n52 + n15*n23*n34*n52 - n13*n25*n34*n52 -
			n14*n23*n35*n52 + n13*n24*n35*n52 + n15*n24*n32*n53 -
			n14*n25*n32*n53 - n15*n22*n34*n53 + n12*n25*n34*n53 +
			n14*n22*n35*n53 - n12*n24*n35*n53 - n15*n23*n32*n54 +
			n13*n25*n32*n54 + n15*n22*n33*n54 - n12*n25*n33*n54 -
			n13*n22*n35*n54 + n12*n23*n35*n54 + n14*n23*n32*n55 -
			n13*n24*n32*n55 - n14*n22*n33*n55 + n12*n24*n33*n55 +
			n13*n22*n34*n55 - n12*n23*n34*n55)/det;

	i[0][4] = (n15*n24*n33*n42 -
			n14*n25*n33*n42 - n15*n23*n34*n42 + n13*n25*n34*n42 +
			n14*n23*n35*n42 - n13*n24*n35*n42 - n15*n24*n32*n43 +
			n14*n25*n32*n43 + n15*n22*n34*n43 - n12*n25*n34*n43 -
			n14*n22*n35*n43 + n12*n24*n35*n43 + n15*n23*n32*n44 -
			n13*n25*n32*n44 - n15*n22*n33*n44 + n12*n25*n33*n44 +
			n13*n22*n35*n44 - n12*n23*n35*n44 - n14*n23*n32*n45 +
			n13*n24*n32*n45 + n14*n22*n33*n45 - n12*n24*n33*n45 -
			n13*n22*n34*n45 + n12*n23*n34*n45)/det;

	// Fill complex couplings beta and productions f_prod vectors from fitter, convert the polar complex numbers to cartesian ones.
	const std::complex<double> Kmatrix_beta1(_Kmatrix_beta1_Amplitude*cos(_Kmatrix_beta1_Phase), _Kmatrix_beta1_Amplitude*sin(_Kmatrix_beta1_Phase));
	const std::complex<double> Kmatrix_beta2(_Kmatrix_beta2_Amplitude*cos(_Kmatrix_beta2_Phase), _Kmatrix_beta2_Amplitude*sin(_Kmatrix_beta2_Phase));
	const std::complex<double> Kmatrix_beta3(_Kmatrix_beta3_Amplitude*cos(_Kmatrix_beta3_Phase), _Kmatrix_beta3_Amplitude*sin(_Kmatrix_beta3_Phase));
	const std::complex<double> Kmatrix_beta4(_Kmatrix_beta4_Amplitude*cos(_Kmatrix_beta4_Phase), _Kmatrix_beta4_Amplitude*sin(_Kmatrix_beta4_Phase));
	const std::complex<double> Kmatrix_beta5(_Kmatrix_beta5_Amplitude*cos(_Kmatrix_beta5_Phase), _Kmatrix_beta5_Amplitude*sin(_Kmatrix_beta5_Phase));

	const std::complex<double> Kmatrix_f_prod_11(_Kmatrix_f_prod_11_Amplitude*cos(_Kmatrix_f_prod_11_Phase), _Kmatrix_f_prod_11_Amplitude*sin(_Kmatrix_f_prod_11_Phase));
	const std::complex<double> Kmatrix_f_prod_12(_Kmatrix_f_prod_12_Amplitude*cos(_Kmatrix_f_prod_12_Phase), _Kmatrix_f_prod_12_Amplitude*sin(_Kmatrix_f_prod_12_Phase));
	const std::complex<double> Kmatrix_f_prod_13(_Kmatrix_f_prod_13_Amplitude*cos(_Kmatrix_f_prod_13_Phase), _Kmatrix_f_prod_13_Amplitude*sin(_Kmatrix_f_prod_13_Phase));
	const std::complex<double> Kmatrix_f_prod_14(_Kmatrix_f_prod_14_Amplitude*cos(_Kmatrix_f_prod_14_Phase), _Kmatrix_f_prod_14_Amplitude*sin(_Kmatrix_f_prod_14_Phase));
	const std::complex<double> Kmatrix_f_prod_15(_Kmatrix_f_prod_15_Amplitude*cos(_Kmatrix_f_prod_15_Phase), _Kmatrix_f_prod_15_Amplitude*sin(_Kmatrix_f_prod_15_Phase));

	std::complex<double> _beta[5];
	_beta[0] = std::complex<double>(Kmatrix_beta1.real(), Kmatrix_beta1.imag());
	_beta[1] = std::complex<double>(Kmatrix_beta2.real(), Kmatrix_beta2.imag());
	_beta[2] = std::complex<double>(Kmatrix_beta3.real(), Kmatrix_beta3.imag());
	_beta[3] = std::complex<double>(Kmatrix_beta4.real(), Kmatrix_beta4.imag());
	_beta[4] = std::complex<double>(Kmatrix_beta5.real(), Kmatrix_beta5.imag());

	const std::complex<double> _fr11prod(Kmatrix_f_prod_11.real(), Kmatrix_f_prod_11.imag());
	const std::complex<double> _fr12prod(Kmatrix_f_prod_12.real(), Kmatrix_f_prod_12.imag());
	const std::complex<double> _fr13prod(Kmatrix_f_prod_13.real(), Kmatrix_f_prod_13.imag());
	const std::complex<double> _fr14prod(Kmatrix_f_prod_14.real(), Kmatrix_f_prod_14.imag());
	const std::complex<double> _fr15prod(Kmatrix_f_prod_15.real(), Kmatrix_f_prod_15.imag());

	const double _s0prod = _Kmatrix_s_prod_0;

	vector<std::complex<double>> U1j;
	for(int j=0;j<5;j++) U1j.push_back(i[0][j]);

	// Compute product of inverse matrix times production vector, split production vector into two pieces
	std::complex<double> value0(0,0);
	std::complex<double> value1(0,0);

	// Compute inverse_matrix times first part of production vector, sum all the poles
	for(int l=0;l<5;l++) {
		for (int pole_index=0;pole_index<5;pole_index++) {
			const std::complex<double> A=_beta[pole_index]*g[pole_index][l];
			const double B=ma[pole_index]*ma[pole_index]-s;

			value0 += U1j[l] * A / B;
		}
	}

	// Compute inverse_matrix times second part of production vector
	value1 += U1j[0]*_fr11prod;
	value1 += U1j[1]*_fr12prod;
	value1 += U1j[2]*_fr13prod;
	value1 += U1j[3]*_fr14prod;
	value1 += U1j[4]*_fr15prod;
	value1 *= (1-_s0prod)/(s-_s0prod); // MR20150303

	// Compute final F0 vector
	std::complex<double> F_0_TComplex = value0 + value1;

	return F_0_TComplex;
}
