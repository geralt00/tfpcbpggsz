#ifndef DALITZ_POINT_CXX
#define DALITZ_POINT_CXX

/***********************************************************************
* Copyright 1998-2020 CERN for the benefit of the EvtGen authors       *
*                                                                      *
* This file is part of EvtGen.                                         *
*                                                                      *
* EvtGen is free software: you can redistribute it and/or modify       *
* it under the terms of the GNU General Public License as published by *
* the Free Software Foundation, either version 3 of the License, or    *
* (at your option) any later version.                                  *
*                                                                      *
* EvtGen is distributed in the hope that it will be useful,            *
* but WITHOUT ANY WARRANTY; without even the implied warranty of       *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
* GNU General Public License for more details.                         *
*                                                                      *
* You should have received a copy of the GNU General Public License    *
* along with EvtGen.  If not, see <https://www.gnu.org/licenses/>.     *
***********************************************************************/

#include "DalitzPoint.hh"

#include <assert.h>
#include <math.h>
#include <stdio.h>

DalitzPoint::DalitzPoint() :
    _m0( -1. ), _mp( -1. ), _mn( -1. ), _q0p( -1. ), _q0n( -1. ), _qpn( -1. )
{
}

DalitzPoint::DalitzPoint( double m_0, double m_p, double m_n, double q_0p, double q_0n,
			  double q_pn ) :
  _m0( m_0 ), _mp( m_p ), _mn( m_n ), _q0p( q_0p ), _q0n( q_0n ), _qpn( q_pn )
{
}

double DalitzPoint::q( std::string parts) const
{
  double ret = -1;
  if ( parts == "0p" )
    ret = _q0p;
  else if ( parts == "0n" )
    ret = _q0n;
  else if ( parts == "pn" )
    ret = _qpn;
  else {
    std::cout << "In DalitzPoint::q -- Wrong particles identifier ---- return -1" << std::endl;
  }
  return ret;
}

double DalitzPoint::m( std::string part ) const
{
  double ret = -1;
  if ( part == "p" ) {
    ret = _m0;
  }
  else if ( part == "p" ) {
    ret = _mp;
  }
  else if ( part == "n" ) {
    ret = _mn;
  }
  else {
    std::cout << "In DalitzPoint::m -- Wrong particle identifier ---- return -1" << std::endl ;
  }
  return ret;
}

bool DalitzPoint::isValid() const
{

  // double M = bigM();  
  // double xmin = pow(_m0 + _mp,2);
  // double xmax = pow(M - _mn,2);
  // double skpip;
  // double yhi = (0.394465 + 1.8821*_q0p - 0.5* TMath::Power(_q0p,2) + 1.86484* TMath::Sqrt(0.044744 - 0.485412*_q0p + 1.13203*TMath::Power(_q0p,2) - 0.541203*TMath::Power(_q0p,3) + 0.0718881*TMath::Power(_q0p,4)))/_q0p;
  // double ylow = (0.394465 + 1.8821*_q0p - 0.5* TMath::Power(_q0p,2) - 1.86484* TMath::Sqrt(0.044744 - 0.485412*_q0p + 1.13203*TMath::Power(_q0p,2) - 0.541203*TMath::Power(_q0p,3) + 0.0718881*TMath::Power(_q0p,4)))/_q0p;  
  // // Initialize boolean
  // bool inside = false;
  // // True if in Dalitz plot
  // if ((xmin <= _q0p) && (_q0p <= xmax) && (_q0n >= ylow) && (_q0n <= yhi)) { inside = true; }
  // return inside;

  double M = bigM();  
  double xmin = pow(_m0 + _mp,2);
  double xmax = pow(M - _mn,2);
  // Find energy of b(c) in ab frame
  double ep_0p = (_q0p - _m0*_m0 + _mp*_mp)/(2.0*sqrt(_q0p));
  double en_0p = (M*M - _q0p - _mn*_mn)/(2.0*sqrt(_q0p));
  double _qpn_hi = pow(ep_0p+en_0p,2) - pow( sqrt(ep_0p*ep_0p-_mp*_mp)-sqrt(en_0p*en_0p-_mn*_mn) ,2);
  double _qpn_lo = pow(ep_0p+en_0p,2) - pow( sqrt(ep_0p*ep_0p-_mp*_mp)+sqrt(en_0p*en_0p-_mn*_mn) ,2);
  // Initialize boolean
  bool inside = false;
  // True if in Dalitz plot
  if ((xmin <= _q0p) && (_q0p <= xmax) && (_qpn_lo <= _qpn) && (_qpn <= _qpn_hi)) { inside = true; }
  return inside;

}  


double DalitzPoint::bigM() const
{
    return sqrt( _q0p + _q0n + _qpn - _m0 * _m0 - _mp * _mp - _mn * _mn );
}

#endif
