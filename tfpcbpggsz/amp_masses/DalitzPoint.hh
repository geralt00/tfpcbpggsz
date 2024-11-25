
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

#ifndef DALITZ_POINT_HH
#define DALITZ_POINT_HH

// This class describes the complete kinematics of the Dalitz decay.
// It holds all the six invariant momentum products, three daughter
// particle masses and three invariant masses of pairs of particles.
// This description is completely symmetric with respect to particle
// permutations.
//
// Another way to slice the six coordinate is to make a transformation
// to the mass of the decaying particle. The four masses make up a
// Dalitz plot. The other two are coordinates of a point in the plot.

#include <iostream>
#include <string>

class DalitzPoint final {
public:
  DalitzPoint();
  DalitzPoint( double m_0, double m_p, double m_n, double q_0p, double q_0n,
	       double q_pn );

  double q( std::string ) const;
  double bigM() const;
  double m( std::string ) const;

  bool isValid() const;

  private:
    double _m0, _mp, _mn;       // masses
    double _q0p, _q0n, _qpn;    // invariant masses squared
};

#endif
