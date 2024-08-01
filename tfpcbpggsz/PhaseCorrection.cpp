#include "AmpGen/PhaseCorrection.h"

#include "AmpGen/CompiledExpression.h"
#include "AmpGen/Event.h"
#include "AmpGen/EventType.h"
#include "AmpGen/Expression.h"
#include "AmpGen/MinuitParameterSet.h"
#include "AmpGen/MsgService.h"
#include "AmpGen/NamedParameter.h"
#include "AmpGen/Particle.h"

#include <string>
#include <map>
#include <vector>

using namespace AmpGen;

using CE = CompiledExpression<real_t(const real_t*, const real_t*)>;

PhaseCorrection::PhaseCorrection()
    :   order_{NamedParameter<size_t>("PhaseCorrection::Order", 0, "Order of polynomial of phase correction")} , 
        correctionType_{NamedParameter<std::string>("PhaseCorrection::CorrectionType", "antiSym_legendre")}
{

    if (correctionType_ == "singleBias" || correctionType_ == "doubleBias"){
        if (correctionType_ == "singleBias"){
            INFO("Setting up phase correction for a single gaussian bias");
            nBias_ = 1;
        }else{
            INFO("Setting up phase correction for a double gaussian bias");
            nBias_ = 2;
        }
        for (size_t i{0}; i < nBias_; i++){
            mu_[0].plus = NamedParameter<real_t>("PhaseCorrection::biasParam"+std::to_string(i+1)+":mu+", 0);
            mu_[0].minus = NamedParameter<real_t>("PhaseCorrection::biasParam"+std::to_string(i+1)+":mu-", 0);
            sigma_[0].plus = NamedParameter<real_t>("PhaseCorrection::biasParam"+std::to_string(i+1)+":sigma+", 0);
            sigma_[0].minus = NamedParameter<real_t>("PhaseCorrection::biasParam"+std::to_string(i+1)+":sigma-", 0);
            epsilon_[0] = NamedParameter<real_t>("PhaseCorrection::biasParam"+std::to_string(i+1)+":epsilon", 0);
            A_[0] = NamedParameter<real_t>("PhaseCorrection::biasParam"+std::to_string(i+1)+":A", 0);
            nBias_ = 1;
        }
        doBias_ = true;
    }else{
        if (correctionType_ != "antiSym_legendre"){
            WARNING("Correction type " << correctionType_ << " not supported :(, defaulted to an anti-symmetric legendre");
        }
        for (size_t i{0}; i < order_ ; i++){
            for (size_t j{1} ; j < order_-i+1 ; j+=2){
                coefficients_.push_back(NamedParameter<real_t>("PhaseCorrection::C_"+std::to_string(i) + "_" + std::to_string(j), 0));
                iIndices_.push_back(i); jIndices_.push_back(j);
            }
        }
        nTerms_ = coefficients_.size();
        INFO("Set up for phase correction order " << order_ << " with " << nTerms_ << " terms");
        if (order_ != 0){ doPolynomial_ = true; }
    }
}

PhaseCorrection::PhaseCorrection(MinuitParameterSet &MPS)
    :   order_{NamedParameter<size_t>("PhaseCorrection::Order", 0, "Order of polynomial of phase correction")} , 
        correctionType_{NamedParameter<std::string>("PhaseCorrection::CorrectionType", "antiSym_legendre")}
{
  
    if (correctionType_ == "singleBias" || correctionType_ == "doubleBias"){
        if (correctionType_ == "singleBias"){
            INFO("Setting up phase correction for a single gaussian bias");
            nBias_ = 1;
        }else{
            INFO("Setting up phase correction for a double gaussian bias");
            nBias_ = 2;
        }
        for (size_t i{0}; i < nBias_; i++){
            mu_[i].plus = MPS["PhaseCorrection::biasParam"+std::to_string(i+1)+":mu+"]->mean();
            mu_[i].minus = MPS["PhaseCorrection::biasParam"+std::to_string(i+1)+":mu-"]->mean();
            sigma_[i].plus  = MPS["PhaseCorrection::biasParam"+std::to_string(i+1)+":sigma+"]->mean();
            sigma_[i].minus  = MPS["PhaseCorrection::biasParam"+std::to_string(i+1)+":sigma-"]->mean();
            epsilon_[i]  = MPS["PhaseCorrection::biasParam"+std::to_string(i+1)+":epsilon"]->mean();
            A_[i] = MPS["PhaseCorrection::biasParam"+std::to_string(i+1)+":A"]->mean();
        }
        doBias_ = true;
    }else{
        if (correctionType_ != "antiSym_legendre"){
            WARNING("Correction type " << correctionType_ << " not supported :(, defaulted to an anti-symmetric legendre");
        }
        for (size_t i{0}; i < order_ ; i++){
            for (size_t j{1} ; j < order_-i+1 ; j+=2){
                coefficients_.push_back(MPS["PhaseCorrection::C_" + std::to_string(i) + "_" + std::to_string(j)]->mean());
                iIndices_.push_back(i); jIndices_.push_back(j);
            }
        }
        nTerms_ = coefficients_.size();
        INFO("Set up for phase correction order " << order_ << " with " << nTerms_ << " terms");
        if (order_ != 0){ doPolynomial_ = true; }
    }


}


dalitzPair<Expression> PhaseCorrection::transformedDalitzCoords(EventType& eventType)
{
    Particle mother(eventType.decayDescriptor(), eventType.finalStates());

    // store 4 momenta for easy access
    Tensor p1(mother.daughter(0)->P());
    Tensor p2(mother.daughter(1)->P());
    Tensor p3(mother.daughter(2)->P());

    //invariant masses
    Expression s12 = 
        (p1[3] + p2[3]) * (p1[3] + p2[3]) - 
        (p1[0] + p2[0]) * (p1[0] + p2[0]) - 
        (p1[1] + p2[1]) * (p1[1] + p2[1]) - 
        (p1[2] + p2[2]) * (p1[2] + p2[2]);
    Expression s13 = 
        (p1[3] + p3[3]) * (p1[3] + p3[3]) - 
        (p1[0] + p3[0]) * (p1[0] + p3[0]) - 
        (p1[1] + p3[1]) * (p1[1] + p3[1]) - 
        (p1[2] + p3[2]) * (p1[2] + p3[2]);


    Expression rotatedSymCoord = (s13 + s12)/2; // rotated sym coord = z+
    Expression rotatedAntiSymCoord = (s13 - s12)/2; // rotated antisym coord = z-
    Expression stretchedSymCoord = m1_ * rotatedSymCoord + c1_; // stretched coords = z'+
    Expression stretchedAntiSymCoord = m2_ * rotatedAntiSymCoord + c2_; // = z'-

    if (NamedParameter<bool>("PhaseCorrection::stretchAntiSym", true)){
        auto antiSym_scale = Expression(NamedParameter<real_t>("PhaseCorrection::stretchAntiSym_scale", 2));
        auto antiSym_offset = Expression(NamedParameter<real_t>("PhaseCorrection::stretchAntiSym_offset", 2));
        stretchedAntiSymCoord = antiSym_scale * stretchedAntiSymCoord/(antiSym_offset + stretchedSymCoord); // = z''-
    }
    return {stretchedSymCoord, stretchedAntiSymCoord};
}

// NOT CURRENTLY USING THIS ONE
dalitzPair<Expression> PhaseCorrection::squareDalitzCoords(EventType& eventType)
{
    // RECALL: event should be D0 -> Ks0 pi- pi+
    Particle mother(eventType.decayDescriptor(), eventType.finalStates());

    // store 4 momenta for easy access
    Tensor p1(mother.daughter(0)->P());
    Tensor p2(mother.daughter(1)->P());
    Tensor p3(mother.daughter(2)->P());

    //invariant masses
    Expression s12 = 
        (p1[3] + p2[3]) * (p1[3] + p2[3]) - 
        (p1[0] + p2[0]) * (p1[0] + p2[0]) - 
        (p1[1] + p2[1]) * (p1[1] + p2[1]) - 
        (p1[2] + p2[2]) * (p1[2] + p2[2]);
    Expression s13 = 
        (p1[3] + p3[3]) * (p1[3] + p3[3]) - 
        (p1[0] + p3[0]) * (p1[0] + p3[0]) - 
        (p1[1] + p3[1]) * (p1[1] + p3[1]) - 
        (p1[2] + p3[2]) * (p1[2] + p3[2]);
    Expression s23 = 
        (p2[3] + p3[3]) * (p2[3] + p3[3]) - 
        (p2[0] + p3[0]) * (p2[0] + p3[0]) - 
        (p2[1] + p3[1]) * (p2[1] + p3[1]) - 
        (p2[2] + p3[2]) * (p2[2] + p3[2]);
    
    // store masses for easy access
    real_t M0 = mother.mass();
    real_t M1 = mother.daughter(0)->mass();
    real_t M2 = mother.daughter(1)->mass();
    real_t M3 = mother.daughter(2)->mass();

    // Energy of the daughters in the rest frame of the D0
    Expression E1 = (fcn::pow(M1, 2) + M0 * M0 - s23)/(2 * M0);
    Expression E2 = (fcn::pow(M2, 2) + M0 * M0 - s13)/(2 * M0);
    Expression E3 = (fcn::pow(M3, 2) + M0 * M0 - s12)/(2 * M0);

    // angle between momenta of particle xy for use in square dalitz plots
    Expression cos12 = (2 * E1 * E2 - (s12 - fcn::pow(M1, 2) - fcn::pow(M2, 2)))/(2 * fcn::sqrt( fcn::pow(E1, 2) - fcn::pow(M1, 2)) * fcn::sqrt(fcn::pow(E2, 2) - fcn::pow(M2, 2)));
    Expression cos13 = (2 * E1 * E3 - (s13 - fcn::pow(M1, 2) - fcn::pow(M3, 2)))/(2 * fcn::sqrt( fcn::pow(E1, 2) - fcn::pow(M1, 2)) * fcn::sqrt(fcn::pow(E3, 2) - fcn::pow(M3, 2)));
    Expression theta = (fcn::acos(cos12) - fcn::acos(cos13));

    // Not looked at/used this yet, don't realy know what it's doing
    real_t z1Min = std::pow(2 * 0.13957039, 2);
    real_t z1Max = std::pow(1.86484 - 0.497611, 2);
    Expression w1 = 2 * (s23 - z1Min)/(z1Max - z1Min) - 1;
    Expression w2 = theta/M_PI;

    return {w1, w2};
}

Expression PhaseCorrection::legendre(Expression& x, real_t n)
{
    if (n==0) return 1;
    if (n==1) return x;
    Expression num = (2 - 1/n) * x * legendre(x, n-1) - (1 - 1/n) * legendre(x, n-2);
    return num;
}

real_t PhaseCorrection::gaussianExponent(real_t& s, real_t& mu, real_t& sigma)
{
    return std::pow( (s-mu)/sigma , 2);
}

real_t PhaseCorrection::bias(dalitzPair<real_t>& coords, size_t& index)
{
    real_t erfArgument{ (coords.plus - coords.minus)/epsilon_[index] };
    real_t plusExponent, minusExponent;

    if (coords.plus > coords.minus){
        plusExponent = gaussianExponent(coords.plus, mu_[index].plus, sigma_[index].plus);
        minusExponent = gaussianExponent(coords.minus, mu_[index].minus, sigma_[index].minus);
    }else{
        plusExponent = gaussianExponent(coords.plus, mu_[index].minus, sigma_[index].minus);
        minusExponent = gaussianExponent(coords.minus, mu_[index].plus, sigma_[index].plus);
    }

    return A_[index] * std::erf(erfArgument) * std::exp(-plusExponent - minusExponent);
}


Expression PhaseCorrection::polynomial(EventType& eventType, size_t i, size_t j)
{
    dalitzPair<Expression> transformedCoords{transformedDalitzCoords(eventType)}; //could use square coords here if you understood them

    Expression Pi{ legendre(transformedCoords.plus, i)};
    Expression Pj{ legendre(transformedCoords.minus, j)};

    return Pi * Pj;
}

void PhaseCorrection::compilePolynomialExpressions(EventType& eventType)
{
    for (int n{0} ; n < nTerms_ ; n++){
        Expression polyExpression{polynomial(eventType, iIndices_[n], jIndices_[n])};
        CE polyCompiledExpression(polyExpression, "P_" + std::to_string(iIndices_[n]) + "_" + std::to_string(jIndices_[n]), eventType.getEventFormat());
        polyCompiledExpression.prepare(); polyCompiledExpression.compile();
        compiledExpressions_.push_back(polyCompiledExpression);
        INFO("Compiled Phase Correction term P_"<<iIndices_[n]<<"_"<<jIndices_[n]);
    }
    return;
}


real_t PhaseCorrection::eval(const Event& event)
{
    real_t deltaC{0};
    for (size_t n{0}; n < nTerms_; n++){
        deltaC += coefficients_[n] * compiledExpressions_[n](event.address());
    }
    return deltaC;
    // return TEMPdeltaC_;
}

real_t PhaseCorrection::evalBias(Event& event)
{
    real_t deltaC{0};
    dalitzPair<real_t> coords{ event.s(0, 2), event.s(0, 1) };
    for (size_t i{0}; i < nBias_; i++){

        deltaC += bias(coords, i);
    }

    return deltaC;
    // return TEMPdeltaC_;
}


void PhaseCorrection::updateCoeffs(MinuitParameterSet& MPS)
{   
    size_t n{0};
    for (size_t i{0}; i < order_ ; i++){
        for (size_t j{1} ; j < order_-i+1 ; j+=2){
            coefficients_[n] = (MPS["PhaseCorrection::C_" + std::to_string(i) + "_" + std::to_string(j)]->mean());
            n++;
        }
    }
    // TEMPdeltaC_ = MPS["deltaC"]->mean();

}

bool PhaseCorrection::doPolynomial()
{
    return doPolynomial_;
}
bool PhaseCorrection::doBias()
{
    return doBias_;
}

