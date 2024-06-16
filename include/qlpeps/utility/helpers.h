/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-06-16
*
* Description: QuantumLiquids/PEPS project.
*/

#ifndef QLPEPS_UTILITY_HELPERS_H
#define QLPEPS_UTILITY_HELPERS_H

#include <complex>
namespace qlpeps {

double ComplexConjugate(const double &x) { return x; }
std::complex<double> ComplexConjugate(const std::complex<double> &z) { return std::conj(z); }

}//qlpeps
#endif //QLPEPS_UTILITY_HELPERS_H
