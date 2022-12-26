// Copyright (C) 2009-2010, Luis Alvarez <lalvarez@dis.ulpgc.es>
/* ami_pol.c */

#ifndef _AMI_POL_H_
#define _AMI_POL_H_

#include <math.h>

#include <malloc.h>

long double ami_horner(long double* pol, int degree, long double x,
                       long double* fp);
long double ami_root_bisection(long double* pol, int degree, long double a,
                               long double b, long double TOL);
int ami_polynomial_root(double* pol, int degree, double* root_r,
                        double* root_i);

#endif
