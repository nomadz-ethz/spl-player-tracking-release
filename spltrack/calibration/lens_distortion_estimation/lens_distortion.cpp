/*
 * Copyright (C) 2009-2010, Luis Alvarez <lalvarez@dis.ulpgc.es>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 *    This is the lens_distortion.cpp program (ANSI C language), associated to
 * the lens_distortion_estimation.cpp program; and to the publication An
 * Algebraic Approach to Lens Distortion by Line Rectification, L. Alvarez, L.
 * Gomez, R. Sendra Published in JMIV, July 2009
 */

/**
 *    Coded by Luis Alvarez and Luis Gomez, AMI Research Group, University of
 *    Las Palmas de Gran Canaria, Canary Islands, SPAIN
 *    First version: February 2010, Second Version: January 2012 (this is the
 * second version) In this version, we optimize the center of distortion using a
 * search patch pattern strategy
 */
/**
 *       ALGEBRAIC PART
 */

#include "lens_distortion.h"

/**
GLOBAL VARIABLES
*/
int n_iterations = 0;     /**  TOTAL NUMBER OF ITERATIONS */
int f_evaluations = 0;    /**  TOTAL FUNCTION EVALUATIONS */
double tol_lambda = 1e-8; /**  TOLERANCE FOR THE LAMBDA SEARCHING */

/**
 * \fn int test_compatibility_lens_distortion_model(double *a,int Na,double
 *max_radius) \brief function checks if the lens distortion model is an
 *increasing function in [0,max_radius] \author Luis Alvarez \return return 0 if
 *the test is satisfied
 **/
int test_compatibility_lens_distortion_model(double* a, int Na,
                                             double max_radius) {
  int k;
  double r, *b; /* AUXILIAR VARIABLES */
  /* WE INIT TO NULL ALL POINTERS */
  b = NULL;

  /* WE ALLOCATE MEMORY */
  b = (double*)malloc(sizeof(double) * (Na + 1));
  // WE BUILD THE DERIVATIVE OF THE POLYNOMIAL
  for (k = 1; k <= Na; k++) b[k] = (k + 1) * a[k];
  for (r = 1; r < max_radius; r += 1.) {
    if (ami_polynomial_evaluation(b, Na, r) < 0) {
      free(b);
      return (-1.);
    }
  }
  free(b);
  return (0);
}

/**
* \fn int ami_line2d_calculation( double line[3],double **Points2D,int N)
* \brief function to compute a line equation by minimizing the distance to a
point collection
* \author Luis Alvarez and Luis Gomez
* \return return 0 if function finishes properly and !0 otherwise
*/
int ami_line2d_calculation(double line[3] /**  line coefficients (ax+by+c=0) */,
                           double** Points2D /**  set of 2D points */,
                           int N /**  number of points */
) {
  int i, j, k; /* AUXILIAR VARIABLES */
  double suu, suv, svv, um, vm, h, r[4][3], min, aux,
      norm;                       /* AUXILIAR VARIABLES */
  double zero = 0.00000000000001; /* AUXILIAR VARIABLES */

  if (N < 2) {
    printf("Number of point to calculate the line must be >2\n");
    return (-1);
  }

  suu = 0;
  suv = 0;
  svv = 0;
  um = 0;
  vm = 0;

  for (i = 0; i < N; i++) {
    um += Points2D[i][0];
    vm += Points2D[i][1];
  }

  um /= N;
  vm /= N;

  for (i = 0; i < N; i++) {
    suu += (Points2D[i][0] - um) * (Points2D[i][0] - um);
    svv += (Points2D[i][1] - vm) * (Points2D[i][1] - vm);
    suv += (Points2D[i][0] - um) * (Points2D[i][1] - vm);
  }

  suu /= N;
  svv /= N;
  suv /= N;

  if (fabs(suv) <= zero) {
    if (suu < svv && svv > zero) {
      line[0] = 1;
      line[1] = 0;
      line[2] = -um;
      return (0);
    }
    if (svv < suu && suu > zero) {
      line[0] = 0;
      line[1] = 1;
      line[2] = -vm;
      return (0);
    }
    printf(" It is not possible to calculate the 2D line\n");
    return (-1);
  }

  r[2][1] = r[3][1] = r[0][0] = r[1][0] = 1.;
  h = 0.5 * (suu - svv) / suv;

  if (h > 0) {
    r[0][1] = -h - sqrt(1. + h * h);
    r[0][2] = -(um + r[0][1] * vm);
    r[1][1] = -1. / r[0][1];
    r[1][2] = -(um + r[1][1] * vm);
    r[2][0] = h + sqrt(1. + h * h);
    r[2][2] = -(r[2][0] * um + vm);
    r[3][0] = -1. / r[2][0];
    r[3][2] = -(r[3][0] * um + vm);
  } else {
    r[0][1] = -h + sqrt(1 + h * h);
    r[0][2] = -(um + r[0][1] * vm);
    r[1][1] = -1. / r[0][1];
    r[1][2] = -(um + r[1][1] * vm);
    r[2][0] = h - sqrt(1 + h * h);
    r[2][2] = -(r[2][0] * um + vm);
    r[3][0] = -1. / r[2][0];
    r[3][2] = -(r[3][0] * um + vm);
  }

  for (j = 0; j < 4; j++) {
    norm = sqrt(r[j][0] * r[j][0] + r[j][1] * r[j][1]);
    for (i = 0; i < 3; i++) r[j][i] /= norm;
  }

  min = 0.;
  k = 0;

  for (i = 0; i < N; i++) {
    aux = r[0][0] * Points2D[i][0] + r[0][1] * Points2D[i][1] + r[0][2];
    min += fabs(aux);
  }

  for (j = 1; j < 4; j++) {
    h = 0;
    for (i = 0; i < N; i++) {
      aux = r[j][0] * Points2D[i][0] + r[j][1] * Points2D[i][1] + r[j][2];
      h += fabs(aux);
    }
    if (h < min) {
      k = j;
      min = h;
    }
  }

  line[0] = r[k][0];
  line[1] = r[k][1];
  line[2] = r[k][2];
  return (0);
}

/**
* \fn int ami_lens_distortion_polynomial_update_distance_2v(double *x,
double *y,int Np,double *a,int Na,double x0,double y0,int k1,int k2,
double **pol, double alpha)
* \brief function to add the information of a line point sequence to the 4
degree polynomial to compute the lens distortion model
* \author Luis Alvarez
* \return return 0 if function finishes properly
*/

int ami_lens_distortion_polynomial_update_distance_2v(
    double* x, double* y /** DISTORTED LINE COORDINATES (INPUT) */,
    int Np /** NUMBER OF POINTS (INPUT)*/,
    double* a /** POLYNOMIAL DEFINING THE LENS DISTORTION MODEL (INPUT) */,
    int Na /** DEGREE OF POLYNOMIAL MODEL FOR LENS DISTORTION */, double x0,
    double y0 /** COORDINATES OF THE IMAGE CENTER */,
    int k1 /** COEFFICIENT 1 OF THE POLYNOMIAL TO BE UPDATED*/,
    int k2 /** COEFFICIENT 2 OF THE POLYNOMIAL TO BE UPDATED*/,
    double** pol /** 4 DEGREE 2 VARIABLE POLYNOM TO MINIMIZE (INPUT-OUTPUT) */,
    double alpha) { /** WEIGHT OF THE DISTANCE IN THE POLYNOM ENERGY */
  int i, j, k;      /* AUXILIAR VARIABLES */
  double *A, *x2, *y2, *d1, *d2;                      /* AUXILIAR VARIABLES */
  double aux, **pol1, **pol2, **p_xx, **p_xy, **p_yy; /* AUXILIAR VARIABLES */

  A = x2 = y2 = d1 = d2 = NULL;
  pol1 = pol2 = p_xx = p_xy = p_yy = NULL;

  /* WE CHECK alpha VALUE */
  if (alpha == 0) return (0);

  /* WE ALLOCATE MEMORY */
  A = (double*)malloc(sizeof(double) * Np);
  x2 = (double*)malloc(sizeof(double) * Np);
  y2 = (double*)malloc(sizeof(double) * Np);
  d1 = (double*)malloc(sizeof(double) * Np);
  d2 = (double*)malloc(sizeof(double) * Np);
  ami_calloc2d(pol1, double, 2, 2);
  ami_calloc2d(pol2, double, 2, 2);
  ami_calloc2d(p_xx, double, 3, 3);
  ami_calloc2d(p_xy, double, 3, 3);
  ami_calloc2d(p_yy, double, 3, 3);

  /* WE COMPUTE THE DISTANCE TO THE IMAGE CENTER */
  for (i = 0; i < Np; i++)
    d1[i] = sqrt((x[i] - x0) * (x[i] - x0) + (y[i] - y0) * (y[i] - y0));

  /* WE COMPUTE THE POINT TRANSFORMATION WITH THE CURRENT LENS DISTORTION MODEL
   */
  for (i = 0; i < Np; i++) {
    A[i] = ami_polynomial_evaluation(a, Na, d1[i]);
    x2[i] = x0 + (x[i] - x0) * A[i];
    y2[i] = y0 + (y[i] - y0) * A[i];
  }
  /* WE COMPUTE THE POLYNOMS CORRESPONDING TO THE DISTANCE ERROR */
  for (i = 0; i <= 2; i++)
    for (j = 0; j <= 2; j++) p_xx[i][j] = 0.;

  for (i = 0; i < Np; i++) {
    aux = 0;
    for (k = 1; k <= Na; k++)
      if (k != k1 && k != k2) aux += a[k] * pow(d1[i], (double)k);
    pol1[0][0] = aux * d1[i];
    pol1[1][0] = pow(d1[i], (double)k1 + 1);
    pol1[0][1] = pow(d1[i], (double)k2 + 1);
    ami_2v_polynom_multiplication(pol1, 1, pol1, 1, p_xx);
  }

  for (i = 0; i <= 2; i++)
    for (j = 0; j <= 2; j++) p_xx[i][j] = alpha * p_xx[i][j] / Np;

  /* WE UPDATE THE ERROR POLYNOM */
  ami_2v_polynom_multiplication(p_xx, 2, p_xx, 2, pol);

  /* WE FREE THE MEMORY */
  if (A != NULL) free(A);
  if (x2 != NULL) free(x2);
  if (y2 != NULL) free(y2);
  if (d1 != NULL) free(d1);
  if (d2 != NULL) free(d2);
  if (p_xx != NULL) ami_free2d(p_xx);
  if (p_xy != NULL) ami_free2d(p_xy);
  if (p_yy != NULL) ami_free2d(p_yy);
  if (pol1 != NULL) ami_free2d(pol1);
  if (pol2 != NULL) ami_free2d(pol2);

  return (0);
}

/**
* \fn double ami_lens_distortion_estimation_2v(double **x,double **y,int Nl,
int *Np,double x0,double y0,double *a,int Na,int k1,int k2, double alpha )
* \brief function to update the lens distortion polynomial model for 2 variables
. If alpha>0, we adapt a[0] to minimize the square distence between distorted
and undistorted points and we add a term to the polynomial also minimizing such
distance with weight alpha
* \author Luis Alvarez
* \return return the Error/Nl
*/
double ami_lens_distortion_estimation_2v(
    double** x, double** y /** DISTORTED LINE COORDINATES (INPUT)*/,
    int Nl /** NUMBER OF LINES (RECTS)*/,
    int* Np /** NUMBER OF POINTS (INPUT)*/, double x0,
    double y0 /** COORDINATES OF THE IMAGE CENTER */,
    double* a /** POLYNOMIAL DEFINING THE LENS DISTORTION MODEL (INPUT) */,
    int Na /** DEGREE OF POLYNOMIAL MODEL FOR LENS DISTORTION */,
    int k1 /** COEFFICIENT 1 OF THE POLYNOMIAL TO BE UPDATED*/,
    int k2 /** COEFFICIENT 2 OF THE POLYNOMIAL TO BE UPDATED*/,
    double alpha /** WEIGHT OF THE DISTANCE IN THE POLYNOM ENERGY */,
    double max_radius /** MAXIMUM RADIAL DISTANCE IN PHOTO */
) {
  int i, k, m;                                      /* AUXILIAR VARIABLES */
  double **pol_v2, d, sum_Ad, sum_dd, A, Error = 0; /* AUXILIAR VARIABLES */

  /* WE ALLOCATE MEMORY */
  ami_calloc2d(pol_v2, double, 5, 5);

  /* WE UPDATE a[0] BY MINIMIZING THE DISTANCE OF THE DISTORTED POINTS TO
  THE UNDISTORTED POINTS */
  if (alpha > 0) {
    sum_dd = sum_Ad = 0;
    for (m = 0; m < Nl; m++) {
      for (i = 0; i < Np[m]; i++) {
        d = sqrt((x[m][i] - x0) * (x[m][i] - x0) +
                 (y[m][i] - y0) * (y[m][i] - y0));
        A = 0;
        for (k = 1; k <= Na; k++) A += a[k] * pow(d, (double)k + 1);
        sum_dd += d * d;
        sum_Ad += A * d;
      }
    }
    a[0] = 1 - sum_Ad / sum_dd;
  }

  for (m = 0; m < Nl; m++) {
    /* WE UPDATE DE POLYNOM TO MINIMIZE */
    ami_lens_distortion_polynomial_update_2v(x[m], y[m], Np[m], a, Na, x0, y0,
                                             k1, k2, pol_v2);
    ami_lens_distortion_polynomial_update_distance_2v(
        x[m], y[m], Np[m], a, Na, x0, y0, k1, k2, pol_v2, alpha);
  }

  /* WE UPDATE THE POLYNOMIAL LENS DISTORTION MODEL */
  ami_lens_distortion_model_update_2v(a, Na, k1, k2, pol_v2, max_radius);
  ami_free2d(pol_v2);
  for (m = 0; m < Nl; m++)
    Error += ami_LensDistortionEnergyError(x[m], y[m], Np[m], x0, y0, a, Na);
  return (Error / Nl);
}

/**
* \fn int ami_lens_distortion_model_update_2v( double *a,int Na,int k1,int k2,
double **pol)
* \brief function to update the lens distortion model by minimizing a 4 degree
2 variable polynom
* \author Luis Alvarez
* \return return 0 if the function finishes properly
*/
int ami_lens_distortion_model_update_2v(
    double*
        a /** POLYNOMIAL DEFINING THE LENS DISTORTION MODEL (INPUT-OUTPUT) */,
    int Na /** DEGREE OF POLYNOMIAL MODEL FOR LENS DISTORTION (INPUT)*/,
    int k1 /** COEFFICIENT 1 OF THE POLYNOMIAL TO BE UPDATED (INPUT)*/,
    int k2 /** COEFFICIENT 2 OF THE POLYNOMIAL TO BE UPDATED (INPUT)*/,
    double** pol /** 4 DEGREE POLYNOM TO MINIMIZE (INPUT) */,
    double max_radius /** MAXIMUM RADIAL DISTANCE IN PHOTO */
) {
  int j, i, M, Nr = 0, m = Na; /* AUXILIAR VARIABLES */
  double *x, **pol_x, **pol_y, *pol_r, xr, yr, Emin, *rx, *ry, *b2, *p3,
      *b;                                            /* AUXILIAR VARIABLES */
  double sx, sy, Energy; /* NORMALIZATION FACTORS */ /* AUXILIAR VARIABLES */
  double p_r3[6][6][19];                             /* AUXILIAR VARIABLES */

  x = pol_r = rx = ry = b2 = p3 = NULL; /*Added by Luis Gomez*/
  pol_x = pol_y = NULL;                 /*Added by Luis Gomez*/

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      for (m = 0; m < 19; m++) p_r3[i][j][m] = 0.;

  /* WE ALLOCATE MEMORY */
  x = (double*)malloc(sizeof(double) * 3);
  ami_calloc2d(pol_x, double, 5, 5);
  ami_calloc2d(pol_y, double, 5, 5);
  p3 = (double*)malloc(sizeof(double) * 5);
  b = (double*)malloc(sizeof(double) * (Na + 1));
  // WE FILL THE AUXILIARY LENS DISTORTION MODEL
  for (i = 0; i <= Na; i++) b[i] = a[i];

  /* WE NORMALIZE POLYNOM COEFFICIENT */
  sx = pow(pol[4][0], (double)0.25);
  sy = pow(pol[0][4], (double)0.25);
  for (i = 0; i <= 4; i++) {
    for (j = 0; j <= 4; j++) {
      if (i > 0) pol[i][j] /= pow(sx, (double)i);
      if (j > 0) pol[i][j] /= pow(sy, (double)j);
    }
  }

  /* WE COMPUTE THE DERIVATIVES OF THE POLYNOM */
  ami_2v_polynom_derivatives(pol, 4, pol_x, pol_y);

  /* WE FILL THE MATRIX TO COMPUTE THE DETERMINANT */
  for (i = 0; i <= 3; i++) {
    for (m = 0; m <= 4; m++) {
      p_r3[2][i + 2][m] = p_r3[1][i + 1][m] = p_r3[0][i][m] = pol_x[3 - i][m];
      p_r3[5][i + 2][m] = p_r3[4][i + 1][m] = p_r3[3][i][m] = pol_y[3 - i][m];
    }
  }
  /* WE COMPUTE THE RESOLVENT POLYNOM */
  pol_r = (double*)malloc(sizeof(double) * 19);
  ami_polynom_determinant(p_r3, 18, 6, pol_r);

  /* WE COMPUTE THE RESOLVENT POLYNOM DEGREE */
  for (i = 0; i <= 18; i++) {
    if (pol_r[i] != 0) Nr = i;
  }
  /* WE COMPUTE THE ROOT OF THE RESOLVENT POLYNOM */
  rx = (double*)malloc(sizeof(double) * Nr);
  ry = (double*)malloc(sizeof(double) * Nr);
  b2 = (double*)malloc(sizeof(double) * (Nr + 1));
  for (i = 0; i <= Nr; i++) b2[i] = pol_r[Nr - i];
  for (i = 0; i < Nr; i++) {
    rx[i] = 0.0; /*Added by Luis Gomez*/
    ry[i] = 0.0;
  }
  Nr = ami_polynomial_root(b2, Nr, rx, ry);
  /* WE COMPUTE THE X COMPONENT BY REPLACING THE ROOTS IN THE DERIVATIVES
  OF THE POLYNOM */
  xr = 0;
  yr = 0;
  Emin = 10e90;
  for (i = 0; i < Nr; i++) {
    if (fabs(ry[i]) > 0.000000000000001) continue;
    ami_2v_polynom_to_1v_polynom(pol_x, 4, p3, rx[i], 1);
    M = ami_RootCubicPolynomial(p3, 3, x);
    for (m = 0; m < M; m++) {
      Energy = ami_2v_polynom_evaluation(pol, 4, x[m], rx[i]);
      if (Energy < Emin) {
        b[k1] = a[k1] + (x[m] / sx);
        b[k2] = a[k2] + (rx[i] / sy);
        if (test_compatibility_lens_distortion_model(b, Na, max_radius) == 0) {
          Emin = Energy;
          xr = rx[i];
          yr = x[m];
        }
      }
    }
    ami_2v_polynom_to_1v_polynom(pol_y, 4, p3, rx[i], 1);
    M = ami_RootCubicPolynomial(p3, 3, x);
    for (m = 0; m < M; m++) {
      Energy = ami_2v_polynom_evaluation(pol, 4, x[m], rx[i]);
      if (Energy < Emin) {
        b[k1] = a[k1] + (x[m] / sx);
        b[k2] = a[k2] + (rx[i] / sy);
        if (test_compatibility_lens_distortion_model(b, Na, max_radius) == 0) {
          Emin = Energy;
          xr = rx[i];
          yr = x[m];
        }
      }
    }
  }
  /* WE UPDATE THE DISTORSION POLYNOMIAL MODEL */
  a[k1] += (yr / sx);
  a[k2] += (xr / sy);

  /* WE FREE THE MEMORY */
  if (x != NULL) free(x);
  if (pol_x != NULL) ami_free2d(pol_x);
  if (pol_y != NULL) ami_free2d(pol_y);
  if (pol_r != NULL) free(pol_r);
  if (p3 != NULL) free(p3);
  if (rx != NULL) free(rx);
  if (ry != NULL) free(ry);
  if (b2 != NULL) free(b2);
  if (b != NULL) free(b);
  return (0);
}

/**
* \fn int ami_lens_distortion_polynomial_update_2v(double *x, double *y,int Np,
double *a,int Na,double x0,double y0,int k1, double **pol)
* \brief function To add the information of a line point sequence to the 4
degree polynomial to compute the lens distortion model
* \author Luis Alvarez
* \return return 0 if the function finishes properly
*/
int ami_lens_distortion_polynomial_update_2v(
    double* x, double* y /** DISTORTED LINE COORDINATES (INPUT)*/,
    int Np /** NUMBER OF POINTS (INPUT)*/,
    double* a /** POLYNOMIAL DEFINING THE LENS DISTORTION MODEL (INPUT) */,
    int Na /** DEGREE OF POLYNOMIAL MODEL FOR LENS DISTORTION */, double x0,
    double y0 /** COORDINATES OF THE IMAGE CENTER */,
    int k1 /** COEFFICIENT 1 OF THE POLYNOMIAL TO BE UPDATED*/,
    int k2 /** COEFFICIENT 2 OF THE POLYNOMIAL TO BE UPDATED*/,
    double** pol /** 4 DEGREE 2 VARIABLE POLYNOM TO MINIMIZE (INPUT-OUTPUT) */
) {
  int i, j; /* AUXILIAR VARIABLES */
  double *A, *x2, *y2, *d1, *d2, x2_m, y2_m, s_xx, s_yy,
      xA_m;                                            /* AUXILIAR VARIABLES */
  double xd1_m, yA_m, yd1_m, xd2_m, yd2_m;             /* AUXILIAR VARIABLES */
  double paso, **pol1, **pol2, **p_xx, **p_xy, **p_yy; /* AUXILIAR VARIABLES */

  A = x2 = y2 = d1 = d2 = NULL;
  pol1 = pol2 = p_xx = p_xy = p_yy = NULL;

  /* WE ALLOCATE MEMORY */
  A = (double*)malloc(sizeof(double) * Np);
  x2 = (double*)malloc(sizeof(double) * Np);
  y2 = (double*)malloc(sizeof(double) * Np);
  d1 = (double*)malloc(sizeof(double) * Np);
  d2 = (double*)malloc(sizeof(double) * Np);
  ami_calloc2d(pol1, double, 2, 2);
  ami_calloc2d(pol2, double, 2, 2);
  ami_calloc2d(p_xx, double, 3, 3);
  ami_calloc2d(p_xy, double, 3, 3);
  ami_calloc2d(p_yy, double, 3, 3);

  /* WE COMPUTE THE DISTANCE TO THE IMAGE CENTER */
  for (i = 0; i < Np; i++)
    d1[i] = sqrt((x[i] - x0) * (x[i] - x0) + (y[i] - y0) * (y[i] - y0));

  /* WE COMPUTE THE POINT TRANSFORMATION WITH THE CURRENT LENS DISTORTION MODEL
   */
  for (i = 0; i < Np; i++) {
    A[i] = ami_polynomial_evaluation(a, Na, d1[i]);
    x2[i] = x0 + (x[i] - x0) * A[i];
    y2[i] = y0 + (y[i] - y0) * A[i];
  }

  /* WE COMPUTE THE DISTANCE POWER k1 AND k2 (THE COEFFICIENT OF THE LENS
  DISTORTION MODEL TO BE UPDATED */
  for (i = 0; i < Np; i++) {
    paso = d1[i];
    d1[i] = pow(paso, (double)k1);
    d2[i] = pow(paso, (double)k2);
  }

  /* WE COMPUTE THE VARIANCE OF THE TRANSFORMED POINTS */
  x2_m = 0;
  for (i = 0; i < Np; i++) x2_m += x2[i];
  x2_m /= Np;
  s_xx = 0;
  for (i = 0; i < Np; i++) s_xx += (x2[i] - x2_m) * (x2[i] - x2_m);
  s_xx /= Np;
  y2_m = 0;
  for (i = 0; i < Np; i++) y2_m += y2[i];
  y2_m /= Np;
  s_yy = 0;
  for (i = 0; i < Np; i++) s_yy += (y2[i] - y2_m) * (y2[i] - y2_m);
  s_yy /= Np;

  /* WE COMPUTE SOME AVERAGES WE NEED */
  xA_m = 0;
  for (i = 0; i < Np; i++) xA_m += (x[i] - x0) * A[i];
  xA_m /= Np;
  xd1_m = 0;
  for (i = 0; i < Np; i++) xd1_m += (x[i] - x0) * d1[i];
  xd1_m /= Np;
  xd2_m = 0;
  for (i = 0; i < Np; i++) xd2_m += (x[i] - x0) * d2[i];
  xd2_m /= Np;
  yA_m = 0;
  for (i = 0; i < Np; i++) yA_m += (y[i] - y0) * A[i];
  yA_m /= Np;
  yd1_m = 0;
  for (i = 0; i < Np; i++) yd1_m += (y[i] - y0) * d1[i];
  yd1_m /= Np;
  yd2_m = 0;
  for (i = 0; i < Np; i++) yd2_m += (y[i] - y0) * d2[i];
  yd2_m /= Np;

  /* WE COMPUTE THE POLYNOMS OF THE SECOND ORDER MOMENT OF THE POINT
      p_xx p_xy AND p_yy DISTRIBUTION */
  for (i = 0; i < Np; i++) {
    pol1[0][0] = (x[i] - x0) * A[i] - xA_m;
    pol1[1][0] = (x[i] - x0) * d1[i] - xd1_m;
    pol1[0][1] = (x[i] - x0) * d2[i] - xd2_m;
    pol2[0][0] = (y[i] - y0) * A[i] - yA_m;
    pol2[1][0] = (y[i] - y0) * d1[i] - yd1_m;
    pol2[0][1] = (y[i] - y0) * d2[i] - yd2_m;
    ami_2v_polynom_multiplication(pol1, 1, pol1, 1, p_xx);
    ami_2v_polynom_multiplication(pol1, 1, pol2, 1, p_xy);
    ami_2v_polynom_multiplication(pol2, 1, pol2, 1, p_yy);
  }
  for (i = 0; i <= 2; i++)
    for (j = 0; j <= 2; j++) p_xx[i][j] /= 1. /*s_max*/;
  ami_2v_polynom_multiplication(p_xx, 2, p_yy, 2, pol);
  for (i = 0; i <= 2; i++)
    for (j = 0; j <= 2; j++) p_xx[i][j] = -p_xy[i][j] / 1. /*s_max*/;
  ami_2v_polynom_multiplication(p_xy, 2, p_xx, 2, pol);

  /* WE FREE THE MEMORY */
  if (A != NULL) free(A);
  if (x2 != NULL) free(x2);
  if (y2 != NULL) free(y2);
  if (d1 != NULL) free(d1);
  if (d2 != NULL) free(d2);
  ami_free2d(p_xx);
  ami_free2d(p_xy);
  ami_free2d(p_yy);
  ami_free2d(pol1);
  ami_free2d(pol2);

  return (0);
}

/**
* \fn void ami_2v_polynom_derivatives( double **p,int N,double **p_x,
double **p_y)
* \brief function to compute the partial derivatives of a 2 variable polynom.
The degree of the derivative polynoms is assumed to be the same that the
original one
* \author Luis Alvarez
* \return return void
*/
void ami_2v_polynom_derivatives(
    double** p /** ORIGINAL POLYNOM (INPUT)*/,
    int N /** DEGREE OF THE ORIGINAL POLYNOM (INPUT) */,
    double** p_x /** DERIVATIVE OF THE POLYNOM WITH RESPECT TO THE FIRST
                    VARIABLE (OUTPUT) */
    ,
    double** p_y /** DERIVATIVE OF THE POLYNOM WITH RESPECT TO THE SECOND
                    VARIABLE(OUTPUT) */
) {
  int i, j; /* AUXILIAR VARIABLES */

  for (i = 0; i <= N; i++)
    for (j = 0; j <= N; j++) p_x[i][j] = p_y[i][j] = 0;

  for (i = 1; i <= N; i++)
    for (j = 0; j <= N; j++) p_x[i - 1][j] = i * p[i][j];

  for (i = 0; i <= N; i++)
    for (j = 1; j <= N; j++) p_y[i][j - 1] = j * p[i][j];
}

/**
 * \fn void ami_polynom_determinant(double p[6][6][19],int Np,int Nd,double *q)
 * \brief function to compute the determinant of a polynom matrix
 * \author Luis Alvarez
 * \return return void
 */
void ami_polynom_determinant(double p[6][6][19], int Np, int Nd, double* q) {
  int i, j, k, l, m, cont; /* AUXILIAR VARIABLES */
  double* q2;              /* AUXILIAR VARIABLES */
  double p2[6][6][19];     /* AUXILIAR VARIABLES */

  q2 = NULL;

  if (Nd == 1) {
    for (i = 0; i <= 18; i++) q[i] = p[0][0][i];
    return;
  }

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      for (m = 0; m < 19; m++) p2[i][j][m] = 0.;
  q2 = (double*)malloc(sizeof(double) * (Np + 1));

  for (i = 0; i <= Np; i++) q[i] = 0;
  cont = -1;
  for (i = 0; i < Nd; i++) {
    for (k = 0; k <= Np; k++) q2[k] = 0;
    cont *= -1;
    for (k = 0; k < (Nd - 1); k++) {
      for (l = 0; l < (Nd - 1); l++) {
        for (m = 0; m <= Np; m++) {
          p2[k][l][m] = p[k + 1][l >= i ? l + 1 : l][m];
        }
      }
    }
    ami_polynom_determinant(p2, Np, Nd - 1, q2);
    if (cont < 0)
      for (m = 0; m <= Np; m++) q2[m] = -q2[m];
    q = ami_1v_polynom_multiplication(p[0][i], Np, q2, Np, q);
  }
  if (q2 != NULL) free(q2);
}

/**
 * \fn double ami_2v_polynom_evaluation(double **p1,int N1,double x,double y)
 * \brief function to evaluate a 2 variable polynom in one point
 * \author Luis Alvarez
 * \return return the evaluation
 */
double ami_2v_polynom_evaluation(
    double** p1 /** 2 VARIABLE POLYNOM (INPUT)*/,
    int N1 /** DEGREE OF POLYNOM 1 (INPUT)*/, double x,
    double y /** POINT COORDINATE WHERE THE POLYNOM WILL BE EVALUATED (INPUT) */
) {
  int i, j;            /* AUXILIAR VARIABLES */
  double *p, *q, eval; /* AUXILIAR VARIABLES */

  p = q = NULL;

  p = (double*)malloc(sizeof(double) * (N1 + 1));
  q = (double*)malloc(sizeof(double) * (N1 + 1));

  for (i = 0; i <= N1; i++) {
    for (j = 0; j <= N1; j++) p[j] = p1[i][j];
    q[i] = ami_polynomial_evaluation(p, N1, y);
  }
  eval = ami_polynomial_evaluation(q, N1, x);
  if (p != NULL) free(p);
  if (q != NULL) free(q);
  return (eval);
}

/**
* \fn void ami_2v_polynom_to_1v_polynom( double **p1,int N1,double *p3,double z,
int flat)
* \brief function to evaluate a 2 variable polynom in one of the variable value.
The output is a 1 degree polynom
* \author Luis Alvarez
* \return return void
*/
void ami_2v_polynom_to_1v_polynom(
    double** p1 /** 2 VARIABLE POLYNOM (INPUT)*/,
    int N1 /** DEGREE OF POLYNOM 1 (INPUT)*/,
    double* p3 /** OUTPUT 1 VARIABLE POLYNOM (OUTPUT)*/,
    double z /** POINT WHERE THE 2 VARIABLE POLYNOM IS GOING TO BE EVALUATED */,
    int flat) { /** VARIABLE WHERE THE POLYNOM IS GOING TO BE EVALUATED */
  int i, j;     /* AUXILIAR VARIABLES */
  double* p;    /* AUXILIAR VARIABLES */

  p = NULL;
  p = (double*)malloc(sizeof(double) * (N1 + 1));
  if (flat == 1) {
    for (i = 0; i <= N1; i++) {
      for (j = 0; j <= N1; j++) p[j] = p1[i][j];
      p3[i] = ami_polynomial_evaluation(p, N1, z);
    }
  } else {
    for (i = 0; i <= N1; i++) {
      for (j = 0; j <= N1; j++) p[j] = p1[j][i];
      p3[i] = ami_polynomial_evaluation(p, N1, z);
    }
  }
  if (p != NULL) free(p);
}

/**
* \fn double* ami_1v_polynom_multiplication(double *p1,int N1,double *p2,int N2,
double *p3)
* \brief function to multiply polinoms of 1 variable. the result is added to the
output polynom COEFFICIENTs
* \author Luis Alvarez
* \return return the calculation
*/
double* ami_1v_polynom_multiplication(
    double* p1 /** POLYNOM 1 (INPUT) */,
    int N1 /** DEGREE OF POLYNOM 1 (INPUT)*/,
    double* p2 /** POLYNOM 2 (INPUT) */,
    int N2 /** DEGREE OF POLYNOM 2 (INPUT) */,
    double* p3 /** OUTPUT POLYNOM (INPUT-OUTPUT)*/
) {
  int i, j; /* AUXILIAR VARIABLES */

  /* WE MULTIPLY THE POLYNOMS */
  for (i = 0; i <= N1; i++) {
    if (p1[i] != 0) {
      for (j = 0; j <= N2; j++)
        if (p2[j] != 0) p3[i + j] += p1[i] * p2[j];
    }
  }
  return (p3);
}
/**
* \fn void ami_2v_polynom_multiplication(double **p1,int N1,double **p2,int N2,
double **p3 )
* \brief function to multiply polynoms of 2 variables
* \author Luis Alvarez
* \return return void
*/
void ami_2v_polynom_multiplication(
    double** p1 /** POLYNOM 1 (INPUT) */,
    int N1 /** DEGREE OF POLYNOM 1 (INPUT)*/,
    double** p2 /** POLYNOM 2 (INPUT) */,
    int N2 /** DEGREE OF POLYNOM 2 (INPUT) */,
    double** p3 /** OUTPUT POLYNOM (INPUT - OUTPUT)*/
) {
  int i, j, k, l;
  for (i = 0; i <= N1; i++) {
    for (j = 0; j <= N1; j++) {
      if (p1[i][j] != 0) {
        for (k = 0; k <= N2; k++)
          for (l = 0; l <= N2; l++)
            if (p2[k][l] != 0) p3[i + k][j + l] += p1[i][j] * p2[k][l];
      }
    }
  }
}

/**
* \fn int ami_RootCubicPolynomial(double *a,int N,double *x)
* \brief function to compute the real roots of a cubic polynomial. It returns
the number of roots found sorted by magnitud
* \author Luis Alvarez
* \return return 3 if the function finishes properly
*/
int ami_RootCubicPolynomial(
    double* a /** POLYNOMIAL COEFFICIENTS a[0]+a[1]x+a[2]x^2 +... */,
    int N /** DEGREE OF POLYNOMIAL (IT HAS TO BE 3) */,
    double* x /** POLYNOMIAL ROOTS */
) {
  double a1, a2, a3, Q, R, S, T, D, A; /* AUXILIAR VARIABLES */

  if (N != 3 || a[3] == 0) return (-100000);
  a1 = a[2] / a[3];
  a2 = a[1] / a[3];
  a3 = a[0] / a[3];
  Q = (3 * a2 - a1 * a1) / 9.;
  R = (9 * a1 * a2 - 27 * a3 - 2 * a1 * a1 * a1) / 54.;
  D = Q * Q * Q + R * R;

  if (D > 0) {
    S = R + sqrt(D);
    T = R - sqrt(D);
    if (S > 0)
      S = pow(S, (double)1. / 3.);
    else
      S = -pow(-S, (double)1. / 3.);
    if (T > 0)
      T = pow(T, (double)1. / 3.);
    else
      T = -pow(-T, (double)1. / 3.);
    x[0] = S + T - a1 / 3.;
    return (1);
  } else {
    double PI2 = acos(-1.);
    if (Q != 0)
      A = acos(R / sqrt(-Q * Q * Q));
    else
      A = 0;

    Q = 2. * sqrt(-Q);
    x[0] = Q * cos(A / 3.) - a1 / 3.;
    x[1] = Q * cos(A / 3. + 2. * PI2 / 3.) - a1 / 3.;
    x[2] = Q * cos(A / 3 + 4. * PI2 / 3.) - a1 / 3.;

    if (fabs(x[0]) > fabs(x[1])) {
      Q = x[1];
      x[1] = x[0];
      x[0] = Q;
    }
    if (fabs(x[0]) > fabs(x[2])) {
      Q = x[2];
      x[2] = x[0];
      x[0] = Q;
    }
    if (fabs(x[1]) > fabs(x[2])) {
      Q = x[2];
      x[2] = x[1];
      x[1] = Q;
    }

    return (3);
  }
}
/**
 * \fn double ami_polynomial_evaluation(double *a,int Na,double x)
 * \brief function to evaluate a polynom using the Horner algorithm
 * \author Luis Alvarez
 * \return return the evaluation
 */
double ami_polynomial_evaluation(
    double* a /** POLYNOM COEFFICIENT */, int Na /** POLYNOM DEGREE */,
    double x /** POINT WHERE THE POLYNOM IS EVALUATED */
) {
  double sol = a[Na]; /* AUXILIAR VARIABLES */
  int i;              /* AUXILIAR VARIABLES */
  for (i = Na - 1; i > -1; i--) sol = sol * x + a[i];
  return (sol);
}

/**
* \fn int ami_lens_distortion_polynomial_update(double *x, double *y,int Np,
double *a,int Na,double x0,double y0,int k,double *pol)
* \brief function to add the information of a line point sequence to the 4
degree polynomial to compute the lens distortion model
* \author Luis Alvarez
* \return return 0 if the function finishes properly
*/
int ami_lens_distortion_polynomial_update(
    double* x, double* y /** DISTORTED LINE COORDINATES (INPUT)*/,
    int Np /** NUMBER OF POINTS (INPUT)*/,
    double* a /** POLYNOMIAL DEFINING THE LENS DISTORTION MODEL (INPUT) */,
    int Na /** DEGREE OF POLYNOMIAL MODEL FOR LENS DISTORTION */, double x0,
    double y0 /** COORDINATES OF THE IMAGE CENTER */,
    int k /** COEFFICIENT OF THE POLYNOMIAL TO BE UPDATED*/,
    double* pol /** 4 DEGREE POLYNOM TO MINIMIZE (INPUT-OUTPUT) */
) {
  int i, j; /* AUXILIAR VARIABLES */
  double *A, *x2, *y2, *d, x2_m, y2_m, s_xx, s_yy, xA_m, xd_m, yA_m,
      yd_m;                         /* AUXILIAR VARIABLES */
  double pol1[5], pol2[5], pol3[5]; /* AUXILIAR VARIABLES */

  A = x2 = y2 = d = NULL;

  /* WE ALLOCATE MEMORY */
  A = (double*)malloc(sizeof(double) * Np);
  x2 = (double*)malloc(sizeof(double) * Np);
  y2 = (double*)malloc(sizeof(double) * Np);
  d = (double*)malloc(sizeof(double) * Np);

  /* WE COMPUTE THE DISTANCE TO THE IMAGE CENTER */
  for (i = 0; i < Np; i++)
    d[i] = sqrt((x[i] - x0) * (x[i] - x0) + (y[i] - y0) * (y[i] - y0));

  /* WE COMPUTE THE POINT TRANSFORMATION WITH THE CURRENT LENS DISTORTION MODEL
   */
  for (i = 0; i < Np; i++) {
    A[i] = ami_polynomial_evaluation(a, Na, d[i]);
    x2[i] = x0 + (x[i] - x0) * A[i];
    y2[i] = y0 + (y[i] - y0) * A[i];
  }

  /* WE COMPUTE THE DISTANCE POWER k (THE COEFFICIENT OF THE LENS DISTORTION
    MODEL TO BE UPDATED */
  for (i = 0; i < Np; i++) d[i] = pow(d[i], (double)k);

  /* WE COMPUTE THE VARIANCE OF THE TRANSFORMED POINTS */
  x2_m = 0;
  for (i = 0; i < Np; i++) x2_m += x2[i];
  x2_m /= Np;
  s_xx = 0;
  for (i = 0; i < Np; i++) s_xx += (x2[i] - x2_m) * (x2[i] - x2_m);
  s_xx /= Np;
  y2_m = 0;
  for (i = 0; i < Np; i++) y2_m += y2[i];
  y2_m /= Np;
  s_yy = 0;
  for (i = 0; i < Np; i++) s_yy += (y2[i] - y2_m) * (y2[i] - y2_m);
  s_yy /= Np;

  /* WE COMPUTE SOME AVERAGES WE NEED */
  xA_m = 0;
  for (i = 0; i < Np; i++) xA_m += (x[i] - x0) * A[i];
  xA_m /= Np;
  xd_m = 0;
  for (i = 0; i < Np; i++) xd_m += (x[i] - x0) * d[i];
  xd_m /= Np;
  yA_m = 0;
  for (i = 0; i < Np; i++) yA_m += (y[i] - y0) * A[i];
  yA_m /= Np;
  yd_m = 0;
  for (i = 0; i < Np; i++) yd_m += (y[i] - y0) * d[i];
  yd_m /= Np;

  /* WE COMPUTE THE POLYNOMIAL TO MINIMIZE */
  for (i = 0; i < 5; i++) pol1[i] = pol2[i] = pol3[i] = 0;
  for (i = 0; i < Np; i++) {
    pol1[0] += ((x[i] - x0) * A[i] - xA_m) * ((x[i] - x0) * A[i] - xA_m);
    pol1[1] += 2. * ((x[i] - x0) * A[i] - xA_m) * ((x[i] - x0) * d[i] - xd_m);
    pol1[2] += ((x[i] - x0) * d[i] - xd_m) * ((x[i] - x0) * d[i] - xd_m);
    pol2[0] += ((y[i] - y0) * A[i] - yA_m) * ((y[i] - y0) * A[i] - yA_m);
    pol2[1] += 2. * ((y[i] - y0) * A[i] - yA_m) * ((y[i] - y0) * d[i] - yd_m);
    pol2[2] += ((y[i] - y0) * d[i] - yd_m) * ((y[i] - y0) * d[i] - yd_m);
    pol3[0] += ((y[i] - y0) * A[i] - yA_m) * ((x[i] - x0) * A[i] - xA_m);
    pol3[1] += ((y[i] - y0) * A[i] - yA_m) * ((x[i] - x0) * d[i] - xd_m) +
               ((y[i] - y0) * d[i] - yd_m) * ((x[i] - x0) * A[i] - xA_m);
    pol3[2] += ((y[i] - y0) * d[i] - yd_m) * ((x[i] - x0) * d[i] - xd_m);
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      pol[i + j] += (pol1[i] * pol2[j] - pol3[i] * pol3[j]) / 1. /*s_max*/;
    }
  }

  /* WE FREE MEMORY */
  if (A != NULL) free(A);
  if (x2 != NULL) free(x2);
  if (y2 != NULL) free(y2);
  if (d != NULL) free(d);
  return (0);
}

/**
* \fn int ami_lens_distortion_model_update(double *a,int Na,int k,double *pol)
* \brief function to update the lens distortion model by minimizing a 4 degree
polynom
* \author Luis Alvarez
* \return return 0 if the function finishes properly
*/
int ami_lens_distortion_model_update(
    double*
        a /** POLYNOMIAL DEFINING THE LENS DISTORTION MODEL (INPUT-OUTPUT) */,
    int Na /** DEGREE OF POLYNOMIAL MODEL FOR LENS DISTORTION (INPUT)*/,
    int k /** COEFFICIENT OF THE POLYNOMIAL TO BE UPDATED (INPUT)*/,
    double* pol /** 4 DEGREE POLYNOM TO MINIMIZE (INPUT) */,
    double max_radius /** MAX RADIUS TO CHECK THE COMPATIBILITY CRITERIUM OF
                         LENS DISTORTION **/
) {
  int j, i, M = Na;         /* AUXILIAR VARIABLES */
  double *x, *b, *b2, p[3]; /* AUXILIAR VARIABLES */

  x = b = NULL;

  /* WE ALLOCATE MEMORY */
  x = (double*)malloc(sizeof(double) * 3);
  b = (double*)malloc(sizeof(double) * 4);
  b2 = (double*)malloc(sizeof(double) * (Na + 1));
  // WE FILL THE AUXILIARY LENS DISTORTION MODEL
  for (i = 0; i <= Na; i++) b2[i] = a[i];

  b[0] = pol[1];
  b[1] = 2 * pol[2];
  b[2] = 3. * pol[3];
  b[3] = 4. * pol[4];
  M = ami_RootCubicPolynomial(b, 3, x);

  for (i = 0; i < M; i++) p[i] = ami_polynomial_evaluation(pol, 4, x[i]);

  double Error = 1e30;
  j = M;
  for (i = 0; i < M; i++) {
    b2[k] = a[k] + x[i];
    if (test_compatibility_lens_distortion_model(b2, Na, max_radius) == 0) {
      if (p[i] < Error) {
        j = i;
        Error = p[i];
      }
    }
  }

  if (j < M) a[k] += x[j];

  if (x != NULL) free(x);
  if (b != NULL) free(b);
  if (b2 != NULL) free(b2);
  return (0);
}

/**
* \fn double ami_LensDistortionEnergyError(double *x,double *y,int Np,double x0,
double y0,double *a,int Na)
* \brief function to compute the lens distortion energy error (the residual
variance of the point distribution
* \author Luis Alvarez
* \return return the evaluation
*/
double ami_LensDistortionEnergyError(
    double* x, double* y /** ORIGINAL POINT DISTRIBUTION (INPUT)*/,
    int Np /** NUMBER OF POINTS (INPUT)*/, double x0,
    double y0 /** CENTER OF THE IMAGE (INPUT)*/,
    double* a /** Lens Distortion Polynomial model (INPUT)*/,
    int Na /** Degree of Polynomial model (INPUT)*/
) {
  int i;                                               /* AUXILIAR VARIABLES */
  double A, *x2, *y2, d, x2_m, y2_m, s_xx, s_yy, s_xy; /* AUXILIAR VARIABLES */

  x2 = y2 = NULL;

  /* WE ALLOCATE MEMORY */
  x2 = (double*)malloc(sizeof(double) * Np);
  y2 = (double*)malloc(sizeof(double) * Np);

  /* WE COMPUTE THE POINT TRANSFORMATION USING THE LENS DISTORTION MODEL*/
  for (i = 0; i < Np; i++) {
    d = sqrt((x[i] - x0) * (x[i] - x0) + (y[i] - y0) * (y[i] - y0));
    A = ami_polynomial_evaluation(a, Na, d);
    x2[i] = x0 + (x[i] - x0) * A;
    y2[i] = y0 + (y[i] - y0) * A;
  }
  /* WE COMPUTE THE VARIANCE OF THE TRANSFORMED POINTS */
  x2_m = 0;
  for (i = 0; i < Np; i++) x2_m += x2[i];
  x2_m /= Np;
  s_xx = 0;
  for (i = 0; i < Np; i++) s_xx += (x2[i] - x2_m) * (x2[i] - x2_m);
  s_xx /= Np;
  y2_m = 0;
  for (i = 0; i < Np; i++) y2_m += y2[i];
  y2_m /= Np;
  s_yy = 0;
  for (i = 0; i < Np; i++) s_yy += (y2[i] - y2_m) * (y2[i] - y2_m);
  s_yy /= Np;
  s_xy = 0;
  for (i = 0; i < Np; i++) s_xy += (y2[i] - y2_m) * (x2[i] - x2_m);
  s_xy /= Np;

  /* WE FREE MEMORY */
  if (x2 != NULL) free(x2);
  if (y2 != NULL) free(y2);
  return ((s_xx * s_yy - s_xy * s_xy));
}

/**
* \fn double ami_LensDistortionEnergyError_Vmin(double *x,double *y,int Np,
double x0,double y0,double *a,int Na)
* \brief function to compute the lens distortion vmin energy error of the point
distribution
* \author Luis Alvarez
* \return return the evaluation
*/
double ami_LensDistortionEnergyError_Vmin(
    double* x, double* y /**ORIGINAL POINT DISTRIBUTION (INPUT)*/,
    int Np /** NUMBER OF POINTS (INPUT)*/, double x0,
    double y0 /** CENTER OF THE IMAGE (INPUT)*/,
    double* a /** Lens Distortion Polynomial model (INPUT)*/,
    int Na /** Degree of Polynomial model (INPUT)*/
) {
  int i; /* AUXILIAR VARIABLES */
  double A, *x2, *y2, d, x2_m, y2_m, s_xx, s_yy, s_max,
      s_xy; /* AUXILIAR VARIABLES */

  x2 = y2 = NULL;

  /* WE ALLOCATE MEMORY */
  x2 = (double*)malloc(sizeof(double) * Np);
  y2 = (double*)malloc(sizeof(double) * Np);

  /* WE COMPUTE THE POINT TRANSFORMATION USING THE LENS DISTORTION MODEL*/
  for (i = 0; i < Np; i++) {
    d = sqrt((x[i] - x0) * (x[i] - x0) + (y[i] - y0) * (y[i] - y0));
    A = ami_polynomial_evaluation(a, Na, d);
    x2[i] = x0 + (x[i] - x0) * A;
    y2[i] = y0 + (y[i] - y0) * A;
  }
  /* WE COMPUTE THE VARIANCE OF THE TRANSFORMED POINTS */
  x2_m = 0;
  for (i = 0; i < Np; i++) x2_m += x2[i];
  x2_m /= Np;
  s_xx = 0;
  for (i = 0; i < Np; i++) s_xx += (x2[i] - x2_m) * (x2[i] - x2_m);
  s_xx /= Np;
  y2_m = 0;
  for (i = 0; i < Np; i++) y2_m += y2[i];
  y2_m /= Np;
  s_yy = 0;
  for (i = 0; i < Np; i++) s_yy += (y2[i] - y2_m) * (y2[i] - y2_m);
  s_yy /= Np;
  s_xy = 0;
  for (i = 0; i < Np; i++) s_xy += (y2[i] - y2_m) * (x2[i] - x2_m);
  s_xy /= Np;
  s_max = s_xx > s_yy ? s_xx : s_yy;

  /* WE FREE MEMORY */
  if (x2 != NULL) free(x2);
  if (y2 != NULL) free(y2);
  return ((s_xx * s_yy - s_xy * s_xy) / s_max);
}

/**
* \fn double ami_lens_distortion_estimation(double **x,double **y,int Nl,
int *Np,double x0,double y0,double *a,int Na,int k,double alpha)
* \brief function to compute the lens distortion model
* \author Luis Alvarez
* \return return (Error/Nl)
*/
double ami_lens_distortion_estimation(
    double** x,
    double** y /** ORIGINAL COLECCION OF LINES DISTRIBUTION (INPUT)*/,
    int Nl /** NUMBER OF LINES */,
    int* Np /** NUMBER OF POINTS FOR EACH LINE(INPUT)*/, double x0,
    double y0 /** CENTER OF THE IMAGE (INPUT)*/,
    double* a /** Lens Distortion Polynomial model (INPUT-OUTPUT) */,
    int Na /** Degree of Polynomial model (INPUT)*/,
    int k /** COEFFICIENT OF THE LENS DISTORTION POLYNOM MODEL TO BE UPDATED */,
    double alpha /** WEIGHT FOR MINIMIZING THE SQUARE OF THE DISTANCE
                         BETWEEN DISTORTED AND UNDISTORTED POINTS */
    ,
    double max_radius /** MAX RADIUS TO CHECK THE COMPATIBILITY CRITERIUM OF
                         LENS DISTORTION **/
)

{
  double *pol, sum_dd, sum_Ad, d, A, Error = 0; /* AUXILIAR VARIABLES */
  int m, i, j;                                  /* AUXILIAR VARIABLES */
  pol = NULL;

  /* WE ALLOCATE MEMORY */
  pol = (double*)malloc(sizeof(double) * 5);

  for (i = 0; i <= 4; i++) pol[i] = 0.;

  /* WE ADAPT a[0] TO MINIMIZE THE SQUARE OF THE DISTANCE BEWTEEN
    DISTORTED AND UNDISTORDED POINTS */
  if (alpha > 0) {
    sum_dd = sum_Ad = 0;
    for (m = 0; m < Nl; m++) {
      for (i = 0; i < Np[m]; i++) {
        d = sqrt((x[m][i] - x0) * (x[m][i] - x0) +
                 (y[m][i] - y0) * (y[m][i] - y0));
        A = 0;
        for (j = 1; j <= Na; j++) A += a[j] * pow(d, (double)j + 1);
        sum_dd += d * d;
        sum_Ad += A * d;
      }
    }
    a[0] = 1 - sum_Ad / sum_dd;
  }
  /* WE COMPUTE THE LENS DISTORTION MODEL */
  for (i = 0; i < Nl; i++) {
    ami_lens_distortion_polynomial_update(x[i], y[i], Np[i], a, Na, x0, y0, k,
                                          pol);
  }
  ami_lens_distortion_model_update(a, Na, k, pol, max_radius);
  /* WE FREE THE MEMORY */
  if (pol != NULL) free(pol);
  for (i = 0; i < Nl; i++)
    Error += ami_LensDistortionEnergyError(x[i], y[i], Np[i], x0, y0, a, Na);
  return (Error / Nl);
}

/**
* \fn void ami_lens_distortion_zoom_normalization(double **x,double **y,int Nl,
int *Np,double x0,double y0,double *a,int Na)
* \brief function to apply the zoom strategy
* \author Luis Alvarez and Luis Gomez
* \return return void
*/
void ami_lens_distortion_zoom_normalization(
    double** x,
    double** y /** ORIGINAL COLECCION OF LINES DISTRIBUTION (INPUT)*/,
    int Nl /** NUMBER OF LINES */,
    int* Np /** NUMBER OF POINTS FOR EACH LINE(INPUT)*/,
    double* a /** [Lens Distortion Polynomial model (INPUT-OUTPUT);x0,y0]*/,
    int Na /** Degree of Polynomial model (INPUT)*/
) {
  int i, k, m, N = 0;             /* AUXILIAR VARIABLES */
  double Z, d, sum_Ad, A, x0, y0; /* AUXILIAR VARIABLES */

  x0 = a[5];
  y0 = a[6];
  /* WE UPDATE a BY ESTIMATING A ZOOM FACTOR Z */
  sum_Ad = 0;
  for (m = 0; m < Nl; m++) {
    for (i = 0; i < Np[m]; i++) {
      N++;
      d = sqrt((x[m][i] - x0) * (x[m][i] - x0) +
               (y[m][i] - y0) * (y[m][i] - y0));
      A = a[0];
      for (k = 1; k <= Na; k++) A += a[k] * pow(d, (double)k);
      sum_Ad += A * A;
    }
  }
  Z = sqrt((double)N / sum_Ad);
  for (k = 0; k <= Na; k++) a[k] *= Z;
}

/**
 *        GRADIENT PART
 */

/**
* \fn int calculate_points(double *amin,double **points_2D_modified,int N,
int Na,double x0,double y0)
* \brief function to estimate the position of 2D points (pixels) for the actual
lens distortion model
* \author Luis Alvarez and Luis Gomez
* \return return 0 if the function finishes properly
*/
int calculate_points(
    double* amin /** Lens distortion model polynom */,
    double** points_2D_modified /** Cloud of points to be fitted to a line */,
    int N /** Number of points */,
    int Na /** Degree of the lens distortion model */,
    double x0 /** x center of the image */,
    double y0 /** y center of the image */
) {
  double d1, sol; /* AUXILIAR VARIABLES */
  int i, j;       /* AUXILIAR VARIABLES */

  /* Calculate the distance from each point to the center of the image */
  /* WE COMPUTE THE POINT TRANSFORMATION WITH THE CURRENT LENS DISTORTION MODEL
   */
  for (i = 0; i < N; i++) {
    d1 = sqrt(pow(points_2D_modified[i][0] - x0, 2.0) +
              pow(points_2D_modified[i][1] - y0, 2.0));
    sol = amin[Na];
    for (j = Na - 1; j > -1; j--) sol = sol * d1 + amin[j];
    points_2D_modified[i][0] = (points_2D_modified[i][0] - x0) * sol + x0;
    points_2D_modified[i][1] = (points_2D_modified[i][1] - y0) * sol + y0;
  }
  return (0);
}

/**
* \fn double distance_function(double *amin,double **x,double **y,int Nl,
int *Np,int Na,double x0,double y0)
* \brief function to be optimized by the gradient (objective distance function)
* \author Luis Alvarez and Luis Gomez
* \return return the RMS distance for all the points to the line
*/
double distance_function(
    double* solution /** [Lens distortion model polynom;x0,y0] */,
    double** x /** Coordinates of points */,
    double** y /** Coordinates of points */, int Nl /** Number of lines */,
    int* Np /** Number of points/line */,
    int Na /** Degree of the lens distortion model */
) {
  double **points_2D_modified, *amin, sum, f_objective, a, b, c,
      tmp;                /* AUXILIAR VARIABLES */
  double line[3], x0, y0; /* AUXILIAR VARIABLES */
  int i, j;               /* AUXILIAR VARIABLES */
  points_2D_modified = NULL;
  amin = NULL;

  ami_calloc1d(amin, double, Na + 1);
  for (i = 0; i <= Na; i++) amin[i] = solution[i];
  x0 = solution[5];
  y0 = solution[6];

  f_evaluations++;
  f_objective = 0.0;
  for (i = 0; i < Nl; i++) {
    points_2D_modified = (double**)malloc(sizeof(double*) * Np[i]);
    for (j = 0; j < Np[i]; j++) {
      points_2D_modified[j] = (double*)malloc(sizeof(double) * 2);
      points_2D_modified[j][0] = x[i][j];
      points_2D_modified[j][1] = y[i][j];
    }
    calculate_points(amin, points_2D_modified, Np[i], Na, x0, y0);
    ami_line2d_calculation(line, points_2D_modified, Np[i]);
    a = line[1];
    b = line[0];
    c = line[2];
    tmp = pow(a, 2.0) + pow(b, 2.0);
    sum = 0.0;
    for (j = 0; j < Np[i]; j++)
      sum = sum +
            pow(b * points_2D_modified[j][0] + a * points_2D_modified[j][1] + c,
                2.0);

    sum = sum / tmp;
    sum = sum / (double)Np[i];
    f_objective = f_objective + sum;
    for (j = 0; j < Np[i]; j++) {
      free(points_2D_modified[j]);
    }
    free(points_2D_modified);
  }
  f_objective = f_objective / (double)Nl;
  free(amin);
  return (f_objective);
}

/**
* \fn   double find_lambda(double lambda1,double lambda2,double lambda3,
double f_1,double f_2,double f_3,double *amin, double **x,double **y,
int Nl, int *Np,int Na,double *grad_f,int  *change_k,double x0,double y0)
* \brief function to minimize in one dimension (searching lambda)
* \author Luis Alvarez and Luis Gomez
* \return return the value of lambda step
*/
double find_lambda(double lambda1 /** First TTP point */,
                   double lambda2 /** Second TTP point */,
                   double lambda3 /** Third TTP point */,
                   double f_1 /** f_objective(lambda1) */,
                   double f_2 /** f_objective(lambda2) */,
                   double f_3 /** f_objective(lambda3) */,
                   double* amin_copy /** Copy of amin */,
                   double* amin /** Lens distortion model polynom */,
                   double** x /** Coordinates of points */,
                   double** y /** Coordinates of points */,
                   int Nl /** Number of lines */,
                   int* Np /** Number of points/line */,
                   int Na /** Degree of the lens distortion model polynomial */,
                   double* grad_f /** Gradient vector at amin */,
                   int* change_k /** To indicate what variable optimize (1:
                                    optimize, 0: no optimize) */
)

{
  int Naa = 7;                       /* AUXILIAR VARIABLES */
  double f_min, lambda, lambda_tmp1; /* AUXILIAR VARIABLES */
  double lambda_tmp2, f_tmp1 = f_3, f_tmp2 = f_2,
                      f = f_1; /* AUXILIAR VARIABLES */
  int i;                       /* AUXILIAR VARIABLES */
  /* minimum is in (lambda1,lambda2) */
  lambda_tmp1 = (lambda2 + lambda1) / 2.0;
  for (i = 0; i < Naa; i++) {
    if (change_k[i] == 1)
      *(amin_copy + i) = *(amin + i) - lambda_tmp1 * (*(grad_f + i));
  }
  f_tmp1 = distance_function(amin_copy, x, y, Nl, Np, Na);
  if (f_tmp1 < f_1) return (lambda_tmp1);

  lambda_tmp2 = (lambda3 + lambda2) / 2.0;
  for (i = 0; i < Naa; i++) {
    if (change_k[i] == 1)
      *(amin_copy + i) = *(amin + i) - lambda_tmp2 * (*(grad_f + i));
  }
  f_tmp2 = distance_function(amin_copy, x, y, Nl, Np, Na);
  if (f_tmp2 < f_1) return (lambda_tmp2);
  f = f_1;
  do {
    f_min = f;
    lambda = lambda1 + 1e-8;
    for (i = 0; i < Naa; i++) {
      if (change_k[i] == 1)
        *(amin_copy + i) = *(amin + i) - lambda * (*(grad_f + i));
    }
    f = distance_function(amin_copy, x, y, Nl, Np, Na);
  } while (f < f_min);
  return (lambda);
}

/**
* \fn double minimize_cuadratic_polynom((double lambda1,double lambda2,double
lambda3, double f_1,double f_2,double f_3,double *amin, double **x,double **y,
int Nl, int *Np,int Na,double *grad_f,int  *change_k,double x0,double y0)
* \brief function to build and minimize the cuadratic TPP polynom
* \author Luis Alvarez and Luis Gomez
* \return return the minimum of the cuadratic polynom
*/
double minimize_cuadratic_polynom(
    double lambda1 /** First TTP point */,
    double lambda2 /** Second TTP point */,
    double lambda3 /** Third TTP point */,
    double f_1 /** f_objective(lambda1) */,
    double f_2 /** f_objective(lambda2) */,
    double f_3 /** f_objective(lambda3) */,
    double* amin_copy /** Copy of amin */,
    double* amin /** Lens distortion model polynom */,
    double** x /** Coordinates of points */,
    double** y /** Coordinates of points */, int Nl /** Number of lines */,
    int* Np /** Number of points/line */,
    int Na /** Degree of the lens distortion model polynomial */,
    double* grad_f /** Gradient vector at amin */,
    int* change_k /** To indicate what variable optimize (1: optimize, 0: no
                     optimize) */

) {
  double a12, a23, a31, b12, b23, b31, min_lambda; /* AUXILIAR VARIABLES */

  a12 = lambda1 - lambda2;
  a23 = lambda2 - lambda3;
  a31 = lambda3 - lambda1;
  b12 = pow(lambda1, 2.0) - pow(lambda2, 2.0);
  b23 = pow(lambda2, 2.0) - pow(lambda3, 2.0);
  b31 = pow(lambda3, 2.0) - pow(lambda1, 2.0);

  min_lambda = 0.5 * (b23 * f_1 + b31 * f_2 + b12 * f_3);
  min_lambda = min_lambda / (a23 * f_1 + a31 * f_2 + a12 * f_3);
  /* to avoid numerical errors, we check the condition*/
  /* lambda1<min_lambda<lambda3 */
  if ((lambda1 < min_lambda) && (min_lambda < lambda3))
    return (min_lambda);
  else {
    min_lambda =
        find_lambda(lambda1, lambda2, lambda3, f_1, f_2, f_3, amin_copy, amin,
                    x, y, Nl, Np, Na, grad_f, change_k);
    return (min_lambda);
  }
}

/**
* \fn double cuadratic_fitting(double *amin_copy,double *amin,double **x,
double **y,int Nl,int Na,int *Np,double lambda1,double lambda2,double lambda3,
double f_1,double f_2,double f_3,double *grad_f,int *change_k,double x0,
double y0)
* \brief function to find the minimum of the interpolating polynom
* \author Luis Alvarez and Luis Gomez
* \return return 0 if the function finishes properly
*/
double cuadratic_fitting(
    double* amin_copy /** Copy of amin */,
    double* amin /** Lens distortion model polynom */,
    double** x /** Coordinates of points */,
    double** y /** Coordinates of points */, int Nl /** Number of lines */,
    int* Np /** Number of points/line */,
    int Na /** Degree of the lens distortion model polynomial */,
    double lambda1 /** First TTP point */,
    double lambda2 /** Second TTP point */,
    double lambda3 /** Third TTP point */,
    double f_1 /** f_objective(lambda1) */,
    double f_2 /** f_objective(lambda2) */,
    double f_3 /** f_objective(lambda3) */,
    double* grad_f /** Gradient vector at amin */,
    int* change_k /** to indicate what variable optimize (1: optimize, 0: no
                     optimize) */
) {
  double minimo_lambda, f_minimo; /* AUXILIAR VARIABLES */
  double error = 1e100;           /* AUXILIAR VARIABLES */
  int iterations_lambda, i;       /* AUXILIAR VARIABLES */
  int Naa = 7;                    /* AUXILIAR VARIABLES */

  iterations_lambda = 0;
  /* We loop till getting the minimum */
  while (error > tol_lambda) {
    minimo_lambda = minimize_cuadratic_polynom(lambda1, lambda2, lambda3, f_1,
                                               f_2, f_3, amin_copy, amin, x, y,
                                               Nl, Np, Na, grad_f, change_k);
    for (i = 0; i < Naa; i++) {
      if (change_k[i] == 1)
        *(amin_copy + i) = *(amin + i) - minimo_lambda * (*(grad_f + i));
    }
    f_minimo = distance_function(amin_copy, x, y, Nl, Np, Na);
    if (minimo_lambda > lambda2) {
      if (f_minimo > f_2) {
        lambda3 = minimo_lambda;
        f_3 = f_minimo;
      } else {
        lambda1 = lambda2;
        f_1 = f_2;
        lambda2 = minimo_lambda;
        f_2 = f_minimo;
      }
    } else {
      if (f_minimo >= f_2) {
        lambda1 = minimo_lambda;
        f_1 = f_minimo;
      } else {
        lambda3 = lambda2;
        f_3 = f_2;
        lambda2 = minimo_lambda;
        f_2 = f_minimo;
      }
    }
    error = fabs(lambda3 - lambda1);
    if (f_minimo == f_2) lambda2 = lambda2 + tol_lambda;
    iterations_lambda++;
    if (iterations_lambda == max_itera_lambda) return (lambda2);
  }
  return (lambda2);
}

/**
* \fn double minimize_lambda(double *amin,double *amin_copy,double **x,
double **y,int Na,int Nl,int *Np,double *grad_f,double f, int *change_k,
double x0,double y0)
* \brief function Unidimensional lambda miminization
* \author Luis Alvarez and Luis Gomez
* \return return the minimum in the 1D search
*/
double minimize_lambda(
    double* amin /** [Lens distortion model polynom;x0,y0] */,
    double** x /** Coordinates of points */,
    double** y /** Coordinates of points */, int Nl /** Number of lines */,
    int* Np /** Number of points/line */,
    int Na /** Degree of the lens distortion model polynomial */,
    double* grad_f /** Gradient vector at amin */,
    double f /** function value at amin */,
    int* change_k /** To indicate what variable optimize (1: optimize, 0: no
                     optimize) */
) {
  double lambda1, lambda2, lambda3, lambda;     /* AUXILIAR VARIABLES */
  double f_1, f_2, f_3 = 0.0, last_f3, last_f2; /* AUXILIAR VARIABLES */
  double tol_ff = 1.0e-10;                      /* AUXILIAR VARIABLES */
  double* amin_copy;                            /* AUXILIAR VARIABLES */
  int i, Naa;                                   /* AUXILIAR VARIABLES */

  Naa = 7; /* AUXILIAR VARIABLES */
  amin_copy = NULL;
  amin_copy = (double*)malloc(sizeof(double) * (Naa));

  f_1 = f;
  /* search the TTP points */
  lambda1 = 0.0;
  /* search lambda2 */
  lambda2 = fabs(grad_f[2]);
  for (i = 0; i < Naa; i++)
    *(amin_copy + i) = *(amin + i) - lambda2 * (*(grad_f + i));
  f_2 = distance_function(amin_copy, x, y, Nl, Np, Na);

  if (f_2 > f_1) {
    lambda3 = lambda2;
    f_3 = f_2;
    /* search lambda2 by dividing the (lambda1,lambda3) interval */
    lambda2 = lambda3 / 2.0;
    while (1) {
      last_f2 = f_2;
      for (i = 0; i < Naa; i++) {
        if (change_k[i] == 1)
          *(amin_copy + i) = *(amin + i) - lambda2 * (*(grad_f + i));
      }
      f_2 = distance_function(amin_copy, x, y, Nl, Np, Na);
      if (f_2 < f_1) break;
      if (fabs((f_2 - last_f2) / last_f2) <= tol_ff) { /* Avoid the flat zone */
        if (amin_copy != NULL) free(amin_copy);        /*free memory*/
        return (lambda2);
      }
      lambda2 = lambda2 / 2.0;
    }
  } else {
    /* search lambda3 by increasing the (lambda1,lambda2) interval */
    lambda3 = lambda2 * 2.0;
    while (1) {
      last_f3 = f_3;
      for (i = 0; i < Naa; i++) {
        if (change_k[i] == 1)
          *(amin_copy + i) = *(amin + i) - lambda3 * (*(grad_f + i));
      }
      f_3 = distance_function(amin_copy, x, y, Nl, Np, Na);
      if (f_3 > f_2) break;
      if (fabs((f_3 - last_f3) / last_f3) <= tol_ff) { /* Avoid the flat zone */
        if (amin_copy != NULL) free(amin_copy);        /* Free memory */
        return (lambda3);                              /* Avoid the flat zone */
      }
      lambda3 = 2 * lambda3;
    }
  }
  /* We have the points satisfying the TTP condition
  lambda1,f_1;lambda_2,f_2;lambda3,f_3
  minimize the cuadratic polynom */
  lambda = cuadratic_fitting(amin_copy, amin, x, y, Nl, Np, Na, lambda1,
                             lambda2, lambda3, f_1, f_2, f_3, grad_f, change_k);
  if (amin_copy != NULL) free(amin_copy);
  return (lambda);
}

/**
* \fn double  gradient_method(double  *amin,double  **x,double  **y,int Nl,
int *Np,int Na,int *change_k, double x0,double y0,int zoom)
* \brief function to minimize the distance function by gradient
* \author Luis Alvarez and Luis Gomez
* \return return the vector solution (distortion coefficients)
*/
double gradient_method(
    double* solution /** [Lens distortion model polynom;x0,y0] */,
    double** x /** Coordinates of points */,
    double** y /** Coordinates of points */, int Nl /** Number of lines */,
    int* Np /** Number of points/line */,
    int Na /** Degree of the lens distortion model polynomial */,
    int* change_k /** to indicate what variable optimize (1: optimize, 0: no
                     optimize) */
    ,
    int zoom /** Zoom strategy */
) {
  double *grad_f, f_objective, last_f = 1.0e100; /* AUXILIAR VARIABLES */
  double kk, lambda;                             /* AUXILIAR VARIABLES */
  int i;                                         /* AUXILIAR VARIABLES */

  grad_f = NULL;
  ami_calloc1d(grad_f, double, 7);

  f_objective = distance_function(solution, x, y, Nl, Np, Na);
  while (fabs((f_objective - last_f) / last_f) > tol_f) {
    last_f = f_objective;
    for (i = 0; i < 7; i++) {
      /* Move along each axis and incremental step to calculate the derivative
       */
      if (change_k[i] == 1) {
        kk = solution[i];
        solution[i] = kk + delta;
        grad_f[i] =
            (distance_function(solution, x, y, Nl, Np, Na) - f_objective) /
            delta;
        solution[i] = kk;
      }
    }
    /* Activate to stop the gradient when the gradient_norm<tol_norma
    gradient_norm=0.0; for(i=0;i<7;i++)
    gradient_norm=gradient_norm+pow(grad_f[i],2.0);
    gradient_norm=sqrt(gradient_norm);      if(gradient_norm<=tol_norma) break;
  */
    lambda = minimize_lambda(solution, x, y, Nl, Np, Na, grad_f, f_objective,
                             change_k);
    for (i = 0; i < 7; i++)
      if (change_k[i] == 1)
        *(solution + i) = *(solution + i) - lambda * (*(grad_f + i));
    n_iterations++;
    f_objective = distance_function(solution, x, y, Nl, Np, Na);
    /* Activate to have the trace of the execution */
    // printf("\nIteracion=%d\tf=%1.18f\t|grad(f)|=%1.18f",n_iteraciones,f_objective,gradient_norm);
    // for(i=0;i<7;i++) printf("\n(in gradient) solution[%d]=%f",i,solution[i]);
    // system("pause");
    if (n_iterations == max_itera) break;
  }
  /** ZOOM UPDATE amin[0] */
  if (zoom == 1)
    ami_lens_distortion_zoom_normalization(x, y, Nl, Np, solution, Na);
  if (grad_f != NULL) free(grad_f);
  return (0);
}

/**
* \fn double  optimize(double  *amin,double **x, double **y,double **xx,
double **yy,int Nl,int *Np,int Na, int *change_k,double x0,double y0,
double factor_n,int zoom, FILE *fp1)
* \brief function to execute the gradient method and save information to a file
* \author Luis Alvarez and Luis Gomez
* \return return 1 if the function finishes properly
*/
int optimize(double* solution /** [Lens distortion model polynom; x0,y0] */,
             double** x /** Coordinates of points (normalized) */,
             double** y /** Coordinates of points (normalized) */,
             double** xx /** Coordinates of points */,
             double** yy /** Coordinates of points */,
             int Nl /** Number of lines */,
             int* Np /** Number of points/line */,
             int Na /** Degree of the lens distortion model polynomial */,
             double factor_n /** Factor to normalize coordinates */,
             int zoom /** Zoom strategy */,
             FILE* fp1 /** Pointer to the output file */,
             int optimize_center /** To optimize the center of distortion*/
) {
  int starttime, stoptime, i; /* AUXILIAR VARIABLES */
  double paso, Emin, Vmin, D; /* AUXILIAR VARIABLES */
  double timeused, x0, y0, x00, y00, *amin,
      factor_n_new; /* AUXILIAR VARIABLES */
  int* change_k;    /* TO  INDICATE WHAT VARIABLE IS
                        GOING TO BE OPTIMIZED BY GRADIENT METHOD */

  change_k = NULL;
  amin = NULL;

  x00 = solution[5];
  y00 = solution[6]; /* WE COPY THE ORIGINAL CENTER OF DISTORTION (pixels) */
  f_evaluations = 0;
  n_iterations = 0;

  solution[5] = 0.0;
  solution[6] = 0.0;              /* TO CALCULATE IN NORMALIZED COORDINATES */
  ami_calloc1d(change_k, int, 7); /* BY DEFAULT, INITIALIZED TO 0 */
  change_k[2] = 1;
  change_k[4] = 1; /* OPTIMIZED ONLY K2 AND K4 */
  starttime = clock();
  /* GRADIENT METHOD WORKS WITH NORMALIZED UNITS */
  if (optimize_center == 1) {
    change_k[5] = 1;
    change_k[6] = 1; /*OPTIMIZE THE CENTER OF DISTORTION */
    gradient_method(solution, x, y, Nl, Np, Na, change_k, zoom);
  } else
    gradient_method(solution, x, y, Nl, Np, Na, change_k, zoom);
  stoptime = clock();
  timeused = difftime(stoptime, starttime);
  x0 = solution[5];
  y0 = solution[6]; /* WE RECOVER THE CENTER OF DISTORTION, OPTIMIZED OR NOT */

  /* Final solution is in solution (distortion parameters) and it is normalized
   */
  /* We undo the un-normalization to have the solution in pixels */
  if (optimize_center ==
      1) { /* WE CALCULATE THE NEW CENTER OF DISTORTION IN PIXELS */
    x0 = x0 + x00;
    y0 = y0 + y00;
  } else {
    x0 = x00;
    y0 = y00;
  }

  /* WE TRANSFORM THE SOLUTION IN NORMALIZED COORDINATES TO PIXEL UNITS */
  if (optimize_center == 0) {
    paso = 1.0;
    for (i = 0; i <= Na; i++) {
      solution[i] = solution[i] * paso;
      paso /= factor_n;
    }
  } else {
    factor_n_new = calculate_factor_n(xx, yy, Nl, Np, x0, y0);
    paso = 1.0;
    for (i = 0; i <= Na; i++) {
      solution[i] = solution[i] * paso;
      paso /= factor_n_new;
    }
  }
  solution[5] = x0;
  solution[6] = y0; /* WE UPDATE THE SOLUTION (CENTER OF DISTORTION) */
  amin = NULL;
  ami_calloc1d(amin, double, Na + 1); /* WE RECOVER THE amin SOLUTION */
  for (i = 0; i <= Na; i++)
    amin[i] = solution[i]; /* WE UPDATE THE SOLUTION (ldm) */
  /* We get the final solution and print into a file */
  fprintf(fp1, "\nMODIFIED VARIABLES THROUGH OPTIMIZATION:");
  for (i = 0; i <= Na; i++)
    if (change_k[i] == 1) fprintf(fp1, "\tk[%d]", i);
  if (optimize_center == 1) fprintf(fp1, ", (x0,y0)");
  Emin = 0.0;
  for (i = 0; i < Nl; i++)
    Emin +=
        ami_LensDistortionEnergyError(xx[i], yy[i], Np[i], x0, y0, amin, Na);
  Emin = Emin / (double)Nl;
  Vmin = 0.0;
  for (i = 0; i < Nl; i++)
    Vmin += ami_LensDistortionEnergyError_Vmin(xx[i], yy[i], Np[i], x0, y0,
                                               amin, Na);
  Vmin = Vmin / (double)Nl;
  D = distance_function(solution, xx, yy, Nl, Np, Na);
  fprintf(fp1, "\n\n(Emin, Vmin, D) = (%1.4e, %1.4e, %1.4e)", Emin, Vmin, D);
  fprintf(fp1, "\n\nDistortion parameters:\n");
  for (i = 0; i <= Na; i++) fprintf(fp1, "k[%d] = %1.15e\n", i, amin[i]);
  fprintf(fp1, "\nCenter of distortion (x0,y0) = (%f, %f)\n", x0, y0);
  fprintf(fp1, "\nNumber of gradient iterations= %d", n_iterations);
  fprintf(fp1, "\nCPU_time = %f (seconds)", timeused / (double)CLOCKS_PER_SEC);
  fprintf(fp1, "\nf_evaluations = %d", f_evaluations);
  if (change_k != NULL) free(change_k); /* FREE THE MEMORY */
  return (1);
}

/**
* \fn algebraic_method_pre_gradient(int Nl,int *Np,double *a,double **x,double
**y, double **xx,double **yy,double factor_n,int zoom,int *change_k)
* \brief function to calculate the solution for the gradient method applied from
the algebraic method solution
* \author Luis Alvarez and Luis Gomez
* \return return 1 if the function finishes properly
*/
int algebraic_method_pre_gradient(
    int Nl /** Number of lines */, int* Np /** Number of points/line */,
    double* a /** Lens distortion model polynom */,
    double** x /** Coordinates of points (normalized) */,
    double** y /** Coordinates of points (normalized) */,
    double** xx /** Coordinates of points */,
    double** yy /** Coordinates of points  */,
    double factor_n /** Factor to normalize coordinates */,
    int zoom /** Zoom strategy */, FILE* fp1 /** Pointer to the output file*/,
    int optimize_center /** To optimize the center of distortion*/,
    double max_radius /** MAXIMUM RADIAL DISTANCE IN PHOTO */
) {
  int Na = 4; /* DEGREE OF THE POLYNOM */

  if (zoom == 0) {
    fprintf(fp1,
            "******************************************************************"
            "*******\n");
    fprintf(fp1,
            "            ALGEBRAIC METHOD + GRADIENT: WITHOUT ZOOM (Na = 4)    "
            "         \n");
    fprintf(fp1,
            "******************************************************************"
            "*******");
  }

  else {
    fprintf(fp1,
            "\n****************************************************************"
            "*********\n");
    fprintf(fp1,
            "*          ALGEBRAIC METHOD + GRADIENT: WITH ZOOM (Na = 4)        "
            "       \n");
    fprintf(fp1,
            "******************************************************************"
            "*******");
  }

  double x0, y0, x00, y00;
  x0 = a[5];
  y0 = a[6]; /* WE CAPTURE THE CENTER OF DISTORTION */
  x00 = x0;
  y00 = y0; /* WE COPY THE ORIGINAL CENTER OF DISTORTION */

  fprintf(fp1, "\n2 parameters, 1 iteration, variables updated:\t2, 4");

  if (optimize_center == 0) {  // IT WORKS FROM THE TRIVIAL SOLUTION
    a[0] = 1.0;                /* trivial solution */
    for (int i = 1; i <= Na; i++) a[i] = 0.0;
    /* WE RUN A SAFE PREVIOUS ITERATION TO AVOID CONVERGENCE PROBLEMS*/
    ami_lens_distortion_estimation(x, y, Nl, Np, (double)0., (double)0., a, Na,
                                   2, (double)0., max_radius);
    ami_lens_distortion_estimation(x, y, Nl, Np, (double)0., (double)0., a, Na,
                                   4, (double)0., max_radius);
    /* WE RUN THE ALGEBRAIC METHOD FOR BOTH PARAMETERS IN ONE ITERATION */
    ami_lens_distortion_estimation_2v(x, y, Nl, Np, (double)0., (double)0., a,
                                      Na, 2, 4, (double)0., max_radius);

    /* WE SET THE ORIGINAL CENTER OF DISTORTION */
    a[5] = x00;
    a[6] = y00;

    /* APPLY THE GRADIENT METHOD FROM THE ALGEBRAIC SOLUTION */
    optimize(a, x, y, xx, yy, Nl, Np, Na, factor_n, zoom, fp1, optimize_center);
  }

  else { /* A SOLUTION -PIXELS- HAS BEEN CALCULATED WHEN SEARCHING FOR THE
      BEST CENTER OF DISTORTION*/
    /* NORMALIZE COORDINATES */
    double paso = 1.0;
    for (int i = 0; i <= Na; i++) {
      a[i] = a[i] * paso;
      paso /= factor_n;
    }
    a[5] = x00; /* WE COPY THE ORIGINAL CENTER OF DISTORTION */
    a[6] = y00; /* WE COPY THE ORIGINAL CENTER OF DISTORTION */
    /* APPLY THE GRADIENT METHOD FROM THE SOLUTION */
    optimize(a, x, y, xx, yy, Nl, Np, Na, factor_n, zoom, fp1, optimize_center);
  }
  return (1);
}

/**
* \fn trivial_solution(int Nl,int *Np,int Na,double *a,double x0,double y0,
double **x,double **y,double **xx,double **yy,double factor_n,FILE *fp1)
* \brief function to calculate the trivial solution
* \author Luis Alvarez and Luis Gomez
* \return return 1 if the function finishes properly
*/

int trivial_solution(
    int Nl /** Number of lines */, int* Np /** Number of points/line */,
    double* a /** Lens distortion model polynom */,
    double** xx /** Coordinates of points */,
    double** yy /** Coordinates of points */,
    double factor_n /** Factor to normalize coordinates */,
    FILE* fp1 /** Pointer to the output file */,
    double* trivial /** Trivial Emin,Vmin,Dmin values */,
    int optimize_center /** To optimize the center of distortion*/
) {
  int starttime, stoptime, i, m, Na;
  double *amin, Emin, Vmin = factor_n, D, x0, y0;
  double timeused;

  /* WE CAPTURE THE CENTER OF DISTORTION */
  x0 = a[5];
  y0 = a[6];

  /* WE DEFINE THE GRADE OF THE LENS DISTORTION POLYNOM (MAXIMUM GRADE) */
  /* WE KEEP THE ORIGINAL FORMAT OF THE ALGEBRAIC FUNCTIONS */
  Na = 4;

  ami_calloc1d(amin, double, Na + 1);
  for (i = 0; i <= Na; i++) amin[i] = a[i];

  fprintf(fp1,
          "********************************************************************"
          "********");
  fprintf(fp1,
          "\n                     LENS DISTORTION MODEL: OUTPUT FILE           "
          "          ");
  fprintf(fp1,
          "\n      CALCULATED USING NORMALIZED COORDINATES (SOLUTION IN "
          "PIXELS)              ");
  fprintf(
      fp1,
      "\n\nALGEBRAIC METHOD:    IT DOES NOT OPTIMIZE THE CENTER OF DISTORTION");
  fprintf(fp1,
          "\nGRADIENT METHOD:     IT IMPROVES THE PREVIOUS ALGEBRAIC SOLUTION");
  fprintf(fp1,
          "\n                                                                  "
          "             ");
  fprintf(fp1,
          "\n            THE CENTER OF DISTORTION IS OPTIMIZED IF INDICATED");
  fprintf(fp1,
          "\n******************************************************************"
          "**********\n");

  if (optimize_center == 1)
    fprintf(fp1, "\n\t\tThe center of distortion is going to be optimized.\n");
  else
    fprintf(fp1,
            "\n\t\tThe center of distortion is not going to be optimized.\n");
  fprintf(fp1,
          "\n******************************************************************"
          "**********\n");

  /* WE DO THE CALCULATION IN NORMALIZED COORDINATES AND TRANSFORM TO PIXELS AT
   * THE END */
  /* SOLUTION AND DISTORTION PARAMETERS: IN PIXELS */
  /* ////////////////////////////////////////////////////////////////////////////////////////////
   */
  /* ******************************WE MEASURE CPU TIME FOR CALCULATIONS
   * ONLY********************* */
  /* ////////////////////////////////////////////////////////////////////////////////////////////
   */
  /* WE COMPUTE THE DISTORTION MODEL */
  /* WE ESTIMATE THE Emin,Vmin IN PIXELS */
  fprintf(fp1,
          "                      TRIVIAL SOLUTION (Na = 4):                    "
          "            ");
  fprintf(fp1,
          "\n******************************************************************"
          "**********\n");
  fprintf(fp1, "Center of distortion (x0,y0) = (%f,%f)", x0, y0);
  starttime = clock();
  Emin = 0;
  for (m = 0; m < Nl; m++)
    Emin +=
        ami_LensDistortionEnergyError(xx[m], yy[m], Np[m], x0, y0, amin, Na);
  Emin = Emin / (double)Nl;
  Vmin = 0;
  for (m = 0; m < Nl; m++)
    Vmin += ami_LensDistortionEnergyError_Vmin(xx[m], yy[m], Np[m], x0, y0,
                                               amin, Na);
  Vmin = Vmin / (double)Nl;
  stoptime = clock();
  timeused = difftime(stoptime, starttime);
  D = distance_function(a, xx, yy, Nl, Np, Na);
  fprintf(fp1, "\n(Emin, Vmin, D) = (%1.4e, %1.4e, %1.4e)", Emin, Vmin, D);
  fprintf(fp1, "\nDistortion parameters:\n");
  for (i = 0; i <= Na; i++) fprintf(fp1, "k[%d] = %1.15e\n", i, a[i]);
  fprintf(fp1, "\nCPU_time=%f (seconds)", timeused / (double)CLOCKS_PER_SEC);
  fprintf(fp1,
          " \n*****************************************************************"
          "********\n\n\n");
  /* SAVE THE TRIVIAL VALUES TO CALCULATE THE % IMPROVEMENT */
  trivial[0] = Emin;
  trivial[1] = Vmin;
  trivial[2] = D;
  free(amin);
  return (1);
}

/**
 * \fn int search_for_best_center(int Nl,int *Np,int Na,double *a,double
 * **x,double **y) \brief function to calculate the best center of distortion
 * (using a 30x30 patch) \author Luis Alvarez and Luis Gomez \return return 1 if
 * the function finishes properly
 */

int search_for_best_center(
    int Nl /** Number of lines */, int* Np /** Number of points/line */,
    double* a /** Lens distortion model polynom */,
    double** xx /** Coordinates of points (pixels) */,
    double** yy /** Coordinates of points (pixels) */,
    int width /** Image size: width (pixels) */,
    int height /** Image size: height (pixels) */,
    double max_radius /** MAXIMUM RADIAL DISTANCE IN PHOTO */
) {
  int Na = 4; /* WE DEFINE THE GRADE OF THE LENS DISTORTION POLYNOM (MAXIMUM
                 GRADE) */

  /* WE CALCULATE THE ALGEBRAIC SOLUTION FOR THE GIVEN CENTER OF DISTORTION */
  /* WE WORK USING PIXELS UNITS */
  double initial_Emin = 0.0;
  int x_c = floor(a[5]); /* WE CAPTURE THE INDICATED CENTER OF DISTORTION (X) */
  int y_c = floor(a[6]); /* WE CAPTURE THE INDICATED CENTER OF DISTORTION (Y) */
  int best_x_c,
      best_y_c; /* AUXILIAR VARIABLE TO STORE THE BEST SOLUTION FOUND */

  float Emin, timeused;
  float** aux_image;                         /* AUXILIAR VARIABLES */
  int starttime, stoptime, n_iterations = 0; /* AUXILIAR VARIABLES */

  aux_image = NULL;
  // TO SPEED-UP CALCULATION OF OPTIMIZED CENTER OF DISTORTION,
  // WE AVOID TO RECALCULATE ENERGY FUNCTION AT PREVIOUS
  // CALCULATED POSITIONS WITHIN THE IMAGE
  /* WE ALLOCATE MEMORY FOR AUXIlIAR IMAGE MATRIX */
  aux_image = (float**)malloc(sizeof(float*) * width);
  for (int i = 0; i < (int)width; i++)
    aux_image[i] = (float*)malloc(sizeof(float) * height);
  // WE INITIALIZE THE aux_image MATRIX TO -1.0
  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++) aux_image[i][j] = -1.0;

  /* WE RUN A SAFE PREVIOUS ITERATION TO AVOID CONVERGENCE PROBLEMS*/
  initial_Emin =
      ami_lens_distortion_estimation(xx, yy, Nl, Np, (double)x_c, (double)y_c,
                                     a, Na, 2, (double)0., max_radius);
  initial_Emin =
      ami_lens_distortion_estimation(xx, yy, Nl, Np, (double)x_c, (double)y_c,
                                     a, Na, 4, (double)0., max_radius);
  /* WE RUN THE ALGEBRAIC METHOD FOR BOTH PARAMETERS IN ONE ITERATION */
  initial_Emin = ami_lens_distortion_estimation_2v(xx, yy, Nl, Np, (double)x_c,
                                                   (double)y_c, a, Na, 2, 4,
                                                   (double)0., max_radius);

  /* initial_Emin has the algebraic solution for the given center of distortion.
     The corresponding solution is in vector a */
  /* This initial_Emin solution is the initial one to compare with, for the
   * patch search */
  /* we scan iteratively the image looking for the best center of distortion */
  float last_Emin = 1e100;
  best_x_c = x_c;
  best_y_c = y_c;
  int patch_half = floor((double)patch_size / 2.0);
  starttime = clock();
  while (fabs(initial_Emin - last_Emin) > tol_f) {
    last_Emin = initial_Emin;
    n_iterations++;
    int lim_inf_x = x_c - patch_half;
    int lim_sup_x = x_c + patch_half;
    int lim_inf_y = y_c - patch_half;
    int lim_sup_y = y_c + patch_half;
    //#pragma omp parallel for
    /* we scan the rectangular patch: pixel precission, subpixel is reached
     * through gradient */
    for (int i = lim_inf_x; i <= lim_sup_x; i++) {
      if (i >= 0 &&
          i <= width) {  // we check that we are inside the image (x-axis)
        for (int j = lim_inf_y; j <= lim_sup_y; j++) {
          if (j >= 0 &&
              j <= height) {  // we check that we are inside the image (y-axis)
            if (aux_image[i][j] ==
                -1.0) {  // we check if this value is already calculated
              a[2] = 0.0;
              a[4] = 0.0;  // we reset the a vector
              /* REMARK: ACTIVATE THE NEXT TWO LINES IF THERE ARE CONVERGENCE
               * PROBLEMS */
              Emin = ami_lens_distortion_estimation(xx, yy, Nl, Np, (double)i,
                                                    (double)j, a, Na, 2,
                                                    (double)0., max_radius);
              Emin = ami_lens_distortion_estimation(xx, yy, Nl, Np, (double)i,
                                                    (double)j, a, Na, 4,
                                                    (double)0., max_radius);
              Emin = ami_lens_distortion_estimation_2v(
                  xx, yy, Nl, Np, (double)i, (double)j, a, Na, 2, 4, (double)0.,
                  max_radius);
              aux_image[i][j] =
                  Emin;  // we save the value for possible use in new iterations
            } else
              Emin = aux_image[i][j];
            if (Emin < initial_Emin) {
              initial_Emin = Emin; /* We save the best solution */
              best_x_c = i;
              best_y_c = j; /* We save the best center of distortion found */
            }
          }
        }
      }
    }
    x_c = best_x_c;
    y_c = best_y_c;
    printf("\nitera=%d,(x_c,y_c)=(%d,%d), Emin=%1.7e, last_Emin=%1.7e",
           n_iterations, best_x_c, best_y_c, initial_Emin, last_Emin);
    if (n_iterations == max_itera_patch) break;
  }
  stoptime = clock();
  timeused = difftime(stoptime, starttime);
  a[5] = best_x_c;
  a[6] = best_y_c; /* We update the new distortion center coordinates */
  printf(
      "\n**********************************************************************"
      "******");
  printf("\nCPU time Optimizing Center of Distortion=%f (seconds)",
         timeused / (double)CLOCKS_PER_SEC);
  printf(
      "\n**********************************************************************"
      "******");
  // we free the memory
  if (aux_image != NULL) {
    for (int j = 0; j < height; j++) {
      if (aux_image[j] != NULL) free(aux_image[j]);
    }
    free(aux_image);
  }
  return (1);
}

/**
 * \fn int read_line_primitives(char filename[300],int *Nl,int **Np,double
 * ***x,double ***y) \brief function to read point line primitives \author Luis
 * Alvarez
 */
int read_line_primitives(char filename[300] /** INPUT FILE NAME */,
                         int* Nl /** OUTPUT NUMBER OF LINES */,
                         int** Np /** OUTPUT NUMBER OF POINTS FOR EACH LINE */,
                         double*** x /** OUTPUT POINT X COORDINATES  */,
                         double*** y /** OUTPUT POINT X COORDINATES  */) {
  FILE* fp_1;
  int i, j, k;
  float tmp1, tmp2;
  if (!(fp_1 = fopen(filename, "r"))) {
    printf("Error while opening the file %s\n", filename);
    return (1);
  }

  fscanf(fp_1, "%d\n", Nl); /*get number of lines*/
  if ((*Nl) <= 0) {
    printf("Error: The number of lines is 0 %s\n", filename);
    return (2);
  }
  *Np = (int*)malloc(sizeof(int) * (*Nl));
  *x = (double**)malloc(sizeof(double*) * (*Nl)); /* x pixels coordinates */
  *y = (double**)malloc(sizeof(double*) * (*Nl)); /* y pixels coordinates */

  printf("Nl=%d\n", *Nl);
  for (i = 0; i < (*Nl); i++) {
    fscanf(fp_1, "%d", &k); /*get number of points for the "i" line  */
    (*Np)[i] = k;
    printf("Np[%d]=%d\n", i, (*Np)[i]);
    (*x)[i] = (double*)malloc(sizeof(double) * k);
    (*y)[i] = (double*)malloc(sizeof(double) * k);
    for (j = 0; j < k; j++) {
      fscanf(fp_1, "%f %f\n", &tmp1, &tmp2);
      (*x)[i][j] = tmp1;
      (*y)[i][j] = tmp2;
      printf("x[%d][%d]=%f x[%d][%d]=%f\n", i, j, (*x)[i][j], i, j, (*y)[i][j]);
    }
  }
  fclose(fp_1);
  return (0);
}

/**
 * \fn double calculate_factor_n(double **x, double **y,int Nl, int *Np, double
 * x0, double y0) \brief function to calculate the factor_n, needed for
 * transforming k (lens distortion model) from normalized coordinates to pixels
 * \return the factor_n value (double)
 * \author Luis Alvarez & Luis Gomez
 */
double calculate_factor_n(double** xx /** x-Coordinates of points (pixels) */,
                          double** yy /** y-Coordinates of points (pixels) */,
                          int Nl /** Number of lines */,
                          int* Np /** Number of points/line */,
                          double x0 /** x center of the image (pixels) */,
                          double y0 /** y center of the image (pixels) */
) {
  double **x_aux, **y_aux; /* AUXILIAR MATRICES TO COPY THE X,Y COORDINATES */
  double factor_n = 0.0;   /* AUXILIAR VARIABLE */
  int i, m, cont = 0;      /* AUXILIAR VARIABLE */

  /* WE ALLOCATE MEMORY FOR AUXILARY POINTS AND WE NORMALIZE ORIGINAL POINTS */
  x_aux = (double**)malloc(sizeof(double*) * Nl); /*  x pixels coordinates */
  y_aux = (double**)malloc(sizeof(double*) * Nl); /*  y pixels coordinates */
  for (i = 0; i < Nl; i++) {
    x_aux[i] = (double*)malloc(sizeof(double) * Np[i]);
    y_aux[i] = (double*)malloc(sizeof(double) * Np[i]);
  }

  for (m = 0; m < Nl; m++) {
    for (i = 0; i < Np[m]; i++) {
      cont++;
      x_aux[m][i] = xx[m][i];
      y_aux[m][i] = yy[m][i];
      x_aux[m][i] -= x0;
      y_aux[m][i] -= y0;
      factor_n += x_aux[m][i] * x_aux[m][i] + y_aux[m][i] * y_aux[m][i];
    }
  }
  factor_n = sqrt(factor_n / cont);
  /* WE FREE THE MEMORY */
  if (x_aux != NULL) {
    for (i = 0; i < Nl; i++) {
      if (x_aux[i] != NULL) free(x_aux[i]);
    }
    free(x_aux);
  }
  if (y_aux != NULL) {
    for (i = 0; i < Nl; i++) {
      if (y_aux[i] != NULL) free(y_aux[i]);
    }
    free(y_aux);
  }

  return (factor_n);
}

/**
 * \fn void ami_lens_distortion_model_evaluation(double *a,int Na, double xc,
     double yc,double x_input,double y_input,double *x_output,double *y_output)
 *  \brief  COMPUTE THE LENS DISTORTION MODEL IN A POINT
 *  \pre Any parameter can be null.
 *  \pre Na has to be positive.
 *  \param [in] a INPUT POLYNOMIAL DISTORTION MODEL
 *  \param [in] Na INPUT DEGREE OF POLYNOMIAL DISTORTION MODEL
 *  \param [in] xc,yc INPUT CENTER OF DISTORTION
 *  \param [in] x_input,y_input INPUT POINT
 *  \param [out] x_output,y_output  OUTPUT UNDISTORTED POINT
 * \author Luis Alvarez
 */
void ami_lens_distortion_model_evaluation(
    double* a,             // INPUT POLYNOMIAL DISTORTION MODEL
    int Na,                // INPUT DEGREE OF POLYNOMIAL DISTORTION MODEL
    double xc, double yc,  // INPUT CENTER OF DISTORTION
    double x_input, double y_input,     // INPUT POINT
    double* x_output, double* y_output  // OUTPUT UNDISTORTED POINT
) {
  double norm =
      sqrt((x_input - xc) * (x_input - xc) + (y_input - yc) * (y_input - yc));
  double A = ami_polynomial_evaluation(a, Na, norm);
  *x_output = xc + (x_input - xc) * A;
  *y_output = yc + (y_input - yc) * A;
}

/**
 * \fn double ami_inverse_lens_distortion_newton_raphson(double x,double y,
 double x0,double y0, double *xt,double *yt, double *a, int Na)
 *  \brief  COMPUTE THE INVERSE OF LENS DISTORTION MODEL IN A POINT USING
 NEWTON-RAPHSON
 *  \param [in] x,y POINT TO INVERSE (INPUT)
 *  \param [in] x0,y0 LENS DISTORTION MODEL CENTER (INPUT)
 *  \param [out] *xt,*yt UNDISTORTED POINT (INVERSE POINT TRANSFORMED) (OUTPUT)
 *  \param [in] *a LENS DISTORTION MODEL COEFFICIENTS
 *  \param [in] Na DEGREE OF THE LENS DISTORTION MODEL POLYNOM
 * \author Luis Alvarez
 */
double ami_inverse_lens_distortion_newton_raphson(
    double x, double y,     /* POINT TO INVERSE (INPUT)*/
    double x0, double y0,   /* CENTER OF THE IMAGE (INPUT)*/
    double* xt, double* yt, /* INVERVE POINT TRANSFORMED (OUTPUT) */
    double* a,              /* LENS DISTORTION MODEL POLYNOM */
    int Na) {               /* DEGREE OF THE LENS DISTORTION MODEL POLYNOM */
  if (a[Na] == 0.) return (-1);

  // AUXILIARY VARIABLES
  double *b, *b2, root2, root;

  // WE ALLOCATE MEMORY
  b = (double*)malloc(sizeof(double) * (Na + 2));
  b2 = (double*)malloc(sizeof(double) * (Na + 2));

  // WE DEFINE THE POLYNOM WE NEED TO COMPUTE ROOTS AND THE DERIVATIVE */
  for (int i = 1; i < (Na + 2); i++) {
    b[i] = a[i - 1];
  }
  for (int i = 0; i < (Na + 1); i++) {
    b2[i] = a[i] * (i + 1);
  }
  root = sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0));
  if (root == 0) {
    *xt = x;
    *yt = y;
    free(b);
    free(b2);
    return (0.);
  }
  b[0] = -root;
  // NEWTON-RAPHSON TO COMPUTE THE POLYNOMIAL ROOT
  for (int k = 0; k < 10000; k++) {
    double pol_eval = ami_polynomial_evaluation(b, Na + 1, root);
    double pol_der = ami_polynomial_evaluation(b2, Na, root);
    if (pol_der == 0.) break;
    root2 = root - pol_eval / pol_der;
    if (fabs(root - root2) < (fabs(root) * 1e-8)) {
      root = root2;
      // printf("k=%d   ",k);
      break;
    }
    root = root2;
  }

  *xt = x0 + (x - x0) * root / (-b[0]);
  *yt = y0 + (y - y0) * root / (-b[0]);

  // WE CHECK THE RESULT
  /* double x_output,y_output;
   ami_lens_distortion_model_evaluation(a,Na,x0,y0,*xt,*yt,&x_output,&y_output);
   printf("root=%lf   (x,y)=(%1.0lf,%1.0lf) output (x,y)=(%lf,%lf)
   \n",root,x,y,x_output,y_output); system("pause"); */

  free(b);
  free(b2);
  return (((root + b[0]) * (root + b[0])));
}

/**
 * \fn int build_l1r_vector(vector<double> &l1r, double max_distance_corner,int
 * Na, double *a) \brief Build an intermediate vector with values of L-1(r) for
 * d = (0 to max_distance_corner) \param [in] [out] l1r vector to store the
 * roots \param [in] max_distance_corner Maximum distance from distortion center
 * to a corner \param [in] a Lens distortion model polynom \param [in] Na Degree
 * of the lens distortion model polynom \author Luis Alvarez, Pedro Henriquez
 */
int build_l1r_vector(std::vector<double>& l1r, double max_distance_corner,
                     int Na, double* a) {
  // BUILD INTERMEDIATE VECTOR WITH L-1(r) FROM CENTER TO FURTHEST CORNER
  if (a[Na] == 0.) return (-1);
  l1r.resize((int)(max_distance_corner + 1.5));

  // AUXILIARY VARIABLES
  double *b, *b2, root2, root = 1.;

  // WE ALLOCATE MEMORY
  b = (double*)malloc(sizeof(double) * (Na + 2));
  b2 = (double*)malloc(sizeof(double) * (Na + 2));

  // WE DEFINE THE POLYNOM WE NEED TO COMPUTE ROOTS AND THE DERIVATIVE */
  for (int i = 1; i < (Na + 2); i++) {
    b[i] = a[i - 1];
  }
  for (int i = 0; i < (Na + 1); i++) {
    b2[i] = a[i] * (i + 1);
  }

  // WE BUILD THE VECTOR OF INVERSE LENS DISTORTION FUNCTION
  for (int dist = 1; dist < (int)(max_distance_corner + 1.5); dist++) {
    // WE DEFINE THE POLYNOM WE NEED TO COMPUTE ROOTS AND THE DERIVATIVE */
    b[0] = -dist;

    // NEWTON-RAPHSON TO COMPUTE THE POLYNOMIAL ROOT
    for (int k = 0; k < 10000; k++) {
      double pol_eval = ami_polynomial_evaluation(b, Na + 1, root);
      double pol_der = ami_polynomial_evaluation(b2, Na, root);
      if (pol_der == 0.) break;
      root2 = root - pol_eval / pol_der;
      if (fabs(root - root2) < (fabs(root) * 1e-8)) {
        root = root2;
        // printf("k=%d   ",k);
        break;
      }
      root = root2;
    }

    // PUSH RESULT
    l1r[dist] = root / dist;
  }
  free(b);
  free(b2);
  l1r[0] = l1r[1];

  return 0;
}

/**
 * \fn  ami::image<unsigned char>
 *      ami::image<U> undistort_image_inverse_fast(ami::image<unsigned char>
 input_image, lens_distortion_model &d,const double &image_amplification_factor)
 * \brief ESTIMATE AN UNDISTORTED IMAGE USING a DISTORTION MODEL (inverse
 method, accelerated)
 * \author Luis Alvarez, Pedro Henriquez
 */

ami::image<unsigned char> undistort_image_inverse_fast(
    ami::image<unsigned char> input_image, int Na, double* a,
    ami::point2d<double> dc, const double& image_amplification_factor) {
  int width0 = input_image.width();
  int height0 = input_image.height();
  int size = width0 * height0;
  int width = (int)(width0 * image_amplification_factor);
  int height = (int)(height0 * image_amplification_factor);
  int sizenew = width * height;

  // WE CREATE OUTPUT IMAGE
  ami::image<unsigned char> output_image(width, height, 0, 0, 0);

  // CALCULATE MAXIMUM DISTANCE FROM CENTRE TO A CORNER
  ami::point2d<double> corner(0, 0);
  double max_distance_corner = (dc - corner).norm();
  corner.y = height0;
  double distance_corner = (dc - corner).norm();
  if (distance_corner > max_distance_corner)
    max_distance_corner = distance_corner;
  corner.x = width0;
  distance_corner = (dc - corner).norm();
  if (distance_corner > max_distance_corner)
    max_distance_corner = distance_corner;
  corner.y = 0;
  distance_corner = (dc - corner).norm();
  if (distance_corner > max_distance_corner)
    max_distance_corner = distance_corner;

  // BUILD INTERMEDIATE VECTOR
  std::vector<double> l1r;
  if (Na == 0) {
    output_image = input_image;
    return (output_image);
  }
  if (build_l1r_vector(l1r, max_distance_corner, Na, a) == -1) {
    output_image = input_image;
    return (output_image);
  }

  int nc, n2, i, j, n2new;
  for (nc = 0; nc < 3; nc++) {
    n2 = nc * size;
    n2new = nc * sizenew;
#pragma omp parallel for shared(width, height, width0, height0, output_image, \
                                input_image, size, nc, n2) private(i, j)
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        ami::point2d<double> temp((double)j / image_amplification_factor,
                                  (double)i / image_amplification_factor);
        double distance_centre =
            (dc - ami::point2d<double>((double)temp.x, (double)temp.y)).norm();
        // INTERPOLATION
        int ind = (int)distance_centre;
        double dl1r =
            l1r[ind] + (distance_centre - ind) * (l1r[ind + 1] - l1r[ind]);
        ami::point2d<double> p;
        p.x = dc.x + (temp.x - dc.x) * dl1r;
        p.y = dc.y + (temp.y - dc.y) * dl1r;
        int m = (int)p.y;
        int n = (int)p.x;

        if (0 <= m && m < height0 && 0 <= n && n < width0) {
          // COLOUR INTERPOLATION
          double di = p.y - m;
          double dj = p.x - n;
          unsigned int k = i * width + j;
          unsigned int k0 = m * width0 + n;
          double accum = 0;
          double w_accum = 0;
          double w = ((1. - di) * (1. - dj));

          accum += (double)w * input_image[k0 + n2];
          w_accum += w;

          if ((di * (1. - dj)) > 0. && (m + 1) < height0) {
            k0 = (m + 1) * width0 + n;
            w = (di * (1. - dj));
            accum += (double)w * input_image[k0 + n2];
            w_accum += w;
          }
          if (((1 - di) * (dj)) > 0. && (n + 1) < width0) {
            k0 = (m)*width0 + n + 1;
            w = (1. - di) * (dj);
            accum += (double)w * input_image[k0 + n2];
            w_accum += w;
          }
          if (((di) * (dj)) > 0. && (n + 1) < width0 && (m + 1) < height0) {
            k0 = (m + 1) * width0 + n + 1;
            w = (di) * (dj);
            accum += (double)w * input_image[k0 + n2];
            w_accum += w;
          }

          if (w_accum > 0.)
            output_image[k + n2new] = (unsigned char)(accum / w_accum);
        }
      }
    }
  }

  return (output_image);
}
