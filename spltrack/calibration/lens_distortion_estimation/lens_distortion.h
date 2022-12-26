// Copyright (C) 2009-2010, Luis Alvarez <lalvarez@dis.ulpgc.es>
/* lens_distortion.c */
/**
 *    Coded by Luis Alvarez and Luis Gomez, AMI Research Group, University of
 *    Las Palmas de Gran Canaria, Canary Islands, SPAIN
 *    First version: February 2010, Second Version: January 2012 (this is the
 * second version) In this version, we optimize the center of distortion using a
 * search patch pattern strategy
 */

#ifndef _LENS_DISTORTION_A_H_
#define _LENS_DISTORTION_A_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#include <malloc.h>

#include "image.h"
#include "point2d.h"
#ifdef AMI_OMP_H
#include <omp.h>
#endif
extern "C" {
#include "ami_pol.h"
}

#define PI 3.1415927
#define ABS(x) (((x) > 0) ? (x) : -(x))
#define Normalize(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : x))
#define Max(a, b) ((a > b) ? a : b)
#define ami_abs(x) ((x) > 0 ? (x) : (-(x)))
#define ami_calloc2d(address, datatype, height, width)                     \
  {                                                                        \
    int ml, mk;                                                            \
    address = (datatype**)malloc(sizeof(datatype*) * (height));            \
    address[0] = (datatype*)malloc(sizeof(datatype) * (width) * (height)); \
    for (ml = 0; ml < (height); ml++)                                      \
      address[ml] = &(address[0][ml * (width)]);                           \
    for (ml = 0; ml < height; ml++)                                        \
      for (mk = 0; mk < width; mk++) address[ml][mk] = 0;                  \
  }
#define ami_malloc2d(address, datatype, height, width)                     \
  {                                                                        \
    int ml;                                                                \
    address = (datatype**)malloc(sizeof(datatype*) * (height));            \
    address[0] = (datatype*)malloc(sizeof(datatype) * (width) * (height)); \
    for (ml = 0; ml < (height); ml++)                                      \
      address[ml] = &(address[0][ml * (width)]);                           \
  }
#define ami_malloc2d_pointers(address, simple_pointer, datatype, height, \
                              width)                                     \
  {                                                                      \
    int ml;                                                              \
    address = (datatype**)malloc(sizeof(datatype*) * (height));          \
    address[0] = (datatype*)simple_pointer;                              \
    for (ml = 0; ml < (height); ml++)                                    \
      address[ml] = &(address[0][ml * (width)]);                         \
  }
#define ami_free2d(address) \
  {                         \
    free(address[0]);       \
    free(address);          \
  }
#define ami_free2d_pointers(address) \
  { free(address); }
#define ami_malloc1d(address, datatype, size) \
  { address = (datatype*)malloc(sizeof(datatype) * (size)); }
#define ami_calloc1d(address, datatype, size)               \
  {                                                         \
    int ml;                                                 \
    address = (datatype*)malloc(sizeof(datatype) * (size)); \
    for (ml = 0; ml < size; ml++) address[ml] = 0;          \
  }

/**
 *            BEGIN: ALGEBRAIC CONTROL PARAMETERS
 */

#define ami_max_iter 1000
#define ami_tol 0.0000001
/**
 *           END: ALGEBRAIC CONTROL PARAMETERS
 */

#define line_length 80 /* LENGTH OF LINE (TO READ A LINE FOR A FILE) */

/**
 *           BEGIN: GRADIENT CONTROL PARAMETERS
 */
#define max_itera 100 /* MAXIMUM NUMBER OF GRADIENT ITERATIONS */
#define delta 1.0e-10 /* DERIVATIVE STEP (FINITE DIFFERENCES) */
#define max_itera_lambda \
  10                 /* MAXIMUM NUMBER OF ITERATIONS IN UNIDIMENSIONAL SEARCH */
#define tol_f 1.0e-6 /* TOLERANCE TO STOP THE GRADIENT ITERATIONS */
#define tol_norma 1.0e-16 /* NORM OF GRADIENT TO STOP THE GRADIENT ALGORITHM \
                           */
/**
 *           END: GRADIENT CONTROL PARAMETERS
 */

/**
 *           BEGIN: SEARCH-OF-DISTORTION-CENTER PARAMETERS
 */
#define patch_size 20
#define max_itera_patch \
  20 /* MAXIMUM NUMBER OF SEARCH-OF-DISTORTION-CENTER ITERATIONS */
/**
 *           END: SEARCH-OF-DISTORTION-CENTER PARAMETERS
 */

int test_compatibility_lens_distortion_model(double* a, int Na,
                                             double max_radius);
int ami_line2d_calculation(double line[3], double** Points2D, int N);
int ami_lens_distortion_polynomial_update_distance_2v(
    double* x, double* y, int Np, double* a, int Na, double x0, double y0,
    int k1, int k2, double** pol, double alpha);
double ami_lens_distortion_estimation_2v(double** x, double** y, int Nl,
                                         int* Np, double x0, double y0,
                                         double* a, int Na, int k1, int k2,
                                         double alpha, double max_radius);
int ami_lens_distortion_model_update_2v(double* a, int Na, int k1, int k2,
                                        double** pol, double max_radius);
int ami_lens_distortion_polynomial_update_2v(double* x, double* y, int Np,
                                             double* a, int Na, double x0,
                                             double y0, int k1, int k2,
                                             double** pol);
void ami_2v_polynom_derivatives(double** p, int N, double** p_x, double** p_y);
void ami_polynom_determinant(double p[6][6][19], int Np, int Nd, double* q);
double ami_2v_polynom_evaluation(double** p1, int N1, double x, double y);
void ami_2v_polynom_to_1v_polynom(double** p1, int N1, double* p3, double z,
                                  int flat);
double* ami_1v_polynom_multiplication(double* p1, int N1, double* p2, int N2,
                                      double* p3);
void ami_2v_polynom_multiplication(double** p1, int N1, double** p2, int N2,
                                   double** p3);
int ami_RootCubicPolynomial(double* a, int N, double* x);
double ami_polynomial_evaluation(double* a, int Na, double x);
int ami_lens_distortion_polynomial_update(double* x, double* y, int Np,
                                          double* a, int Na, double x0,
                                          double y0, int k, double* pol);
int ami_lens_distortion_model_update(double* a, int Na, int k, double* pol);
double ami_LensDistortionEnergyError(double* x, double* y, int Np, double x0,
                                     double y0, double* a, int Na);
double ami_LensDistortionEnergyError_Vmin(double* x, double* y, int Np,
                                          double x0, double y0, double* a,
                                          int Na);
double ami_lens_distortion_estimation(double** x, double** y, int Nl, int* Np,
                                      double x0, double y0, double* a, int Na,
                                      int k, double alpha);
void ami_lens_distortion_zoom_normalization(double** x, double** y, int Nl,
                                            int* Np, double* solution, int Na);
int calculate_points(double* amin, double** points_2D_modified, int N, int Na,
                     double x0, double y0);
double distance_function(double* solution, double** x, double** y, int Nl,
                         int* Np, int Na);
double find_lambda(double lambda1, double lambda2, double lambda3, double f_1,
                   double f_2, double f_3, double* amin_copy, double* amin,
                   double** x, double** y, int Nl, int* Np, int Na,
                   double* grad_f, int* change_k);
double minimize_cuadratic_polynom(double lambda1, double lambda2,
                                  double lambda3, double f_1, double f_2,
                                  double f_3, double* amin_copy, double* amin,
                                  double** x, double** y, int Nl, int* Np,
                                  int Na, double* grad_f, int* change_k);
double cuadratic_fitting(double* amin_copy, double* amin, double** x,
                         double** y, int Nl, int* Np, int Na, double lambda1,
                         double lambda2, double lambda3, double f_1, double f_2,
                         double f_3, double* grad_f, int* change_k);
double minimize_lambda(double* amin, double** x, double** y, int Nl, int* Np,
                       int Na, double* grad_f, double f, int* change_k);
double gradient_method(double* solution, double** x, double** y, int Nl,
                       int* Np, int Na, int* change_k, int zoom);
double calculate_factor_n(double** x, double** y, int Nl, int* Np, double x0,
                          double y0);
int optimize(double* solution, double** x, double** y, double** xx, double** yy,
             int Nl, int* Np, int Na, double factor_n, int zoom, FILE* fp1,
             int optimize_center);
int algebraic_method_pre_gradient(int Nl, int* Np, double* a, double** x,
                                  double** y, double** xx, double** yy,
                                  double factor_n, int zoom, FILE* fp1,
                                  int optimize_center, double max_radius);
int trivial_solution(int Nl, int* Np, double* a, double** xx, double** yy,
                     double factor_n, FILE* fp1, double* trivial,
                     int optimize_center);
int read_line_primitives(char filename[300], int* Nl, int** Np, double*** x,
                         double*** y);
int search_for_best_center(int N, int* Np, double* a, double** xx, double** yy,
                           int width, int height, double max_radius);
void ami_lens_distortion_model_evaluation(double* a, int Na, double xc,
                                          double yc, double x_input,
                                          double y_input, double* x_output,
                                          double* y_output);
double ami_inverse_lens_distortion_newton_raphson(double x, double y, double x0,
                                                  double y0, double* xt,
                                                  double* yt, double* a,
                                                  int Na);
int ami_inverse_lens_distortion_fast(double x, double y, double x0, double y0,
                                     double* xt, double* yt, double* a, int Na,
                                     double dl1r);
int build_l1r_vector(std::vector<double>& l1r, double max_distance_corner,
                     int Na, double* a);
ami::image<unsigned char> undistort_image_inverse_fast(
    ami::image<unsigned char> input_image, int Na, double* a,
    ami::point2d<double> dc, const double& image_amplification_factor);

#endif
