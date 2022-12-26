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
 *    This is the main program (ANSI C language), associated to the publication
 *    An Algebraic Approach to Lens Distortion by Line Rectification,
 *    L. Alvarez, L. Gomez, R. Sendra
 *    Published in JMIV, July 2009
 */

/**  This program has the following components.
 *    - lens_distortion_estimation.cpp:  (with the main() function)
 *    - lens_distortion.h:(having the prototypes of the functions being in
 * lens_distortion.c)
 *    - lens_distortion.cpp:(Set of user defined brief functions for the
 * algebraic and gradient methods)
 *    - ami_pol.h , ami.pol.cpp:(polynomial real root estimation library.)
 *    - image, libs:(Folder with basic libraries to read/write BMP, JPG, TIFF
 * and PNG image formats. These libraries run both on 32-bit and 64-bit
 * operating systems)
 *    - point2d.h:  class to manage 2D points.
 */

/**
 *    Coded by Luis Alvarez and Luis Gomez, AMI Research Group, University of
 *    Las Palmas de Gran Canaria, Canary Islands, SPAIN
 *    First version: February 2010, Second Version: January 2012 (this is the
 * second version) In this version, we optimize the center of distortion using a
 * simple local search pattern strategy
 */

/**
 *     SUMMARY:
 *    - this program calculates the radial distortion parameters for the radial
 * lens distortion model for a given image (input).
 *    - It is necessary to provide the primitives for a reference line which is
 * known to be in the image (or some other straigth motif reference). This is
 * not automatically done, it must be provided by the user.
 *    - It applies a new algebraic method (the one explained in the
 * publication).
 *    - A standard numerical algorithm (gradient-like) is also implemented to
 * minimize the distance function and to compare results with.
 *    - The center of distortion can be indicated by the user (to apply the
 * center of the loaded image or a desired value) through the program input
 * arguments.
 *    - The user can indicate if the center of distortion is considered fixed or
 * it is going to be optimized, through the program input arguments. The center
 * of distortion is optimized using a search patch pattern strategy operating at
 * pixel precision. There is a patch (defined by default of size 20 x 20, in the
 * lens_distortion.h header file #define patch_size 20). It is assumed  that the
 * user provides a valid estimate of the center of distortion, and, the simple
 * pattern strategy, will locate it through the local search optimization
 *      strategy. Once it has been located, its value is optimized at subpixel
 * level by means of the gradient. In order to optimize the center of
 * distortion, user must provide more than one line, to avoid the center being
 * too free (unconstrained optimization), and producing a bad solution. Note
 * that increasing the patch size, increases the CPU time.
 */

/**
 *   Requeriments:
 *  - to be compiled using GCC (on Windows or Linux/UNIX systems).
 *
 *    INPUT/OUTPUT:
 *    - INPUT:
 *            an image in format BMP, JPG, TIFF or PNG which a radial distortion
 * to be corrected.
 *
 *    - OUTPUT:
 *            - the corrected distortion free image,
 *            - numerical results (optimized energy values, distortion
 *            parameters),
 *            - CPU time (just for the running not for the pre-and post-
 *            processing phases).
 */

/**
 *   IMPORTANT: CPU time strongly depends on the size image, not being linearly
 *              scaled for the gradient method.
 */

/**
 *   IMPORTANT:
 *   This is the code for the improved version of the original first IPOL
 * published algorithm (February 2010) New version of the code (January 2012)
 */

#include <stdio.h>
#include <stdlib.h>

#include <malloc.h>

#include "image.h"           /* Functions code associated to read/write images*/
#include "lens_distortion.h" /* Functions code associated to amy_prototypes.h */
#include "point2d.h"

int main(int argc, char* argv[]) {
  int width;           /* Image width */
  int height;          /* Image height */
  int Na;              /* Degree of the lens distortion model polynomial */
  int* Np;             /* Number of points/line */
  int Nl;              /* Number of lines */
  int optimize_center; /* To indicate if the center of distortion must be
                          optimized as well */
  int zoom;            /* To indicate if the zoom strategy applied */
  double* a;         /* Lens distortion model polynom: it corresponds to the "k"
                     coefficients of the lens distortion model polynom */
  double **x, **y;   /* Coordinates of points (normalized) */
  double **xx, **yy; /* Coordinates of points (pixels) */
  double x0, y0;     /* x,y center of distortion of the image (pixels) */
  double*
      trivial; /* Vector to save the trivial initial solution, Emin, Vmin, D */
  ami::image<unsigned char> img; /* intermediate image to use image libraries */

  /* AUXILIAR VARIABLES */
  double factor_n = 0; /* AUXILIAR VARIABLES TO NORMALIZE COORDINATES */
  int i, cont, m;      /* AUXILIAR VARIABLES */
  char filename[300];  /* POINTER TO THE OUTPUT FILE */
  FILE* fp2;

  /* WE INIT TO NULL ALL POINTERS */
  a = NULL;
  x = y = xx = yy = NULL;
  Np = NULL;
  trivial = NULL;

  /* WE CHECK COMMAND LINE PARAMETES */
  /* WE HAVE SEVERAL OPTIONS */

  /* FIRST OPTION: NUMBER OF INPUT PARAMETERS IS 5
      "lens_distortion_estimation input_image.bmp output_undistorted_image.bmp
       input_line_primitives.dat output_lens_distortion_models.dat"
      - CENTER OF DISTORTION: CENTER OF THE IMAGE
       (IT IS READ INTERNALLY FROM THE LOADED IMAGE)
      - THE CENTER OF DISTORTION IS NOT OPTIMIZED
  */

  /* SECOND OPTION: NUMBER OF INPUT PARAMETERS IS 6
      "lens_distortion_estimation input_image.bmp output_undistorted_image.bmp
     input_line_primitives.dat output_lens_distortion_models.dat
     optimize_center"
     - CENTER OF DISTORTION: CENTER OF THE IMAGE (IT IS READ INTERNALLY FROM THE
     LOADED IMAGE)
     - THE CENTER OF DISTORTION IS OPTIMIZED
  */

  /* THIRD OPTION: NUMBER OF INPUT PARAMETERS IS 7
      "lens_distortion_estimation input_image.bmp output_undistorted_image.bmp
     input_line_primitives.dat output_lens_distortion_models.dat x_center
     y_center"
     - CENTER OF DISTORTION: IT IS INDICATED BY THE USER
     - THE CENTER OF DISTORTION IS NOT OPTIMIZED
  */

  /* FOURTH OPTION: NUMBER OF INPUT PARAMETERS IS 8
      "lens_distortion_estimation input_image.bmp output_undistorted_image.bmp
     input_line_primitives.dat output_lens_distortion_models.dat x_center
     y_center optimize_center"
     - CENTER OF DISTORTION: IT IS INDICATED BY THE USER
     - THE CENTER OF DISTORTION IS OPTIMIZED
  */

  /*
    input_image.bmp:                    image to correct the radial distortion
    output_undistorted_image.bmp        output corrected image
    input_line_primitives.dat           file (ASCII) with the primitives of the
    image (points & lines) output_lens_distortion_models.dat   output file with
    the lens distortion coefficients and energy values center_image 1 to select
    the center of the image as center of distortion x_center if center_image==0,
    the x_center is taken as x_center of distortion y_center if center_image==0,
    the y_center is taken as y_center of distortion optimize_center 1: to
    indicate to optimize the center of distortion 0: to indicate not to optimize
    the center of distortion
  */

  // WE PRINT ON SCREEN THAT THE PROGRAM STARTS WORKING
  printf(
      "\n**********************************************************************"
      "******");
  printf("\nLENS DISTORTION PROGRAM STARTS WORKING\n");

  /* WE CAPTURE THE MULTIPLE CHOICE */
  if (argc == 5) { /* FIRST OPTION */
    /* WE READ INPUT IMAGE FROM DISK */
    strcpy(filename, argv[1]);
    if (img.read(filename) != 0) {
      printf("Image can not be loaded\n");
      return (-1);
    }
    width = img.width();
    height = img.height();
    /* WE READ POINT LINE PRIMITES FROM DISK */
    if (read_line_primitives(argv[3], &Nl, &Np, &x, &y) != 0) {
      printf("point line primitives can not be loaded\n");
      return (-1);
    }
    x0 = (double)width / 2.0;
    y0 = (double)height / 2.0;
    optimize_center = 0;
  }

  else if (argc == 6) { /* SECOND OPTION */
    /* WE READ INPUT IMAGE FROM DISK */
    strcpy(filename, argv[1]);
    if (img.read(filename) != 0) {
      printf("Image can not be loaded\n");
      return (-1);
    }
    width = img.width();
    height = img.height();
    /* WE READ POINT LINE PRIMITES FROM DISK */
    if (read_line_primitives(argv[3], &Nl, &Np, &x, &y) != 0) {
      printf("point line primitives can not be loaded\n");
      return (-1);
    }
    x0 = (double)width / 2.0;
    y0 = (double)height / 2.0;
    /* WE CAPTURE THE OPTION TO OPTIMIZE THE CENTER OF DISTORTION */
    optimize_center = atoi(argv[5]);
  }

  else if (argc == 7) { /* THIRD OPTION */
    /* WE READ INPUT IMAGE FROM DISK */
    strcpy(filename, argv[1]);
    if (img.read(filename) != 0) {
      printf("Image can not be loaded\n");
      return (-1);
    }
    width = img.width();
    height = img.height();
    /* WE READ POINT LINE PRIMITES FROM DISK */
    if (read_line_primitives(argv[3], &Nl, &Np, &x, &y) != 0) {
      printf("point line primitives can not be loaded\n");
      return (-1);
    }
    /* WE CAPTURE THE CENTER OF DISTORTION FROM THE GRAPHIC INTERFACE */
    x0 = atof(argv[5]);
    y0 = atof(argv[6]);
    optimize_center = 0;
  }

  else if (argc == 8) { /* FOURTH OPTION */
    /* WE READ INPUT IMAGE FROM DISK */
    strcpy(filename, argv[1]);
    if (img.read(filename) != 0) {
      printf("Image can not be loaded\n");
      return (-1);
    }
    width = img.width();
    height = img.height();
    /* WE READ POINT LINE PRIMITES FROM DISK */
    if (read_line_primitives(argv[3], &Nl, &Np, &x, &y) != 0) {
      printf("point line primitives can not be loaded\n");
      return (-1);
    }
    /* WE CAPTURE THE CENTER OF DISTORTION FROM THE GRAPHIC INTERFACE */
    x0 = atof(argv[5]);
    y0 = atof(argv[6]);
    /* WE CAPTURE THE OPTION TO OPTIMIZE THE CENTER OF DISTORTION */
    optimize_center = atoi(argv[7]);
  }

  else {
    printf("lens_distortion_estimation : number of parameter inadequate\n");
    printf("Usage\n");
    printf(
        "lens_distortion_estimation input_image.bmp "
        "output_undistorted_image.bmp input_line_primitives.dat "
        "output_lens_distortion_models.dat  [x_center y_center] "
        "[optimize_center]");
    return (-1);
  }

  /* WE CHECK THAT THE CENTER OF DISTORTION IS CONSISTENT WITH THE IMAGE */
  if (x0 < 0.0) {
    printf("\n The x-coordinate for the center of distortion is < 0.");
    printf("\n The x-coordinate should be 0.0 <x < width of the image.");
    return (-1);
  }
  if (x0 > width) {
    printf(
        "\n The x-coordinate for the center of distortion is > width of the "
        "image.");
    printf("\n The x-coordinate should be 0.0 < x <width of the image.");
    return (-1);
  }
  if (y0 < 0.0) {
    printf("\n The y-coordinate for the center of distortion is < 0.");
    printf("\n The y-coordinate should be 0.0 <x < height of the image.");
    return (-1);
  }
  if (y0 > height) {
    printf(
        "\n The y-coordinate for the center of distortion is > width of the "
        "image.");
    printf("\n The y-coordinate should be 0.0 < x < height of the image.");
    return (-1);
  }

  /* WE CHECK THAT THE SELECTED VALUE FOR THE VARIABLE TO ACCOUNT FOR OPTIMIZING
     THE CENTER OF DISTORTION IS CONSISTENT
        optimize_center=0: do not optimize the center of distortion (it is fixed
                         to the indicated (x-center,y_center)
        optimize_center=1: optimize the center of distortion using the patch
     search strategy -pixel precision- and then, it is optimized by the gradient
     from the algebraic solution at sub-pixel precision.
   */
  if (optimize_center != 0 && optimize_center != 1) {
    printf("\n The selection for this input variable is not allowed.");
    printf("\n The selection should be 0 (not optimize); or 1 (optimize).");
    return (-1);
  }

  /* TO AVOID A BAD SOLUTION WHEN OPTIMIZING THE CENTER OF DISTORTION USING THE
  A MINIMUM NUMBER OF PRIMITIVES, IT IS NOT ALLOWED OPTIMIZING THE CENTER OF
  DISTORTION FOR THE CASE OF USING ONLY ONE LINE. FROM THE EXPERIENCE, WE NOTE
  THAT THE ALGORITHM WHICH OPTIMIZES THE CENTER OF DISTORTION CAN BE TRAPPED
  INTO A BAD LOCAL MINIMA. AS A RESULT, THE SOLUTION CAN BE A BAD ONE, AND
  HENCE, THIS SITUATION IS SIMPLY AVOIDED */
  if (Nl == 1) optimize_center = 0;

  /* WE ALLOCATE MEMORY FOR AUXILARY POINTS AND WE NORMALIZE ORIGINAL POINTS */
  xx = (double**)malloc(sizeof(double*) * Nl); /*  xx pixels coordinates */
  yy = (double**)malloc(sizeof(double*) * Nl); /*  yy pixels coordinates */
  for (i = 0; i < Nl; i++) {
    xx[i] = (double*)malloc(sizeof(double) * Np[i]);
    yy[i] = (double*)malloc(sizeof(double) * Np[i]);
  }
  printf(
      "************************************************************************"
      "****");
  printf("\nInitial Center of distortion:");
  printf("\nx0=%f,y0=%f", x0, y0);
  if (optimize_center == 1)
    printf("\nIs the Center of Distortion going to be optimized?: YES");
  else
    printf("\nIs the Center of Distortion going to be optimized?: NO");
  printf(
      "\n**********************************************************************"
      "******");

  /* WE KEEP A COPY OF THE PIXEL COORDINATES VECTORS */
  for (m = 0; m < Nl; m++) {
    for (i = 0; i < Np[m]; i++) {
      xx[m][i] = x[m][i];
      yy[m][i] = y[m][i];
    }
  }

  /* WE ALLOCATE MEMORY FOR MAXIMUM LENS DISTORTION POLYNOMIAL SIZE AND WE
   * INITIALIZE TO TRIVIAL CASE */
  /* WE USE IT ONLY FOR THE ALGEBRAIC ALONE METHOD */
  ami_calloc1d(a, double, 7);
  a[0] = 1.0;
  /* MAXIMUM DEGREE OF THE LENS DISTORTION POLYNOM */
  Na = 4;

  /* WE COPY THE CENTER OF DISTORTION TO THE THE 5 AND 6th POSITIONS*/
  a[5] = x0;
  a[6] = y0;

  /* INITIAL TRIVIAL SOLUTION IS IN VECTOR a=[1,0,0,0,0,x0,y0] */

  /* WE COMPUTE THE SOLUTIONS AND SAVE IT TO THE INDICATED
  "output_lens_distortion_models.dat" WE PREPARE THE OUTPUT FILE */
  if (!(fp2 = fopen(argv[4], "w"))) {
    printf("Error while opening the %s file", filename);
    exit(0);
  }

  /* FIRST WE COMPUTE THE TRIVIAL SOLUTION AND SAVE IT TO THE OUTPUT FILE */
  /* WE ALLOCATE MEMORY TRIVIAL SOLUTION FOR Emin, Vmin, D */
  ami_calloc1d(trivial, double, 3);
  trivial_solution(Nl, Np, a, xx, yy, factor_n, fp2, trivial, optimize_center);

  /* SELECT THE BIGGEST VALUE (FOR THE INPUT IMAGE): THE POINT IN A CORNER, TO
   BE USED IN THE LOCATION OF THE ALGEBRAIC ROOTS */
  double xtmp = width - width / 2.0;
  double ytmp = height - height / 2.0;
  double max_radius = sqrt(pow(xtmp, 2.0) + pow(ytmp, 2.0));

  /* OPTIMIZED CENTER: WE SCAN A PATCH CENTERED AT THE GIVEN CENTER OF
     DISTORTION: WE SELECT THE BEST SOLUTION AND THEN APPLIED THE GRADIENT
     METHOD, TO GET THE SOLUTIONS USING ZOOM AND NOT USING ZOOM
    */
  if (optimize_center == 1) {
    search_for_best_center(Nl, Np, a, xx, yy, width, height, max_radius);
    x0 = a[5]; /* WE CAPTURE THE OPTIMIZED CENTER OF DISTORTION */
    y0 = a[6];
  }

  /* ALGEBRAIC METHOD & GRADIENT METHOD WORK WITH NORMALIZED UNITS */
  factor_n = 0.0;
  cont = 0;
  for (m = 0; m < Nl; m++) {
    for (i = 0; i < Np[m]; i++) {
      x[m][i] -= x0;
      y[m][i] -= y0;
      cont++;
      factor_n += x[m][i] * x[m][i] + y[m][i] * y[m][i];
    }
  }
  factor_n = sqrt(factor_n / cont);
  for (m = 0; m < Nl; m++)
    for (i = 0; i < Np[m]; i++) {
      x[m][i] /= factor_n;
      y[m][i] /= factor_n;
    }

  /* IN NORMALIZED COORDINATES */
  xtmp = xtmp / factor_n;
  ytmp = ytmp / factor_n;
  max_radius = sqrt(pow(xtmp, 2.0) + pow(ytmp, 2.0));

  printf("\nmax_radius=%f", max_radius);
  printf(
      "\n**********************************************************************"
      "******");

  /* ALGEBRAIC METHOD FROM TRIVIAL SOLUTION + GRADIENT TO IMPROVE THE ALGEBRAIC
   * SOLUTION; WITHOUT ZOOM */
  /* ALGEBRAIC METHOD & GRADIENT METHOD WORK WITH NORMALIZED UNITS */
  zoom = 0; /* NO ZOOM APPLIED */
  algebraic_method_pre_gradient(Nl, Np, a, x, y, xx, yy, factor_n, zoom, fp2,
                                optimize_center, max_radius);

  /* ALGEBRAIC METHOD FROM TRIVIAL SOLUTION + GRADIENT TO IMPROVE THE ALGEBRAIC
   * SOLUTION; WITH ZOOM */
  zoom = 1; /* ZOOM APPLIED */
  algebraic_method_pre_gradient(Nl, Np, a, x, y, xx, yy, factor_n, zoom, fp2,
                                optimize_center, max_radius);
  fprintf(fp2,
          "\n*******************************END OF "
          "FILE*******************************");
  fclose(fp2); /* CLOSE THE OUTPUT FILE */

  /* PRINT SOME INFORMATION ON SCREEN */
  printf("\nFinal Center of Distortion:");
  printf("\nx0=%f,y0=%f", a[5], a[6]);
  printf(
      "\n**********************************************************************"
      "******");
  printf("\nSolution:\n");
  for (i = 0; i <= Na; i++) printf("k[%d]=%e\n", i, a[i]);
  printf(
      "************************************************************************"
      "****");
  /* WE COMPUTE THE UNDISTORTED IMAGE */
  ami::point2d<double> dc(a[5], a[6]);

  /* FAST UNDISTORT INVERSE */
  cout << endl << "Undistorting image..." << endl;
  ami::image<unsigned char> img_out3 =
      undistort_image_inverse_fast(img, Na, a, dc, 1.);
  strcpy(filename, argv[2]);
  if (img_out3.write(filename) != 0) printf("Image can not be saved\n");

  /* WE FREE THE MEMORY */
  if (a != NULL) free(a);
  if (x != NULL) {
    for (i = 0; i < Nl; i++) {
      if (x[i] != NULL) free(x[i]);
    }
    free(x);
  }
  if (y != NULL) {
    for (i = 0; i < Nl; i++) {
      if (y[i] != NULL) free(y[i]);
    }
    free(y);
  }
  if (xx != NULL) {
    for (i = 0; i < Nl; i++) {
      if (xx[i] != NULL) free(xx[i]);
    }
    free(xx);
  }
  if (yy != NULL) {
    for (i = 0; i < Nl; i++) {
      if (yy[i] != NULL) free(yy[i]);
    }
    free(yy);
  }
  if (trivial != NULL) free(trivial);
  if (Np != NULL) free(Np);

  // WE PRINT ON SCREEN THAT THE PROGRAM FINISHED WORKING
  printf(
      "\n**********************************************************************"
      "******");
  printf("\nLENS DISTORTION PROGRAM FINISHED WORKING\n");
  return (0);
}
