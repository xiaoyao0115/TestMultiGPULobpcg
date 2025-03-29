#include <petscksp.h>
#include <petsclog.h>
#include <petsc.h>
#include <iostream>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <unistd.h>
#include <mpi.h>

#ifndef TEST_LINEAR_SOLVER_H
#define TEST_LINEAR_SOLVER_H

#define VecXDot(x, y, a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x, y, a) : VecTDot(x, y, a))
typedef struct
{
  KSPCGType type; /* type of system (symmetric or Hermitian) */

  // The following arrays are of size ksp->maxit
  PetscScalar *e, *d;
  PetscReal *ee, *dd; /* work space for Lanczos algorithm */

  /* Trust region support */
  PetscReal radius;
  PetscReal obj;
  PetscReal obj_min;

  PetscBool singlereduction; /* use variant of CG that combines both inner products */
} KSP_CG;



// struct timeval{
// long tv_sec; /*秒*/
// long tv_usec; /*微秒*/
// };

#endif