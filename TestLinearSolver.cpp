static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>
#include <petsc.h>
#include <iostream>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <unistd.h>
#include <mpi.h>
// #include <cuda_runtime.h>
//#include <device_launch_parameters.h>
#include "TestLinearSolver.h"

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  int num_gpus;


  // 获取当前进程使用的 GPU ID
  // int current_gpu_id;
  // cudaGetDevice(&current_gpu_id);

  // 输出当前进程的 rank 和对应的 GPU ID
  //std::cout << "MPI Rank " <<  " is using GPU " << current_gpu_id << std::endl;


  Mat A;
  KSP ksp;
  PetscBool flg;
  PetscViewer viewer;
  PetscInt n, iter;
  Vec b, x, bb;
  PetscBool use_acc_in = PETSC_FALSE, sleep_on = PETSC_FALSE, first_only = PETSC_FALSE, second_only = PETSC_FALSE, mix_only = PETSC_FALSE;
  PetscScalar r;
  const char *converge_reason;

  PetscInt CGiter = 10000;
  PetscScalar cg_tol = 1e-800; // 1e-12
  PetscBool use_acc_out = PETSC_FALSE, use_acc_mix = PETSC_FALSE;

  double start, end;

  PetscRandom ran;
  PetscRandomCreate(PETSC_COMM_WORLD, &ran);
  PetscRandomSetSeed(ran, 1);
  PetscRandomSeed(ran);

  char filename[PETSC_MAX_PATH_LEN];

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  MatSetType(A, MATMPIAIJCUSPARSE); // MATAIJCUSPARSE  MATAIJ MATAIJMKL MATAIJPERM MATAIJSELL
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-file", filename, sizeof(filename), &flg));
  PetscOptionsGetBool(nullptr, nullptr, "-SleepOn", &sleep_on, &flg);
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer)); // 先把矩阵A读入viewer
  PetscCall(MatLoad(A, viewer));                                                         // 再load
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatGetSize(A, &n, &n)); // 获取问题的阶数，存在n里
  VecCreate(PETSC_COMM_WORLD, &b);
  VecSetSizes(b, PETSC_DECIDE, n);
  VecSetType(b, VECMPICUDA);
  VecSetFromOptions(b);
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecDuplicate(b, &bb));
  PetscCall(VecSet(b,1));
  PetscCall(VecSet(bb,1));
  //VecSetRandom(b, NULL);
  //VecSetRandom(bb, NULL);

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  // PetscCall(KSPSetType(ksp, KSPFGMRES));  //kspcg
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetTolerances(ksp, 1e-80, 1e-9, 1000000000000000000000000, CGiter)); // 这个地方设置CG求解的最大迭代次数
  KSPSetFromOptions(ksp);
  PC pc;
  std::string pc_type = "SPAI", pc_type1 = "SPAI", pc_type2 = "GS";
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  //PetscCall(PCHYPRESetType(pc, "boomeramg"));
  PCSetFromOptions(pc);

  MatInfo info;
  MatGetInfo(A, MAT_GLOBAL_SUM, &info);
  int nnz = info.nz_allocated;

  if (sleep_on)
  {
    sleep(15);
  }
  
  // ksp->ops->solve=CG_ACC::KSPSolve_CG_NO_DEFINATE_CHECK;

  KSPSolve(ksp, b, x);

   //struct timeval t1,t2;
  double timeuse;
  int repeat_num=5;
  start = clock(); // 计时开始！！
  //gettimeofday(&t1,NULL);
  for(int i=0;i<repeat_num;i++)
  {
  KSPSolve(ksp, b, x);
  }
  end = clock(); // 计时开始！！
  timeuse = (double)(end - start)/double(repeat_num);
  //gettimeofday(&t2,NULL);
  //timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
  //end = MPI_Wtime(); // 计时结束！！
  KSPGetIterationNumber(ksp, &iter);
  KSPGetResidualNorm(ksp, &r);
  KSPGetConvergedReasonString(ksp, &converge_reason);
  PetscPrintf(PETSC_COMM_WORLD, "matrix:%s\n", filename);

  PetscPrintf(PETSC_COMM_WORLD, "petsc:iteration= %d,time: %g ms,res: %g,reason: %s\n", iter, timeuse/CLOCKS_PER_SEC*1000, r, converge_reason);

  // //ksp->ops->solve = KSPSolve_CG_DASP;
  // //gettimeofday(&t1,NULL);
  // //start = MPI_Wtime(); // 计时开始！！
  // VecSet(x,0);
  // start = clock(); // 计时开始！！
  // KSPSolve(ksp, bb, x);
  // end = clock(); // 计时开始！！
  // timeuse = (double)(end - start);
  // //gettimeofday(&t2,NULL);
  // //end = MPI_Wtime(); // 计时结束！！
  // KSPGetIterationNumber(ksp, &iter);
  // KSPGetResidualNorm(ksp, &r);
  // KSPGetConvergedReasonString(ksp, &converge_reason);
  // PetscPrintf(PETSC_COMM_WORLD, "第二次:iteration: %d,time: %g ms,res: %g,reason:%s\n", iter, timeuse/CLOCKS_PER_SEC*1000, r, converge_reason);
PetscCall(PetscFinalize());
  return 0;
}
