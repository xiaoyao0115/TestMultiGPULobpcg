#include <iostream>
#include <petscksp.h>
#include <petsclog.h>
#include <slepc.h>
#include <slepceps.h>
#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <slepc/private/epsimpl.h>
#include <vector>

static char help[] = "Solves a standard eigensystem Ax=kx with the matrix loaded from a file.\n"
                     "This example works for both real and complex numbers.\n\n"
                     "The command line options are:\n"
                     "  -file <filename>, where <filename> = matrix file in PETSc binary form.\n\n";

int SpecialOrthogonalize(BV &Z)
{
    PetscReal norm;
    PetscInt l, k, j, i;
    std::vector<int> dependent; // 记录Z中线性相关的列
    BVGetActiveColumns(Z, &l, &k);

    for (i = l; i < k; ++i)
    {
        BVOrthogonalizeColumn(Z, i, nullptr, &norm, nullptr);
        if (norm)
            BVScaleColumn(Z, i, 1.0 / norm);
        else
            dependent.push_back(i);
    }

    if (!dependent.size())
    {

        return 0;
    }
    if (*dependent.end() != l - k)
        dependent.push_back(l - k);
    j = 0;
    i = l;
    auto it = dependent.begin();

    while (i < k && it < dependent.end())
    {
        if (i == *it)
        {
            ++j;
            ++it;
            ++i;
            continue;
        }
        if (j)
        {
            BVCopyColumn(Z, i, i - j);
        }
        ++i;
        ++it;
    }
    BVSetActiveColumns(Z, l, k - dependent.size());
    return dependent.size();
}

typedef struct
{
    PetscInt bs;       /* block size */
    PetscBool lock;    /* soft locking active/inactive */
    PetscReal restart; /* restart parameter */
    PetscInt guard;    /* number of guard vectors */
} EPS_LOBPCG;

PetscErrorCode EPSSolve_LOBPCG(EPS eps)
{
    EPS_LOBPCG *ctx = (EPS_LOBPCG *)eps->data;
    PetscInt i, j, k, nv, ini, nmat, nc, nconv, locked, its, prev = 0;
    PetscReal norm;
    PetscScalar *eigr, dot;
    PetscBool breakdown, countc, flip = PETSC_FALSE, checkprecond = PETSC_FALSE;
    Mat A, B, M, V = NULL, W = NULL;
    Vec v, z, w = eps->work[0];
    BV X, Y = NULL, Z, R, P, AX, BX;
    SlepcSC sc;

    KSP ksp;
    PC pc;
    STGetKSP(eps->st, &ksp);
    KSPGetPC(ksp, &pc);
    PCSetUp(pc);

    PetscFunctionBegin;
    PetscCall(STGetNumMatrices(eps->st, &nmat));
    PetscCall(STGetMatrix(eps->st, 0, &A));
    if (nmat > 1)
        PetscCall(STGetMatrix(eps->st, 1, &B));
    else
        B = NULL;

    if (eps->which == EPS_LARGEST_REAL)
    { /* flip spectrum */
        flip = PETSC_TRUE;
        PetscCall(DSGetSlepcSC(eps->ds, &sc));
        sc->comparison = SlepcCompareSmallestReal;
    }

    /* undocumented option to check for a positive-definite preconditioner (turn-off by default) */
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-eps_lobpcg_checkprecond", &checkprecond, NULL));

    // PetscPrintf(PETSC_COMM_WORLD, "guard的默认大小是:%d", ctx->guard);

    /*added by mtl for setting new values*/
    BV ZZ;

    ctx->bs = 10;
    ctx->guard = 3; /// ILU的所有算例 guard都是3

    ctx->lock = PETSC_TRUE;
    PetscCall(BVDuplicateResize(eps->V, 3 * ctx->bs, &ZZ));

    /* 1. Allocate memory */
    PetscCall(PetscCalloc1(3 * ctx->bs, &eigr));
    PetscCall(BVDuplicateResize(eps->V, 3 * ctx->bs, &Z));
    PetscCall(BVDuplicateResize(eps->V, ctx->bs, &X));
    PetscCall(BVDuplicateResize(eps->V, ctx->bs, &R));
    PetscCall(BVDuplicateResize(eps->V, ctx->bs, &P));
    PetscCall(BVDuplicateResize(eps->V, ctx->bs, &AX));
    if (B)
    {
        PetscCall(BVDuplicateResize(eps->V, ctx->bs, &BX));
    }

    nc = eps->nds;
    if (nc > 0 || eps->nev > ctx->bs - ctx->guard)
        PetscCall(BVDuplicateResize(eps->V, nc + eps->nev, &Y));
    if (nc > 0)
    {
        for (j = 0; j < nc; j++)
        {
            PetscCall(BVGetColumn(eps->V, -nc + j, &v));
            PetscCall(BVInsertVec(Y, j, v));
            PetscCall(BVRestoreColumn(eps->V, -nc + j, &v));
        }
        PetscCall(BVSetActiveColumns(Y, 0, nc));
    }

    if (B)
    {
        PetscBool flag = PETSC_FALSE;
        BVSetMatrix(X, B, flag);
        BVSetMatrix(Y, B, flag);
        BVSetMatrix(Z, B, flag);
        BVSetMatrix(R, B, flag);
        BVSetMatrix(P, B, flag);
        BVSetMatrix(eps->V, B, flag);
    }

    /* 2. Apply the constraints to the initial vectors */
    /* 3. B-orthogonalize initial vectors */
    for (k = eps->nini; k < eps->ncv - ctx->bs; k++)
    { /* Generate more initial vectors if necessary */
        PetscCall(BVSetRandomColumn(eps->V, k));
        PetscCall(BVOrthonormalizeColumn(eps->V, k, PETSC_TRUE, NULL, NULL));
    }
    nv = ctx->bs;
    PetscCall(BVSetActiveColumns(eps->V, 0, nv));
    PetscCall(BVSetActiveColumns(Z, 0, nv));
    PetscCall(BVCopy(eps->V, Z));
    PetscCall(BVCopy(Z, X));

    /* 4. Compute initial Ritz vectors */
    PetscCall(BVMatMult(X, A, AX));
    PetscCall(DSSetDimensions(eps->ds, nv, 0, 0));
    PetscCall(DSGetMat(eps->ds, DS_MAT_A, &M));
    PetscCall(BVMatProject(AX, NULL, X, M));
    if (flip)
        PetscCall(MatScale(M, -1.0));
    PetscCall(DSRestoreMat(eps->ds, DS_MAT_A, &M));
    if (B)
    {
        PetscCall(DSGetMat(eps->ds, DS_MAT_B, &M));
        PetscCall(BVMatProject(Z, B, Z, M)); /* covers also the case B=nullptr */
        PetscCall(DSRestoreMat(eps->ds, DS_MAT_B, &M));
    }
    else
    {
        PetscCall(DSSetIdentity(eps->ds, DS_MAT_B));
    }
    PetscCall(DSSetState(eps->ds, DS_STATE_RAW));
    PetscCall(DSSolve(eps->ds, eigr, NULL));
    PetscCall(DSSort(eps->ds, eigr, NULL, NULL, NULL, NULL));
    PetscCall(DSSynchronize(eps->ds, eigr, NULL));
    for (j = 0; j < nv; j++)
        eps->eigr[j] = flip ? -eigr[j] : eigr[j];
    PetscCall(DSVectors(eps->ds, DS_MAT_X, NULL, NULL));
    PetscCall(DSGetMat(eps->ds, DS_MAT_X, &M));
    PetscCall(BVMultInPlace(X, M, 0, nv));
    PetscCall(BVMultInPlace(AX, M, 0, nv));
    PetscCall(DSRestoreMat(eps->ds, DS_MAT_X, &M));

    /* 5. Initialize range of active iterates */
    locked = 0; /* hard-locked vectors, the leading locked columns of V are eigenvectors */
    nconv = 0;  /* number of converged eigenvalues in the current block */
    its = 0;    /* iterations for the current block */

    /*added by mtl for additional orthogonalization*/
    PetscCall(BVSetOrthogonalization(Z, BV_ORTHOG_MGS, BV_ORTHOG_REFINE_ALWAYS, 0.5, BV_ORTHOG_BLOCK_GS)); // BV_ORTHOG_BLOCK_TSQR  BV_ORTHOG_BLOCK_GS
    /* 6. Main loop */
    while (eps->reason == EPS_CONVERGED_ITERATING)
    {

        if (ctx->lock)
        {
            PetscCall(BVSetActiveColumns(R, nconv, ctx->bs));
            PetscCall(BVSetActiveColumns(AX, nconv, ctx->bs));
            if (B)
                PetscCall(BVSetActiveColumns(BX, nconv, ctx->bs));
        }

        /* 7. Compute residuals */
        ini = (ctx->lock) ? nconv : 0;
        PetscCall(BVCopy(AX, R));
        if (B)
            PetscCall(BVMatMult(X, B, BX));
        for (j = ini; j < ctx->bs; j++)
        {
            PetscCall(BVGetColumn(R, j, &v));
            PetscCall(BVGetColumn(B ? BX : X, j, &z));
            PetscCall(VecAXPY(v, -eps->eigr[locked + j], z));
            PetscCall(BVRestoreColumn(R, j, &v));
            PetscCall(BVRestoreColumn(B ? BX : X, j, &z));
        }
        /* 8. Compute residual norms and update index set of active iterates */
        k = ini;
        countc = PETSC_TRUE;
        for (j = ini; j < ctx->bs; j++)
        {
            i = locked + j;
            PetscCall(BVGetColumn(R, j, &v));
            PetscCall(VecNorm(v, NORM_2, &norm));
            PetscCall(BVRestoreColumn(R, j, &v));
            PetscCall((*eps->converged)(eps, eps->eigr[i], eps->eigi[i], norm, &eps->errest[i], eps->convergedctx));
            if (countc)
            {
                if (eps->errest[i] < eps->tol)
                    k++;
                else
                    countc = PETSC_FALSE;
            }
            if (!countc && !eps->trackall)
                break;
        }

        nconv = k;
        eps->nconv = locked + nconv;
        if (its)
            PetscCall(EPSMonitor(eps, eps->its + its, eps->nconv, eps->eigr, eps->eigi, eps->errest, locked + ctx->bs));
        PetscCall((*eps->stopping)(eps, eps->its + its, eps->max_it, eps->nconv, eps->nev, &eps->reason, eps->stoppingctx));
        if (eps->reason != EPS_CONVERGED_ITERATING || nconv >= ctx->bs - ctx->guard)
        {
            PetscCall(BVSetActiveColumns(eps->V, locked, eps->nconv));
            PetscCall(BVSetActiveColumns(X, 0, nconv));
            PetscCall(BVCopy(X, eps->V));
        }
        if (eps->reason != EPS_CONVERGED_ITERATING)
        {
            break;
        }
        else if (nconv >= ctx->bs - ctx->guard)
        {
            eps->its += its - 1;
            its = 0;
        }
        else
            its++;
        if (nconv >= ctx->bs - ctx->guard)
        { /* force hard locking of vectors and compute new R */

            /* extend constraints */
            PetscCall(BVSetActiveColumns(Y, nc + locked, nc + locked + nconv));
            PetscCall(BVCopy(X, Y));
            PetscCall(BVSetActiveColumns(Y, 0, nc + locked + nconv));

            /* shift work BV's */
            for (j = nconv; j < ctx->bs; j++)
            {
                PetscCall(BVCopyColumn(X, j, j - nconv));
                PetscCall(BVCopyColumn(R, j, j - nconv));
                PetscCall(BVCopyColumn(P, j, j - nconv));
                PetscCall(BVCopyColumn(AX, j, j - nconv));
                if (B)
                    PetscCall(BVCopyColumn(BX, j, j - nconv));
            }

            /* set new initial vectors */
            PetscCall(BVSetActiveColumns(eps->V, locked + ctx->bs, locked + ctx->bs + nconv));
            PetscCall(BVSetActiveColumns(X, ctx->bs - nconv, ctx->bs));
            PetscCall(BVCopy(eps->V, X));
            for (j = ctx->bs - nconv; j < ctx->bs; j++)
            {
                PetscCall(BVGetColumn(X, j, &v));
                PetscCall(BVOrthogonalizeVec(Y, v, NULL, &norm, &breakdown));
                if (norm > 0.0 && !breakdown)
                    PetscCall(VecScale(v, 1.0 / norm));
                else
                {
                    PetscCall(PetscInfo(eps, "Orthogonalization of initial vector failed\n"));
                    eps->reason = EPS_DIVERGED_BREAKDOWN;
                    goto diverged;
                }
                PetscCall(BVRestoreColumn(X, j, &v));
            }
            locked += nconv;
            nconv = 0;
            PetscCall(BVSetActiveColumns(X, nconv, ctx->bs));

            /* B-orthogonalize initial vectors */
            PetscCall(BVOrthogonalize(X, NULL));
            PetscCall(BVSetActiveColumns(Z, nconv, ctx->bs));
            PetscCall(BVSetActiveColumns(AX, nconv, ctx->bs));
            PetscCall(BVCopy(X, Z));

            /* compute initial Ritz vectors */
            nv = ctx->bs;
            PetscCall(BVMatMult(X, A, AX));
            PetscCall(DSSetDimensions(eps->ds, nv, 0, 0));
            PetscCall(DSGetMat(eps->ds, DS_MAT_A, &M));
            PetscCall(BVMatProject(AX, NULL, X, M));
            if (flip)
                PetscCall(MatScale(M, -1.0));
            PetscCall(DSRestoreMat(eps->ds, DS_MAT_A, &M));
            if (B)
            {
                PetscCall(DSGetMat(eps->ds, DS_MAT_B, &M));
                PetscCall(BVMatProject(Z, B, Z, M)); /* covers also the case B=nullptr */
                PetscCall(DSRestoreMat(eps->ds, DS_MAT_B, &M));
            }
            else
            {
                PetscCall(DSSetIdentity(eps->ds, DS_MAT_B));
            }
            PetscCall(DSSetState(eps->ds, DS_STATE_RAW));
            PetscCall(DSSolve(eps->ds, eigr, NULL));
            PetscCall(DSSort(eps->ds, eigr, NULL, NULL, NULL, NULL));
            PetscCall(DSSynchronize(eps->ds, eigr, NULL));
            for (j = 0; j < nv; j++)
                if (locked + j < eps->ncv)
                    eps->eigr[locked + j] = flip ? -eigr[j] : eigr[j];
            PetscCall(DSVectors(eps->ds, DS_MAT_X, NULL, NULL));
            PetscCall(DSGetMat(eps->ds, DS_MAT_X, &M));
            PetscCall(BVMultInPlace(X, M, 0, nv));
            PetscCall(BVMultInPlace(AX, M, 0, nv));
            PetscCall(DSRestoreMat(eps->ds, DS_MAT_X, &M));

            continue; /* skip the rest of the iteration */
        }

        ini = (ctx->lock) ? nconv : 0;
        if (ctx->lock)
        {
            PetscCall(BVSetActiveColumns(R, nconv, ctx->bs));
            PetscCall(BVSetActiveColumns(P, nconv, ctx->bs));
            PetscCall(BVSetActiveColumns(AX, nconv, ctx->bs));
            if (B)
                PetscCall(BVSetActiveColumns(BX, nconv, ctx->bs));
        }
        /*警告！ILU算例里没有使用这个！*/
        BVNormalize(R, NULL);

        /* 9. Apply preconditioner to the residuals */
        PetscCall(BVGetMat(R, &V));
        if (prev != ctx->bs - ini)
        {
            prev = ctx->bs - ini;
            PetscCall(MatDestroy(&W));
            PetscCall(MatDuplicate(V, MAT_SHARE_NONZERO_PATTERN, &W));
        }
        PetscCall(STApplyMat(eps->st, V, W));
        if (checkprecond)
        {
            for (j = ini; j < ctx->bs; j++)
            {
                PetscCall(MatDenseGetColumnVecRead(V, j - ini, &v));
                PetscCall(MatDenseGetColumnVecRead(W, j - ini, &w));
                PetscCall(VecDot(v, w, &dot));
                PetscCall(MatDenseRestoreColumnVecRead(W, j - ini, &w));
                PetscCall(MatDenseRestoreColumnVecRead(V, j - ini, &v));
                if (PetscRealPart(dot) < 0.0)
                {
                    PetscCall(PetscInfo(eps, "The preconditioner is not positive-definite\n"));
                    eps->reason = EPS_DIVERGED_BREAKDOWN;
                    goto diverged;
                }
            }
        }
        if (nc + locked > 0)
        {
            for (j = ini; j < ctx->bs; j++)
            {
                PetscCall(MatDenseGetColumnVecWrite(W, j - ini, &w));
                PetscCall(BVOrthogonalizeVec(Y, w, NULL, &norm, &breakdown));
                if (norm > 0.0 && !breakdown)
                    PetscCall(VecScale(w, 1.0 / norm));
                PetscCall(MatDenseRestoreColumnVecWrite(W, j - ini, &w));
                if (norm <= 0.0 || breakdown)
                {
                    PetscCall(PetscInfo(eps, "Orthogonalization of preconditioned residual failed\n"));
                    eps->reason = EPS_DIVERGED_BREAKDOWN;
                    goto diverged;
                }
            }
        }

        PetscCall(MatCopy(W, V, SAME_NONZERO_PATTERN));
        PetscCall(BVRestoreMat(R, &V));

        /* 11. B-orthonormalize preconditioned residuals */
        PetscCall(BVOrthogonalize(R, NULL));

        /* 13-16. B-orthonormalize conjugate directions */
        if (its > 1)
            PetscCall(BVOrthogonalize(P, NULL));

        /* 17-23. Compute symmetric Gram matrices */
        PetscCall(BVSetActiveColumns(Z, 0, ctx->bs));
        PetscCall(BVSetActiveColumns(X, 0, ctx->bs));
        PetscCall(BVCopy(X, Z));
        PetscCall(BVSetActiveColumns(Z, ctx->bs, 2 * ctx->bs - ini));
        PetscCall(BVCopy(R, Z));
        if (its > 1)
        {
            PetscCall(BVSetActiveColumns(Z, 2 * ctx->bs - ini, 3 * ctx->bs - 2 * ini));
            PetscCall(BVCopy(P, Z));
        }
        if (its > 1)
            nv = 3 * ctx->bs - 2 * ini;
        else
            nv = 2 * ctx->bs - ini;

        PetscCall(BVSetActiveColumns(Z, 0, nv));
        nv -= SpecialOrthogonalize(Z);

        PetscCall(DSSetDimensions(eps->ds, nv, 0, 0));
        PetscCall(DSGetMat(eps->ds, DS_MAT_A, &M));
        PetscCall(BVMatProject(Z, A, Z, M));
        if (flip)
            PetscCall(MatScale(M, -1.0));
        PetscCall(DSRestoreMat(eps->ds, DS_MAT_A, &M));
        if (B)
        {
            PetscCall(DSGetMat(eps->ds, DS_MAT_B, &M));
            PetscCall(BVMatProject(Z, B, Z, M)); /* covers also the case B=nullptr */
            PetscCall(DSRestoreMat(eps->ds, DS_MAT_B, &M));
        }
        else
        {
            PetscCall(DSSetIdentity(eps->ds, DS_MAT_B));
        }
        /* 24. Solve the generalized eigenvalue problem */
        PetscCall(DSSetState(eps->ds, DS_STATE_RAW));
        PetscCall(DSSolve(eps->ds, eigr, NULL));
        PetscCall(DSSort(eps->ds, eigr, NULL, NULL, NULL, NULL));
        PetscCall(DSSynchronize(eps->ds, eigr, NULL));
        for (j = 0; j < nv; j++)
            if (locked + j < eps->ncv)
                eps->eigr[locked + j] = flip ? -eigr[j] : eigr[j];
        PetscCall(DSVectors(eps->ds, DS_MAT_X, NULL, NULL));
        /* 25-33. Compute Ritz vectors */

        PetscCall(DSGetMat(eps->ds, DS_MAT_X, &M));
        PetscCall(BVSetActiveColumns(Z, ctx->bs, nv));
        if (ctx->lock)
            PetscCall(BVSetActiveColumns(P, 0, ctx->bs));
        PetscCall(BVMult(P, 1.0, 0.0, Z, M));
        PetscCall(BVCopy(P, X));
        if (ctx->lock)
            PetscCall(BVSetActiveColumns(P, nconv, ctx->bs));
        PetscCall(BVSetActiveColumns(Z, 0, ctx->bs));
        PetscCall(BVMult(X, 1.0, 1.0, Z, M));
        if (ctx->lock)
            PetscCall(BVSetActiveColumns(X, nconv, ctx->bs));
        PetscCall(BVMatMult(X, A, AX));
        PetscCall(DSRestoreMat(eps->ds, DS_MAT_X, &M));
    }

diverged:
    eps->its += its;

    if (flip)
        sc->comparison = SlepcCompareLargestReal;
    PetscCall(PetscFree(eigr));
    PetscCall(MatDestroy(&W));
    if (V)
        PetscCall(BVRestoreMat(R, &V)); /* only needed when goto diverged is reached */
    PetscCall(BVDestroy(&Z));
    PetscCall(BVDestroy(&X));
    PetscCall(BVDestroy(&R));
    PetscCall(BVDestroy(&P));
    PetscCall(BVDestroy(&AX));
    if (B)
        PetscCall(BVDestroy(&BX));
    if (nc > 0 || eps->nev > ctx->bs - ctx->guard)
        PetscCall(BVDestroy(&Y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
    Mat A, B; /* operator matrix */
    EPS eps;  /* eigenproblem solver context */
    EPSType type;
    PetscReal tol;
    PetscInt nev, maxit, its;
    char filenameA[PETSC_MAX_PATH_LEN];
    char filenameB[PETSC_MAX_PATH_LEN];
    char my_pc[10], res_out_name[20];
    PetscViewer viewer;
    PetscBool flgA, flgB, terse, flg;
    PetscCall(SlepcInitialize(&argc, &argv, (char *)0, help));
    ST st; /* spectral transformation context */
    KSP ksp;
    PC pc;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Load the operator matrix that defines the eigensystem, Ax=kx
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-fileA", filenameA, sizeof(filenameA), &flgA));
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-fileB", filenameB, sizeof(filenameB), &flgB));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nmatrix:%s\n", filenameA));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nEigenproblem stored in file.\n\n"));
    PetscCheck(flgA, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a file name with the -file option");
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Reading COMPLEX matrix from a binary file...\n"));
#else
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Reading REAL matrix from a binary file...\n"));
#endif
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filenameA, FILE_MODE_READ, &viewer)); // 先读入viewer
    // PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    // MatSetType(A,MATAIJCUSPARSE);
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, viewer)); // 再load
    PetscCall(PetscViewerDestroy(&viewer));

    if (flgB)
    {
        PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filenameB, FILE_MODE_READ, &viewer)); // 先读入viewer
        // PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer));
        PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
        PetscCall(MatSetFromOptions(B));
        PetscCall(MatLoad(B, viewer)); // 再load
        PetscCall(PetscViewerDestroy(&viewer));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Reading complete!\n"));

    // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF));

    PetscInt n_ev, bs, n_r, guard, ksp_iter, vec_num, nnz, n;
    PetscScalar cg_tol, *conv_eigr;
    std::string pc_type, res_file;
    MatInfo info;

    n_ev = 15;
    bs = 10;
    n_r = 3;
    tol = 1e-3;
    guard = 3;
    ksp_iter = 10;

    cg_tol = 1e-30; // 1e-12
    pc_type = "GS"; // JACOBI GS SPAI AMG ILU

    PetscCall(MatGetSize(A, &n, &n)); // 获取问题的阶数，存在n里
    MatGetInfo(A, MAT_GLOBAL_SUM, &info);
    nnz = info.nz_allocated;
    conv_eigr = new PetscScalar[n_ev];

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create the eigensolver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Create eigensolver context
    */
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));

    /*
       Set operators. In this case, it is a standard eigenvalue problem
    */
    if (flgB)
    {
        EPSSetOperators(eps, A, B);
    }
    else
    {
        EPSSetOperators(eps, A, nullptr);
    }
    /*
       Set solver parameters at runtime
    */
    PetscCall(PetscOptionsSetValue(nullptr, "-eps_nev", std::to_string(n_ev).c_str()));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the eigensystem
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STPRECOND));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetType(eps, EPSLOBPCG)); // EPSLANCZOS  EPSBLOPEX EPSJD EPSLOBPCG
    EPSSetTolerances(eps, tol, 5000);      /// 设置误差

    PetscOptionsGetInt(nullptr, nullptr, "-iter", &ksp_iter, &flg);
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-PC", my_pc, sizeof(my_pc), &flg));

    if (flg)
    {
        pc_type = my_pc;
    }

    PetscCall(EPSSetFromOptions(eps));

    eps->ops->solve = EPSSolve_LOBPCG;

    STGetKSP(st, &ksp);
    PetscCall(KSPSetType(ksp, KSPCG));
    KSPGetPC(ksp, &pc);
    // PCSetOperators()
    PCSetType(pc, PCJACOBI);
    PetscCall(KSPSetTolerances(ksp, cg_tol, cg_tol, PETSC_DEFAULT, ksp_iter)); // 这个地方设置CG求解的最大迭代次数

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " 准备求解!\n"));
    double start = MPI_Wtime();
    PetscCall(EPSSolve(eps));
    double end = MPI_Wtime(); // 计时开始！！
    PetscCall(EPSGetIterationNumber(eps, &its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Number of iterations of the method: %" PetscInt_FMT "\n", its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " time cost: %g \n", end - start));

    /*
       Optional: Get some information from the solver and display it
    */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Display solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*记录运行结果*/

    /* show detailed info unless -terse option is given by user */
    PetscCall(PetscOptionsHasName(nullptr, nullptr, "-terse", &terse));
    if (terse)
        PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE, nullptr));
    else
    {
        PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL));
        PetscCall(EPSConvergedReasonView(eps, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(EPSErrorView(eps, EPS_ERROR_RELATIVE, PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A));
    PetscCall(SlepcFinalize());
    return 0;
}
/*TEST

   test:
      suffix: 1
      args: -file ${SLEPC_DIR}/share/slepc/datafiles/matrices/rdb200.petsc -eps_nev 4 -terse
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

   testset:
      args: -file ${DATAFILESPATH}/matrices/complex/qc324.petsc -eps_type ciss -terse
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      test:
         suffix: ciss_1
         args: -rg_type ellipse -rg_ellipse_center -.012-.08i -rg_ellipse_radius .05
      test:
         suffix: ciss_2
         args: -rg_type interval -rg_interval_endpoints -0.062,.038,-.13,-.03 -eps_max_it 1

TEST*/