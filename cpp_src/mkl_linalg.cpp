
#include "mkl_linalg.h"

extern "C"
{

void c64dot( const fcomp * v1, const fcomp * v2, const int  N,\
    fcomp * result )
{
    cblas_cdotu_sub(N, v1, 1, v2, 1, result);
}

void c64matvec( const fcomp * mat, const fcomp * vec, const int nrows,\
    const int ncolms, fcomp * result)
{
    const CBLAS_LAYOUT layout = CblasRowMajor;
    const CBLAS_TRANSPOSE transp = CblasNoTrans;
    const std::complex<float> alpha = fcomp(1.0,0.0);
    const std::complex<float> beta = fcomp(0.0,0.0);

    cblas_cgemv(layout, transp, nrows, ncolms, &alpha, mat, ncolms, vec, 1, &beta, result ,1);
}

} //end extern "C"
