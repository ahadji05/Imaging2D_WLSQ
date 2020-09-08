
#include "types.h"
#include "mkl.h"

extern "C"
{

void c64dot( const fcomp * v1, const fcomp * v2, const int  N,\
    fcomp * result );

void c64matvec( const fcomp * mat, const fcomp * vec, const int nrows,\
    const int ncolms, fcomp * result);

} // end extern "C"
