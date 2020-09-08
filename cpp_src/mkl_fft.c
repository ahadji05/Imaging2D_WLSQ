
#include "types.h"
#include "mkl.h"

extern "C"

{

void c64fft1dforw(fcomp * data, int N){
    
    MKL_LONG length = N;

    DFTI_DESCRIPTOR_HANDLE dfti_handle;

    DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, length);

    DftiCommitDescriptor(dfti_handle);

    DftiComputeForward(dfti_handle, data);

    DftiFreeDescriptor(&dfti_handle);
}

void c64fft1dback(fcomp * data, int N){

    MKL_LONG length = N;

    DFTI_DESCRIPTOR_HANDLE dfti_handle;

    DftiCreateDescriptor(&dfti_handle, DFTI_SINGLE, DFTI_COMPLEX, 1, length);

    DftiSetValue(dfti_handle, DFTI_BACKWARD_SCALE, 1.0/(float)(N));
    
    DftiCommitDescriptor(dfti_handle);
    
    DftiComputeBackward(dfti_handle, data);
    
    DftiFreeDescriptor(&dfti_handle);
}

} //end extern C

