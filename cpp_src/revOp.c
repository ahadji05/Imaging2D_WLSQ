
#include "revOp.h"

fcomp * reverseOperator(fcomp * op, int nextrap, int nf, int nx, int length_M)
{
    long int sizeOp = nextrap*nf*nx*length_M;
    fcomp * op_rev = new fcomp[sizeOp];
    
    #pragma omp parallel for
    for(int l=0; l<nextrap; ++l){
        int depthIdx = l*nf*nx*length_M;

        for(int j=0; j<nf; ++j){
            int freqIdx = j*nx*length_M;
            
            for(int i=0; i<nx; ++i){

                for(int k=0; k<length_M; ++k){

                    op_rev[depthIdx + freqIdx + k*nx + i] = op[depthIdx + freqIdx + i*length_M + k];
                }
            }
        }
    }

    return op_rev;
}