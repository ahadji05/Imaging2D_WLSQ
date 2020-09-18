
#include "wfPad.h"
#include "mkl_linalg.h"

extern "C"
{

// propgation and imaging of two wavefields using padded zeros in halo regions
void extrapolate(int nextrap, int nz, int nx, int nf, int nt, int M,\
    fcomp * w_op_fs_forw, fcomp * w_op_fs_back,\
    fcomp * wave_fs, fcomp * shot_fs,\
    float * image)
{   
    int length_M = 2*M + 1;
    int dim_x = nx+2*M;

    wfpad new_forw(nf, nx, 1, M, 0);
    wfpad old_forw(nf, nx, 1, M, 0, wave_fs);
    wfpad new_back(nf, nx, 1, M, 0);
    wfpad old_back(nf, nx, 1, M, 0, shot_fs);

    fcomp * conv = new fcomp[nx];
    
    for (int l=0; l<nextrap; ++l){
        long int depthIdx = l*nf*nx*length_M;

        #pragma omp parallel for
        for (int j=0; j<nf; ++j){
            long int freqIdx = j*nx*length_M;
         
            for (int i=0; i<nx; ++i){                
                long int locIdx = i*length_M;

                c64dot(&w_op_fs_forw[depthIdx+freqIdx+locIdx], &old_forw.wf[j*dim_x + i], length_M, &new_forw.wf[j*dim_x + i + M]);
                c64dot(&w_op_fs_back[depthIdx+freqIdx+locIdx], &old_back.wf[j*dim_x + i], length_M, &new_back.wf[j*dim_x + i + M]);

            } //end loop over frequencies
            
        } //end loop over locations

        #pragma omp parallel for
        for (int j=0; j<nf; ++j){
            for (int i=0; i<dim_x; ++i){
                old_forw.wf[j*dim_x+i] = new_forw.wf[j*dim_x+i];
                old_back.wf[j*dim_x+i] = new_back.wf[j*dim_x+i];
            }
        }

        /*-------------IMAGING--------------*/
        for (int i=0; i<nx; i++)
            conv[i] = fcomp(0.0,0.0);

        for (int j=0; j<nf; j++){
            for (int i=0; i<nx; i++){
                conv[i] += old_forw.wf[j*dim_x+i+M] * thrust::conj(old_back.wf[j*dim_x+i+M]);
            }
        }
        for (int i=0; i<nx; i++){
            image[l*nx+i] = reinterpret_cast<float*>(conv)[2*i];
        }
        /*------------END IMAGING------------*/
    
	} //end loop over depths

    for (int j=0; j<nf; ++j){
        for (int i=0; i<nx; ++i){
            wave_fs[j*nx+i] = old_forw.wf[j*dim_x+i+M];
            shot_fs[j*nx+i] = old_back.wf[j*dim_x+i+M];
        }
    }
	delete [] conv;

}

} // end extern C

