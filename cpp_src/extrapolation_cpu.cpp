
#include "wfPad.h"
#include "mkl_linalg.h"
#include <vector>
#include "timer.h"

extern "C"
{

// propgation and imaging of two wavefields using padded zeros in halo regions
void extrapolate(int ns, int nextrap, int nz, int nx, int nf, int nt, int M,\
    fcomp * w_op_fs_forw, fcomp * w_op_fs_back,\
    fcomp * wave_fs, fcomp * shot_fs,\
    float * image)
{
    int length_M = 2*M + 1;
    int dim_x = nx+2*M;
    size_t sizeImage = nz * nx;
    timer t0("EXTRAPOLATION");
    timer t1("IMAGING");
    timer t2("CONSTRUCT PADDED WAVEFIELDS");
    timer t3("WRITE-BACK UNPADDED WAVEFIELDS");

    t2.start();
    std::vector<wfpad> new_forw(ns);
    std::vector<wfpad> old_forw(ns);
    std::vector<wfpad> new_back(ns);
    std::vector<wfpad> old_back(ns);
    for(int is=0; is<ns; ++is){
        old_forw[is] = wfpad(nf, nx, 1, M, 0, &wave_fs[is*nt*nx]);
        new_forw[is] = wfpad(nf, nx, 1, M, 0);
        old_back[is] = wfpad(nf, nx, 1, M, 0, &shot_fs[is*nt*nx]);
        new_back[is] = wfpad(nf, nx, 1, M, 0);
    }
    t2.stop();

    fcomp * conv = new fcomp[ns*nx];
    
    for (int l=0; l<nextrap; ++l){
        long int depthIdx = l*nf*nx*length_M;

        t0.start();
        #pragma omp parallel for schedule(dynamic,1)
        for (int j=0; j<nf; ++j){
            long int freqIdx = j*nx*length_M;

            for (int is=0; is<ns; ++is){
         
                for (int i=0; i<nx; ++i){                
                    long int locIdx = i*length_M;

                    c64dot(&w_op_fs_forw[depthIdx + freqIdx + locIdx], &old_forw[is].wf[j*dim_x + i], length_M, &new_forw[is].wf[j*dim_x + i + M]);
                    c64dot(&w_op_fs_back[depthIdx + freqIdx + locIdx], &old_back[is].wf[j*dim_x + i], length_M, &new_back[is].wf[j*dim_x + i + M]);

                } //end loop over locations

            } //end loop over sources
            
        } //end loop over frequencies
        t0.stop();

        for (int is=0; is<ns; ++is){
            std::swap(old_forw[is].wf, new_forw[is].wf);
            std::swap(old_back[is].wf, new_back[is].wf);
        }

        t1.start();
        /*-------------IMAGING--------------*/
        #pragma omp parallel for schedule(dynamic,1)
        for (int is=0; is<ns; ++is)
            for (int i=0; i<nx; i++)
                conv[is*nx + i] = fcomp(0.0,0.0);

        #pragma omp parallel for schedule(dynamic,1)
        for (int is=0; is<ns; ++is)
            for (int j=0; j<nf; j++)
                for (int i=0; i<nx; i++){
                    conv[is*nx + i] += old_forw[is].wf[j*dim_x+i+M] * thrust::conj(old_back[is].wf[j*dim_x+i+M]);
                }

        #pragma omp parallel for schedule(dynamic,1)
        for (int is=0; is<ns; ++is)
            for (int i=0; i<nx; i++){
                image[is*sizeImage + l*nx+i] = reinterpret_cast<float*>(&conv[is*nx])[2*i];
            }
        /*------------END IMAGING------------*/
        t1.stop();
    
	} //end loop over depths

    t3.start();
    #pragma omp parallel for schedule(dynamic,1)
    for (int is=0; is<ns; ++is)
        for (int j=0; j<nf; ++j)
            for (int i=0; i<nx; ++i){
                wave_fs[is*nt*nx + j*nx + i] = old_forw[is].wf[j*dim_x + i + M];
                shot_fs[is*nt*nx + j*nx + i] = old_back[is].wf[j*dim_x + i + M];
            }
    t3.stop();

	delete [] conv;

    t0.dispInfo();
    t1.dispInfo();
    t2.dispInfo();
    t3.dispInfo();
}


/*-----------------------------------------------------------------------------------
 * This version performs extrapolation on CPU as version 1 above, however it is not
 * optimized for cache reuse of operators over all sources.
 */

void extrapolate_v0(int ns, int nextrap, int nz, int nx, int nf, int nt, int M,\
    fcomp * w_op_fs_forw, fcomp * w_op_fs_back,\
    fcomp * wave_fs, fcomp * shot_fs,\
    float * image)
{
    int length_M = 2*M + 1;
    int dim_x = nx+2*M;
    size_t sizeImage = nz * nx;
    timer t0("EXTRAPOLATION");
    timer t1("IMAGING");
    timer t2("CONSTRUCT PADDED WAVEFIELDS");
    timer t3("WRITE-BACK UNPADDED WAVEFIELDS");

    t2.start();
    std::vector<wfpad> new_forw(ns);
    std::vector<wfpad> old_forw(ns);
    std::vector<wfpad> new_back(ns);
    std::vector<wfpad> old_back(ns);
    for(int is=0; is<ns; ++is){
        old_forw[is] = wfpad(nf, nx, 1, M, 0, &wave_fs[is*nt*nx]);
        new_forw[is] = wfpad(nf, nx, 1, M, 0);
        old_back[is] = wfpad(nf, nx, 1, M, 0, &shot_fs[is*nt*nx]);
        new_back[is] = wfpad(nf, nx, 1, M, 0);
    }
    t2.stop();

    fcomp * conv = new fcomp[ns*nx];
    
    for (int l=0; l<nextrap; ++l){
        long int depthIdx = l*nf*nx*length_M;

        t0.start();
        for (int is=0; is<ns; ++is){

            #pragma omp parallel for schedule(dynamic,1)
            for (int j=0; j<nf; ++j){
                long int freqIdx = j*nx*length_M;

                for (int i=0; i<nx; ++i){                
                    long int locIdx = i*length_M;

                    c64dot(&w_op_fs_forw[depthIdx + freqIdx + locIdx], &old_forw[is].wf[j*dim_x + i], length_M, &new_forw[is].wf[j*dim_x + i + M]);
                    c64dot(&w_op_fs_back[depthIdx + freqIdx + locIdx], &old_back[is].wf[j*dim_x + i], length_M, &new_back[is].wf[j*dim_x + i + M]);

                } //end loop over locations

            } //end loop over frequencies
            
        } //end loop over sources
        t0.stop();

        for (int is=0; is<ns; ++is){
            std::swap(old_forw[is].wf, new_forw[is].wf);
            std::swap(old_back[is].wf, new_back[is].wf);
        }

        t1.start();
        /*-------------IMAGING--------------*/
        #pragma omp parallel for schedule(dynamic,1)
        for (int is=0; is<ns; ++is)
            for (int i=0; i<nx; i++)
                conv[is*nx + i] = fcomp(0.0,0.0);

        #pragma omp parallel for schedule(dynamic,1)
        for (int is=0; is<ns; ++is)
            for (int j=0; j<nf; j++)
                for (int i=0; i<nx; i++){
                    conv[is*nx + i] += old_forw[is].wf[j*dim_x+i+M] * thrust::conj(old_back[is].wf[j*dim_x+i+M]);
                }

        #pragma omp parallel for schedule(dynamic,1)
        for (int is=0; is<ns; ++is)
            for (int i=0; i<nx; i++){
                image[is*sizeImage + l*nx+i] = reinterpret_cast<float*>(&conv[is*nx])[2*i];
            }
        /*------------END IMAGING------------*/
        t1.stop();
    
	} //end loop over depths

    t3.start();
    #pragma omp parallel for schedule(dynamic,1)
    for (int is=0; is<ns; ++is)
        for (int j=0; j<nf; ++j)
            for (int i=0; i<nx; ++i){
                wave_fs[is*nt*nx + j*nx + i] = old_forw[is].wf[j*dim_x + i + M];
                shot_fs[is*nt*nx + j*nx + i] = old_back[is].wf[j*dim_x + i + M];
            }
    t3.stop();

	delete [] conv;

    t0.dispInfo();
    t1.dispInfo();
    t2.dispInfo();
    t3.dispInfo();
}

} // end extern C

