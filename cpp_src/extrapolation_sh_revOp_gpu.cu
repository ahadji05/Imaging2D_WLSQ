
#include <iostream>
#include <vector>
#include <cmath>

#include "wfPad.h"
#include "revOp.h"
#include "timer.h"

extern "C"
{

/*
-----------------------------------------------------------------
*/
__global__ void copyPadded(fcomp * paste, fcomp * copyied, \
    int fIdx, int nx, int M, int ns, int sizePulse)
{
    int dim_x = nx+2*M;
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if( xIdx < nx ){

        for(int is=0; is<ns; ++is){

            int srcIdx = is*sizePulse;
            int pixelIdx = srcIdx + fIdx * dim_x + xIdx + M;
        
            paste[pixelIdx] = copyied[pixelIdx];
        }
    }
}

/*
------------------------------------------------------------------
*/
__global__ void imaging(fcomp * image, fcomp * forw_pulse, fcomp * back_pulse, \
    int nf, int nx, int M, int depth_l, int sizePulse, int sizeImage)
{
    int dim_x = nx+2*M;
    int xIdx = threadIdx.x;
    int sIdx = blockIdx.x;

    fcomp conv = fcomp(0.0,0.0);

    for(int j=0; j<nf; j++){
        int Idx = sIdx*sizePulse + j*dim_x + xIdx + M;
        conv += forw_pulse[Idx] * thrust::conj(back_pulse[Idx]);
    }

    image[sIdx*sizeImage + depth_l*nx + xIdx] = conv;
}

/*
------------------------------------------------------------------
*/
__global__ void extrapDepth(fcomp * new_wf, int nf, int nx, \
    int M, int fIdx, int ns, int sizePulse, fcomp * w_op, fcomp * old_wf)
{
    int dim_x = nx+2*M;
    int length_M = 2*M+1;
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    fcomp pixel;

    if( xIdx < nx ){

        for(int is=0; is<ns; ++is){

            int srcIdx = is*sizePulse;

            pixel = fcomp(0.0,0.0);

            for(int k=0; k<length_M; ++k){
                pixel += w_op[k*nx + xIdx] * old_wf[srcIdx + fIdx*dim_x + xIdx + k];
            }

            new_wf[srcIdx + fIdx*dim_x + M + xIdx] = pixel;
        }
    }

} // end extrapolation to next depth


/*
------------------------------------------------------------------
*/
void extrapolate(int ns, int nextrap, int nz, int nt, int nf, int nx, int M,\
    fcomp * w_op_forw, fcomp * forw_pulse, fcomp * w_op_back, fcomp * back_pulse,\
    float * image)
{
    //define important dimensionality parameters
    int length_M = 2*M+1;
    int dim_x = nx+2*M;

    size_t sizePulse = nf * dim_x;
    size_t sizeAllSources = ns * sizePulse;
    size_t sizeImage = nz * nx;
    size_t sizeAllImages = ns * sizeImage;

    //rearrange operators
    timer t0("REARRANGE OPERATORS");
    fcomp * h_w_op_forw = reverseOperator(w_op_forw, nextrap, nf, nx, length_M, t0); //reverse operator's last two indices on host
    fcomp * h_w_op_back = reverseOperator(w_op_back, nextrap, nf, nx, length_M, t0); //reverse operator's last two indices on host

    //allocate device memory
    fcomp * d_image;
    cudaMalloc(&d_image, sizeAllImages * sizeof(fcomp));

    fcomp * d_old_forw, * d_new_forw, * d_old_back, * d_new_back;
    cudaMalloc(&d_old_forw, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_new_forw, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_old_back, sizeAllSources * sizeof(fcomp));
    cudaMalloc(&d_new_back, sizeAllSources * sizeof(fcomp));
    
    timer t1("CONSTRUCT PADDED WAVEFIELDS");
    t1.start();
    //allocate and read wavefields
    fcomp * h_image = new fcomp[sizeAllImages];
    std::vector<wfpad> h_forw_pulses(ns);
    std::vector<wfpad> h_back_pulses(ns);
    for(int is=0; is<ns; ++is){
        h_forw_pulses[is] = wfpad(nf, nx, 1, M, 0, &forw_pulse[is*nt*nx]);
        h_back_pulses[is] = wfpad(nf, nx, 1, M, 0, &back_pulse[is*nt*nx]);
        cudaMemcpy(&d_old_forw[is*sizePulse], h_forw_pulses[is].wf, sizePulse * sizeof(fcomp), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_old_back[is*sizePulse], h_back_pulses[is].wf, sizePulse * sizeof(fcomp), cudaMemcpyHostToDevice);
    }
    t1.stop();

    size_t sizeOp = nf * nx * length_M; //note that we allocate memory for one depth only!!
    fcomp * d_w_op_forw, * d_w_op_back;
    cudaMalloc(&d_w_op_forw, sizeOp * sizeof(fcomp));
    cudaMalloc(&d_w_op_back, sizeOp * sizeof(fcomp));

    //define number of blocks and number of threads per block
    //define number of blocks and number of threads per block
    int x_stride = 32;
    int sh_per_block = x_stride*length_M*sizeof(fcomp);
    dim3 nThreads(x_stride, 1, 1);
    size_t nBlocks_x = nx % nThreads.x == 0 ? size_t(nx/nThreads.x) : size_t(1 + nx/nThreads.x);
    size_t nBlocks_y = 1;
    size_t nBlocks_z = 1;
    dim3 nBlocks(nBlocks_x, nBlocks_y, nBlocks_z);
    std::cout << "nThreads: (" << nThreads.x << ", " << nThreads.y << ", " << nThreads.z << ")" << std::endl;
    std::cout << "nBlocks: (" << nBlocks.x << ", " << nBlocks.y << ", " << nBlocks.z << ")" << std::endl;

    //create one stream per source
    cudaStream_t streams[nf];
    for(int j=0; j<nf; ++j)
        cudaStreamCreate(&streams[j]);

    timer t5("EXTRAPOLATION AND IMAGING");
    t5.start();    
    for(int l=0; l<nextrap; ++l){

        int depthIdx = l*nf*nx*length_M;

        for(int j=0; j<nf; ++j){

            int freqIdx = j*nx*length_M;

            cudaMemcpyAsync(&d_w_op_forw[freqIdx], &h_w_op_forw[depthIdx + freqIdx], \
                nx*length_M*sizeof(fcomp), cudaMemcpyHostToDevice, streams[j]);

            cudaMemcpyAsync(&d_w_op_back[freqIdx], &h_w_op_back[depthIdx + freqIdx], \
                nx*length_M*sizeof(fcomp), cudaMemcpyHostToDevice, streams[j]);

            extrapDepth<<<nBlocks, nThreads, 0, streams[j]>>>(d_new_forw, nf, nx, \
                M, j, ns, sizePulse, &d_w_op_forw[freqIdx], d_old_forw);
            
            extrapDepth<<<nBlocks, nThreads, 0, streams[j]>>>(d_new_back, nf, nx, \
                M, j, ns, sizePulse, &d_w_op_back[freqIdx], d_old_back);
            
            copyPadded<<<nBlocks, nThreads, 0, streams[j]>>>(d_old_forw, d_new_forw,\
                j, nx, M, ns, sizePulse);
            
            copyPadded<<<nBlocks, nThreads, 0, streams[j]>>>(d_old_back, d_new_back, \
                j, nx, M, ns, sizePulse);
            
        }
        //implicit synchronization using NULL stream (the default stream)
        imaging<<<ns, nx>>>(d_image, d_new_forw, d_new_back, \
            nf, nx, M, l, sizePulse, sizeImage);
    }

    for(int j=0; j<nf; ++j)
        cudaStreamDestroy(streams[j]);

    // copy data back to Host
    cudaMemcpy(h_image, d_image, sizeAllImages * sizeof(fcomp), cudaMemcpyDeviceToHost);
    for(int is=0; is<ns; ++is){
        cudaMemcpy(h_forw_pulses[is].wf, &d_new_forw[is*sizePulse], \
            sizePulse*sizeof(fcomp), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_back_pulses[is].wf, &d_new_back[is*sizePulse], \
            sizePulse*sizeof(fcomp), cudaMemcpyDeviceToHost);
    }
    t5.stop();


    timer t2("WRITE-BACK UNPADDED WAVEFIELDS");
    t2.start();
    //copy to unpadded memory
    for(int is=0; is<ns; ++is)
        for (int j=0; j<nf; ++j)
            for (int i=0; i<nx; ++i){
                forw_pulse[is*sizePulse + j*nx + i] = h_forw_pulses[is].wf[j*dim_x + i + M];
                back_pulse[is*sizePulse + j*nx + i] = h_back_pulses[is].wf[j*dim_x + i + M];
            }
    t2.stop();

    timer t3("READ IMAGES");
    t3.start();
    //take real part of images
    for(int is=0; is<ns; ++is)
        for(int l=0; l<nextrap; ++l)
            for(int i=0; i<nx; ++i){
                image[is*sizeImage + l*nx + i] = reinterpret_cast<float*>(h_image)[2*(is*sizeImage + l*nx + i)];
            }
    t3.stop();

    timer t4("FREE HOST MEMORY");
    t4.start();
    //free host memory
    delete [] h_image;
    delete [] h_w_op_forw;
    delete [] h_w_op_back;
    t4.stop();

    //free device memory
    cudaFree(d_w_op_forw);
    cudaFree(d_w_op_back);
    cudaFree(d_new_forw);
    cudaFree(d_old_forw);
    cudaFree(d_new_back);
    cudaFree(d_old_back);
    cudaFree(d_image);

    std::cout << std::endl;
    std::cout << "------- Timer info -------" << std::endl;
    std::cout << "--------------------------" << std::endl;
    t0.dispInfo();
    t1.dispInfo();
    t2.dispInfo();
    t3.dispInfo();
    t4.dispInfo();
    t5.dispInfo();

}

} //end extern "C"
