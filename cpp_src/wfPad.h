#ifndef WFPAD_H
#define WFPAD_H

#include <iostream>
#include "types.h"
#include <cstdlib>

extern "C"
{

//----------------------------------------------------------------------------
//this class represents wavefields of dimensionality = nf*nx*ny
class wfpad {

    public:
    
        size_t nf, nx, ny, Mx, My, dim_x, dim_y; //dimensions
        size_t wfSize; //size of dimensions
        fcomp * wf; //data

        //constructors
        wfpad();
        wfpad(size_t nf, size_t nx, size_t ny, size_t Mx, size_t My);
        wfpad(size_t nf, size_t nx, size_t ny, size_t Mx, size_t My, fcomp * data);

        //copy constructor
        wfpad(const wfpad & wfOther);

        //copy assignment operators
        wfpad & operator=(const wfpad & wfOther);
        wfpad & operator=(wfpad & wfOther);

        //move constructor
        wfpad(wfpad && wfOther);

        //move assignment operator
        wfpad & operator=(wfpad && wfOther);

        //destructor
        ~wfpad();

        void dispInfo();

        inline wfpad & operator+=(const wfpad & wfOther);
};

wfpad operator+(const wfpad & a, const wfpad & b);

wfpad::wfpad()
    : nf(0), nx(0), ny(0), Mx(0), My(0)
{

#ifdef debugClasses
    std::cout << "wavefield: default constructor" << std::endl;
#endif

    wfSize = 0;
    dim_x = 0;
    dim_y = 0;
    wf = nullptr;
}

//----------------------------------------------------------------------------

wfpad::wfpad(size_t nf, size_t nx, size_t ny, size_t Mx, size_t My)
    : nf(nf), nx(nx), ny(ny), Mx(Mx), My(My) {

#ifdef debugClasses
    std::cout << "wavefield: constructor 1" << std::endl;
#endif
    
    size_t dim_x = nx+2*Mx;
    size_t dim_y = ny+2*My;
    wfSize = nf*dim_x*dim_y;

    wf = new fcomp[wfSize];

    //set zeros in halo regions
    for(size_t j=0; j<nf; ++j)
        for(size_t iy=0; iy<My; ++iy)
            for(size_t ix=0; ix<Mx; ++ix){
                wf[j*dim_y*dim_x + iy*dim_x + ix] = fcomp(0.0,0.0);
                wf[j*dim_y*dim_x + iy*dim_x + nx + Mx + ix] = fcomp(0.0,0.0);
    }
}

//----------------------------------------------------------------------------

wfpad::wfpad(size_t nf, size_t nx, size_t ny, size_t Mx, size_t My, fcomp * data)
    : nf(nf), nx(nx), ny(ny), Mx(Mx), My(My) {

#ifdef debugClasses
    std::cout << "wavefield: constructor 2" << std::endl;
#endif

    size_t dim_x = nx+2*Mx;
    size_t dim_y = ny+2*My;
    wfSize = nf*dim_x*dim_y;

    wf = new fcomp[wfSize];

    //set zeros in halo regions
    for(size_t j=0; j<nf; ++j)
        for(size_t iy=0; iy<My; ++iy)
            for(size_t ix=0; ix<Mx; ++ix){
                wf[j*dim_y*dim_x + iy*dim_x + ix] = fcomp(0.0,0.0);
                wf[j*dim_y*dim_x + iy*dim_x + nx + Mx + ix] = fcomp(0.0,0.0);
            }
    //read wavefield in the non-halo regions
    for(size_t j=0; j<nf; ++j)
        for(size_t iy=0; iy<ny; ++iy)
            for(size_t ix=0; ix<nx; ++ix)
                wf[j*dim_y*dim_x + (iy+My)*dim_x + ix + Mx] = data[j*nx*ny + iy*nx + ix];

}

//----------------------------------------------------------------------------

wfpad::wfpad(const wfpad & wfOther)
    : nf(wfOther.nf), nx(wfOther.nx), ny(wfOther.ny), Mx(wfOther.Mx), \
    My(wfOther.My), dim_x(wfOther.dim_x), dim_y(wfOther.dim_y), \
    wfSize(wfOther.wfSize) {

#ifdef debugClasses
    std::cout << "wavefield: copy constructor" << std::endl;
#endif

    wf = new fcomp[wfSize];

    for(size_t i=0; i<wfSize; ++i)
        wf[i] = wfOther.wf[i];
}

//----------------------------------------------------------------------------

wfpad & wfpad::operator=(const wfpad & wfOther){

#ifdef debugClasses
    std::cout << "wavefield: copy assignment operator 1" << std::endl;
#endif

    nf = wfOther.nf;
    nx = wfOther.nx;
    ny = wfOther.ny;
    Mx = wfOther.Mx;
    My = wfOther.My;
    dim_x = wfOther.dim_x;
    dim_y = wfOther.dim_y;
    wfSize = wfOther.wfSize;

    wf = new fcomp[wfSize];

    for(size_t i=0; i<wfSize; ++i)
        wf[i] = wfOther.wf[i];

    return *this;
}

//----------------------------------------------------------------------------

wfpad & wfpad::operator=(wfpad & wfOther){

#ifdef debugClasses
    std::cout << "wavefield: copy assignment operator 2" << std::endl;
#endif

    nf = wfOther.nf;
    nx = wfOther.nx;
    ny = wfOther.ny;
    Mx = wfOther.Mx;
    My = wfOther.My;
    dim_x = wfOther.dim_x;
    dim_y = wfOther.dim_y;
    wfSize = wfOther.wfSize;

    wf = new fcomp[wfSize];

    for(size_t i=0; i<wfSize; ++i)
        wf[i] = wfOther.wf[i];

    return *this;
}

//----------------------------------------------------------------------------

wfpad::wfpad(wfpad && wfOther)
    : nf(wfOther.nf), nx(wfOther.nx), ny(wfOther.ny), Mx(wfOther.Mx), \
    My(wfOther.My), dim_x(wfOther.dim_x), dim_y(wfOther.dim_y), \
    wfSize(wfOther.wfSize) {

#ifdef debugClasses
    std::cout << "wavefield: move constructor" << std::endl;
#endif

    wf = wfOther.wf;

    wfOther.wf = nullptr;
    wfOther.nf = 0;
    wfOther.nx = 0;
    wfOther.ny = 0;
    wfOther.Mx = 0;
    wfOther.My = 0;
    wfOther.dim_x = 0;
    wfOther.dim_y = 0;
    wfOther.wfSize = 0;
}

//----------------------------------------------------------------------------

wfpad & wfpad::operator=(wfpad && wfOther){

#ifdef debugClasses
    std::cout << "wavefield: move assignment operator" << std::endl;
#endif

    if(this != &wfOther){

        delete [] wf;

        wf = wfOther.wf;
        nf = wfOther.nf;
        nx = wfOther.nx;
        ny = wfOther.ny;
        Mx = wfOther.Mx;
        My = wfOther.My;
        dim_x = wfOther.dim_x;
        dim_y = wfOther.dim_y;
        wfSize = wfOther.wfSize;

        wfOther.wf = nullptr;
        wfOther.nf = 0;
        wfOther.nx = 0;
        wfOther.ny = 0;
        wfOther.Mx = 0;
        wfOther.My = 0;
        wfOther.dim_x = 0;
        wfOther.dim_y = 0;
        wfOther.wfSize = 0;
    }

    return *this;
}

//----------------------------------------------------------------------------

wfpad::~wfpad(){

#ifdef debugClasses
    std::cout << "wavefield: destructor";
    std::cout << " - data = "<< this->wfSize*sizeof(fcomp);
    std::cout << " bytes." << std::endl;
#endif

    delete [] wf;
}

//----------------------------------------------------------------------------

void wfpad::dispInfo(){
    std::cout << "nx = " << nx << std::endl;
    std::cout << "ny = " << ny << std::endl;
    std::cout << "nf = " << nf << std::endl;
    std::cout << "wfSize = " << wfSize << std::endl;
}

//----------------------------------------------------------------------------

inline wfpad & wfpad::operator+=(const wfpad & wfOther){

#ifdef debugClasses
    std::cout << "wavefield: increment class" << std::endl;
#endif

#ifdef debugConditionals
    if( this->wfSize != wfOther.wfSize)
    if( this->nx != wfOther.nx || this->ny != wfOther.ny || this->nf != wfOther.nf ){
        std::cout << "wf: invalid classes increment\n";
        throw "invalid-operation";
        exit(1);
    }
#endif

    for(unsigned int i=0; i<this->wfSize; i++)
            this->wf[i] += wfOther.wf[i];

    return *this;
}

//----------------------------------------------------------------------------

inline wfpad operator+(const wfpad & a, const wfpad & b){

#ifdef debugClasses
    std::cout << "wavefield: classes addition" << std::endl;
#endif

    wfpad c(a.nf, a.nx, a.ny, a.Mx, a.My);

#ifdef debugConditionals
    if( a.nf != b.nf || a.nx != b.nx || a.ny != b.ny || a.Mx != b.Mx || a.My != b.My){
        std::cout << "wf: invalid classes addition\n";
        throw "invalid-operation";
        exit(1);
    }
#endif

    for(unsigned int i=0; i<c.wfSize; i++)
            c.wf[i] = a.wf[i] + b.wf[i];

    return c;
}

//----------------------------------------------------------------------------

} //end extern "C"

#endif
