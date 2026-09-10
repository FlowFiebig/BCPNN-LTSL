/*****************************************************************

  Created: 2024-10-13  Modified: 2024-10-17

  Authors: Anders Lansner

******************************************************************/
/*****************************************************************

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
SuOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/

#include "AxoDelay.h"
#include "Prj2.h"
#include "Prj.cuh"

using namespace std;
using namespace Globals;

Prj2::Prj2(Pop *srcpop, Pop *trgpop, int nactHi, int nsilHi, std::string name)
    : Prj(srcpop, trgpop, nactHi, nsilHi, name, false) {
    if (Mi == 2)
        error("Prj2::Prj2","Illegal: srcpop already binarized (Mi == 2)");
    axoHi = srcN;
    Mi = 2;
    axoNi = axoHi * Mi;
    Hj = trgpop->N;
    Mj = 2;
    Nj = Hj * Mj;

    if ((recurrent and axoHi - srcM < nactHi + nsilHi) or
        (axoHi < nactHi + nsilHi))
        error("Prj2::Prj2", "Illegal axoHi < nactHi + nsilHi");

    axoNi = axoHi * Mi;
    // printf("Prj2:: axoHi = %d axoNi = %d Mi = %d\n", axoHi, axoNi, Mi);

    if (axoHi < nactHi + nsilHi)
        error("Prj::Prj", "Illegal axoHi < nactHi + nsilHi");
    this->nactHi = nactHi;
    this->nsilHi = nsilHi;

    initialize();
    // trgact2 added 20260107 by ALa
    CUDA_CHECK_ERROR(cudaMallocManaged(&trgact2, Nj * sizeof(float)));

    if (libverbosity > 1)
        printf("name = %s, axoHi = %d axoNi = %d nactHi = %d nsilHi = %d srcM = %d denNi = %d denHi = %d\n",
               name.c_str(), axoHi, axoNi, this->nactHi, this->nsilHi, srcM, denNi, denHi);
}


#ifndef CUDADENACT

void Prj2::upddenact() {
    if (axo == nullptr) {
        CUDA_CHECK_ERROR(cudaMemcpy(axoact, srcpop->act, srcN * sizeof(float), cudaMemcpyDeviceToDevice));
    } else
        axo->updstate();
    for (int hj = 0; hj < Hj; hj++) {
        if (Hihjhi == nullptr) {
            for (int hi = 0; hi < axoHi; hi++) {
                denact[hj * denNi + hi * Mi + 1] = axoact[hi];
                denact[hj * denNi + hi * Mi] = 1 - axoact[hi];
            }
        } else {
            for (int dhi = 0; dhi < denHi; dhi++) {
                int hi = Hihjhi[hj * denHi + dhi];
                denact[hj * denNi + dhi * Mi + 1] = axoact[hi];
                denact[hj * denNi + dhi * Mi] = 1 - axoact[hi];
            }
        }
    }
}

#else

void Prj2::upddenact() {
    printf("Prj2::upddenact\n");
    if (axo == nullptr) {
        CUDA_CHECK_ERROR(cudaMemcpy(axoact, srcpopact, srcN * sizeof(float), cudaMemcpyDeviceToDevice));
    } else
        axo->updstate();
    if (Hihjhi == nullptr) {
            for (int hi = 0; hi < axoHi; hi++) {
                denact[hj * denNi + hi * Mi + 1] = axoact[hi];
                denact[hj * denNi + hi * Mi] = 1 - axoact[hi];
            }
    } else {
        error("Prj2::upddenact", "Under construction");
        upddenact_cu(axoact, Hihjhi, Chjhi, Hj, denHi, Mi, denact);
    }
}

#endif // CUDADENACT

// trgact2 and handling added 20260107 by ALa
void Prj2::updtrgact2() {
    for (int hj = 0; hj < Hj; hj++) {
        trgact2[2*hj+1] = trgpopact[hj];
        trgact2[2*hj] = 1 - trgpopact[hj];
    }
}


void Prj2::updtraces(float prn) {
    updtrgact2();
    Prj::updtraces(denact, trgact2, prn);
}


bool Prj2::onhdiag(int hj, int hihjhi) {
    return hj == hihjhi;
}


int Prj2::nhdiagoff() {
    return srcM * (selfc == HDOFF);
}

