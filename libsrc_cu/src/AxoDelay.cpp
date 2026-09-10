/*****************************************************************

  Created: 2024-05-01  Modified: 2024-05-01

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

#include "Pop.h"
#include "Prj.h"
#include "AxoDelay.h"

using namespace std;
using namespace Globals;

Axo::Axo(Pop *pop, Prjbase *prj) {
    this->pop = pop;
    this->prj = prj;
    axodeltap = nullptr;
    maxidelay = 0;
    pop->axos.push_back(this);
}


Axo::~Axo() {
    delete axodeltap;
}

void Axo::setdelays(float delay, float spread) {
    if (simstep > 0)
        error("Axo::addaxodeltap", "Illegal: Creating delays after starting simulation");
    axodeltap = new AxoDeltap(pop, prj);
    axodeltap->setidelays(delay, spread, &maxidelay);
}

void Axo::setdelays(vector<vector<float> > delaymat) {
    if (simstep > 0)
        error("Axo::addaxodeltap", "Illegal: Creating delays after starting simulation");
    axodeltap = new AxoDeltap(pop, prj);
    axodeltap->setidelays(delaymat, pop->H, prj->Hj, prj->Nj, &maxidelay);
}


void Axo::updstate() {
    if (axodeltap != nullptr) {
        if (not pop->axodelbuf) {
            pop->axodelbuf = new AxoDelbuf(pop, maxidelay);
        }
        axodeltap->updstate();
        prj->putsrcact(axodeltap->axoact);
    } else
        prj->putsrcact(pop->act);
}


AxoDelbuf::AxoDelbuf(Pop *pop, int maxidelay) {
    N = pop->N;
    this->maxidelay = maxidelay;
    axodelbuf = new float[N * maxidelay];
    memset(axodelbuf, 0, N * maxidelay * sizeof(float));
    now = 0;
    popact = pop->act;
}

AxoDelbuf::~AxoDelbuf() {
    delete [] axodelbuf;
    axodelbuf = nullptr;
}

int AxoDelbuf::nallocbyte() {
    int nbyte = N * maxidelay;
    return 4 * nbyte;
}

void AxoDelbuf::reset() {
    memset(axodelbuf, 0, N * maxidelay * sizeof(float));
}

void AxoDelbuf::updstate() {
    for (int n = 0; n < N; n++)
        axodelbuf[n * maxidelay + now] = popact[n];
    now = (now + 1) % maxidelay;
}

int AxoDelbuf::getdslot(int idelay) {
    return (maxidelay + now - idelay) % maxidelay;
}

float AxoDelbuf::getdelact(int m, int idelay) {
    if (N * maxidelay <= m * maxidelay + getdslot(idelay)) {
        error("", "axodelbuf[] index too large: " + to_string(N * maxidelay) + " -- " +
              to_string(m * maxidelay + getdslot(idelay)));
    }
    return axodelbuf[m * maxidelay + getdslot(idelay)];
}

float *AxoDelbuf::getdelacts(int idelay) {
    if (maxidelay <= idelay)
        error("AxoDelbuf::getdelacts", "'idelay' too large");
    return &axodelbuf[getdslot(idelay * N)];
}

void AxoDelbuf::prn(FILE *outfp) {
    for (int n = 0; n < N; n++) {
        // for (int d = now; d < maxidelay + now; d++)
        for (int d = 0; d < maxidelay; d++)
            fprintf(outfp, "%.1f ", axodelbuf[n * maxidelay + d % maxidelay]);
        fprintf(outfp, "\n");
    }
}

AxoDeltap::AxoDeltap(Pop *pop, Prjbase *prj) {
    this->pop = pop;
    this->prj = prj;
    Ni = pop->N;
    axoact = alloc1f(Ni);
    fill1f(axoact, Ni, 0);
    axodeltap = alloc1i(Ni);
    fill1i(axodeltap, Ni);
}

AxoDeltap::~AxoDeltap() {
    delete [] axodeltap;
    axodeltap = nullptr;
}

int AxoDeltap::nallocbyte() {
    return 4 * 2 * Ni;
}

void AxoDeltap::setidelays(float delay, float spread, int *maxidelay) {
    float meandelay = 0;
    int minidelay = MAXINT;
    delay /= timestep;
    spread /= timestep;
    gsetpoissonmean(spread);
    for (int i = 0; i < Ni; i++) {
        int idelay = delay - spread + gnextpoisson() + 0.5;
        if (verbosity > 2)
            fprintf(stderr, "%.0f %.0f %d\n", delay, spread, idelay);
        if (idelay < 1)
            error("AxoDeltap::setidelays", "idelay<1");
        axodeltap[i]  = idelay;
        if (idelay < minidelay)
            minidelay = idelay;
        if (idelay > *maxidelay)
            *maxidelay = idelay;
        meandelay += idelay;
    }
    meandelay /= Ni;
    if (verbosity > 2)
        printf("\n");
    if (verbosity > 1)
        printf("minidelay = %d maxidelay = %d meandelay = %.2f\n", minidelay, *maxidelay, meandelay);
}

void AxoDeltap::setidelays(std::vector<std::vector<float> > delaymat, int Hi, int Hj, int Nj,
                           int *maxidelay) {
    // 'idelaymat' contains mean delays in simsteps between src- and
    // trg-hypercolumns or src- and trg-units
    int nrow = delaymat.size();
    int ncol = delaymat[0].size();
    axodeltap = alloc1i(Ni);
    fill1i(axodeltap, Ni);
    if (nrow == Hi and ncol == Hj) {
        for (int hj = 0; hj < Hj; hj++) {
            for (int hi = 0; hi < Hi; hi++) {
                for (int i = 0; i < Ni; i++) {
                    int idelay = delaymat[hi][hj] / timestep + 0.5;
                    if (idelay < 1)
                        error("AxoDeltap::setidelays", "idelay<1");
                    axodeltap[i] = idelay;
                    if (idelay > *maxidelay)
                        *maxidelay = idelay;
                }
            }
        }
    } else if (nrow == Ni and ncol == Nj) {
        for (int j = 0; j < Nj; j++) {
            for (int i = 0; i < Ni; i++) {
                int idelay = delaymat[i][j] / timestep + 0.5;
                if (idelay < 1)
                    error("AxoDeltap::setidelays", "idelay<1");
                axodeltap[i] = idelay;
                if (idelay > *maxidelay)
                    *maxidelay = idelay;
            }
        }
    } else
        error("AxoDeltap::setidelays", "idelaymat size mismatch");
}

void AxoDeltap::updstate() {
    memset(axoact, 0, Ni * sizeof(float));
    for (int n = 0; n < Ni; n++)
        axoact[n] = pop->axodelbuf->getdelact(n, axodeltap[n]);
}
