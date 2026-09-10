/*****************************************************************

  Author: Anders Lansner

  Created: 2023-09-15     Modified: 2023-09-15

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

#include "Analys.h"
#include "Globals.h"

using namespace std;
using namespace Globals;

Analys::Analys(int H, int M) {
    this->H = H;
    this->M = M;
    N = H * M;
}

bool Analys::iscorr(float *rec, float *fac) {
    for (int h = 0; h < H; h++) {
        int rimax = 0, fimax = 0;
        float rmax = rec[h*M], fmax = fac[h*M];
        for (int m = 0; m < M; m++) {
            if (rec[h*M + m] > rmax) { rmax = rec[h*M + m]; rimax = m; }
            if (fac[h*M + m] > fmax) { fmax = fac[h*M + m]; fimax = m; }
        }
        if (rimax != fimax) return false;
    }
    return true;
}

bool Analys::iscorr(vector<float> rec, vector<float> fac) {
    if (rec.size() != fac.size())
        error("Analys::iscorr", "Vector size mismatch");
    if (H != 1)
        error("Analys::iscorr", "Illegal: H != 1");
    int rimax = 0, fimax = 0;
    float rmax = rec[0], fmax = fac[0];
    for (int n = 0; n < N; n++) {
        if (rec[n] > rmax) { rmax = rec[n]; rimax = n; }
        if (fac[n] > fmax) { fmax = fac[n]; fimax = n; }
    }
    return rimax == fimax;
}

void Analys::update_confusion(float *rec, float *fac, float *confusion) {
    for (int h = 0; h < H; h++) {
        int rimax = 0, fimax = 0;
        float rmax = rec[h*M], fmax = fac[h*M];
        for (int m = 0; m < M; m++) {
            if (rec[h*M + m] > rmax) { rmax = rec[h*M + m]; rimax = m; }
            if (fac[h*M + m] > fmax) { fmax = fac[h*M + m]; fimax = m; }
        }
        confusion[rimax * M + fimax] += 1;
    }
}

float Analys::corrfrac(vector<vector<float> > rec, vector<vector<float> > fac, int offs, int nelem) {
    // WARNING UNSAFE!
    int ncorr = 0;
    struct timeval total_time;
    gettimeofday(&total_time, 0);
    vector<float> meanrec(N, 0);
    int npat = rec.size() / nelem;
    if ((int)fac.size() != npat)
        error("Analys::corrfac", "fac.size -- npat mismatch ");
    for (int p = 0; p < npat; p++) {
        if (verbosity > 1 and p % 1000 == 0)
            fprintf(stderr, "%d ", p);
        int r0 = p * nelem + offs;
        meanrec = vmean(rec, r0, nelem - offs);
        ncorr += argmax(meanrec, 0, N) == argmax(fac[p], 0, N);
        // for (int n=0; n<N; n++) printf("%3.1f ",meanrec[n]); printf("\n");
        // for (int n=0; n<N; n++) printf("%3.1f ",fac[p][n]);
        // printf(" %d %d ncorr = %d\n",argmax(meanrec,0,N),argmax(fac[p],0,N),ncorr);
    }
    /// printf("Analys: Total time elapsed = %.3f sec\n",getDiffTime(total_time)/1000);
    return ncorr / (float)npat;
}

float Analys::mean(vector<float> data) {
    float vsum = 0;
    for (size_t i = 0; i < data.size(); i++)
        vsum += data[i];
    return vsum / data.size();
}

float Analys::std(vector<float> data) {
    float mea = mean(data), vsum2 = 0;
    for (size_t i = 0; i < data.size(); i++)
        vsum2 += (data[i] - mea) * (data[i] - mea);
    return sqrt(vsum2 / (data.size() - 1));
}

float Analys::sem(vector<float> data) {
    return std(data) / data.size();
}

vector<float> Analys::vmean(vector<vector<float> > data, int offs, int nelem) {
    if (nelem == 0)
        nelem = data.size();
    // printf("rstart = %d rend = %d\n",offs,offs + nelem);
    vector<float>vsum(N, 0);
    for (size_t p = offs; (int)p < offs + nelem; p++) {
        // for (int n=0; n<N; n++) printf("%3.1f ",data[p][n]); printf("\n");
        for (int n = 0; n < N; n++)
            vsum[n] += data[p][n];
    }
    for (size_t i = 0; (int)i < N; i++)
        vsum[i] /= nelem;
    return vsum;
}
