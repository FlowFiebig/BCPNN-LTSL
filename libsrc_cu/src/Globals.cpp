/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-01-03     Modified: 2024-01-03

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

#include "Globals.h"
#include "Pop.h"
#include "Prjbase.h"
#include "Logger.h"
#include "Probe.h"
#include "Netw.h"
#include "PatternFactory.h"

using namespace std;
using namespace Globals;

int Globals::simstep = 0;
float Globals::timestep = 0.001, Globals::simtime = 0, Globals::EPS = 1e-7;
int Globals::libverbosity = 0, Globals::verbosity = 0;
float Globals::gNaN = std::numeric_limits<float>::quiet_NaN();
int Globals::MAXINT = std::numeric_limits<int>::max();
float Globals::MAXFLT = std::numeric_limits<float>::max();
float Globals::LOWESTFLT = std::numeric_limits<float>::lowest();

void Globals::error(string errloc, string errstr, int errcode) {
    fprintf(stderr, "ERROR in %s: %s\n", errloc.c_str(), errstr.c_str());
    exit(4711);
}

void Globals::warning(string warnloc, string warnstr) {
    fprintf(stderr, "WARNING in %s: %s\n", warnloc.c_str(), warnstr.c_str());
}

float Globals::getDiffTime(struct timeval start_time) {
    struct timeval t_time;
    gettimeofday(&t_time, 0);
    // time difference in milli-seconds
    return (1000.0 * (t_time.tv_sec - start_time.tv_sec)
            + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}

LIF_T Globals::stringtoliftype(string liftype) {
    for (size_t i = 0; i < LIF_T_STRING.size(); i++)
        if (LIF_T_STRING[i] == liftype)
            return Globals::LIF_T(i);
    error("Globals::stringtoliftype", "No such liftype: '" + liftype + "'");
    return NOLIF;
}

string Globals::fieldtostring(int field) {
    return FIELD_STRING[field];
}

const char *Globals::fieldtocstr(int field) {
    return fieldtostring(field).c_str();
}

int Globals::stringtofield(string field) {
    for (size_t i = 0; i < FIELD_STRING.size(); i++)
        if (FIELD_STRING[i] == field)
            return i;
    error("Globals::stringtofield", "No such field: '" + field + "'");
    return -1;
}

void Globals::clear() {
    Pop::clear();
    Netw::clear();
    Prjbase::clear();
    PatternFactory::clear();
    Logging::clearlogs();
    Probing::clearprobes();
    simstep = 0;
    simtime = 0;
}

int *Globals::delete1i(int *intvec) {
    if (intvec != nullptr)
        delete [] intvec;
    return nullptr;
}

float *Globals::delete1f(float *fltvec) {
    if (fltvec != nullptr)
        delete [] fltvec;
    return nullptr;
}

int **Globals::delete2i(int **intmat, int nrow) {
    if (intmat != nullptr)
        for (int row = 0; row < nrow; row++)
            delete [] intmat[row];
    return nullptr;
}

float **Globals::delete2f(float **fltmat, int nrow) {
    if (fltmat != nullptr)
        for (int row = 0; row < nrow; row++)
            delete [] fltmat[row];
    return nullptr;
}

int *Globals::alloc1i(int n, int intval) {
    int *intvec = new int[n];
    for (int i = 0; i < n; i++)
        intvec[i] = intval;
    return intvec;
}

float *Globals::alloc1f(int n, float fltval) {
    float *fltvec = new float[n];
    for (int i = 0; i < n; i++)
        fltvec[i] = fltval;
    return fltvec;
}

int **Globals::alloc2i(int nrow, int ncol, int intval) {
    int **intmat = new int*[nrow];
    for (int row = 0 ; row < nrow; row++) {
        intmat[row] = new int[ncol];
        for (int col = 0; col < ncol; col++)
            intmat[row][col] = intval;
    }
    return intmat;
}

float **Globals::alloc2f(int nrow, int ncol, float fltval) {
    float **fltmat = new float*[nrow];
    for (int row = 0 ; row < nrow; row++) {
        fltmat[row] = new float[ncol];
        for (int col = 0; col < ncol; col++)
            fltmat[row][col] = fltval;
    }
    return fltmat;
}

void Globals::fill1i(int *intvec, int n, int intval) {
    for (int i = 0; i < n; i++)
        intvec[i] = intval;
}

void Globals::fill1f(float *fltvec, int n, float fltval) {
    for (int i = 0; i < n; i++)
        fltvec[i] = fltval;
}

void Globals::fill2i(int **intmat, int nrow, int ncol, int intval) {
    for (int row = 0; row < nrow; row++)
        for (int col = 0; col < ncol; col++)
            intmat[row][col] = intval;
}

void Globals::fill2f(float **fltmat, int nrow, int ncol, float fltval) {
    for (int row = 0; row < nrow; row++)
        for (int col = 0; col < ncol; col++)
            fltmat[row][col] = fltval;
}

int *Globals::flatteni(int **intmat, int nrow, int ncol) {
    int *flatintmat = new int[nrow * ncol];
    for (int row = 0, k = 0; row < nrow; row++)
        for (int col = 0; col < ncol; col++, k++)
            flatintmat[k] = intmat[row][col];
    return flatintmat;
}

float *Globals::flattenf(float **fltmat, int nrow, int ncol) {
    float *flatfltmat = new float[nrow * ncol];
    for (int row = 0, k = 0; row < nrow; row++)
        for (int col = 0; col < ncol; col++, k++)
            flatfltmat[k] = fltmat[row][col];
    return flatfltmat;
}

void Globals::tofile(int *vec, int n, FILE *outfp) {
    fwrite (vec, sizeof(int), n, outfp);
}

void Globals::tofile(int *vec, int n, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(vec, n, outfp);
    fclose(outfp);
}

void Globals::tofile(float *vec, int n, FILE *outfp) {
    fwrite (vec, sizeof(float), n, outfp);
}

void Globals::tofile(float *vec, int n, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(vec, n, outfp);
    fclose(outfp);
}

void Globals::tofile(int **mat, int nrow, int ncol, FILE *outfp) {
    printf("enter Globals::tofile: simstep = %d %d %d\n", simstep, nrow, ncol);
    for (int row = 0, n; row < nrow; row++) {
        n = 0;
        for (int col = 0; col < ncol; col++)
            if (mat[row][col] == 2)
                n++;
        printf("%2d %2d\n", row, n);
    }
    for (int r = 0; r < nrow; r++)
        fwrite (mat[r], sizeof(int), ncol, outfp);
    printf("exit Globals::tofile\n");
}

void Globals::tofile(int **mat, int nrow, int ncol, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(mat, nrow, ncol, outfp);
    fclose(outfp);
}

void Globals::tofile(float **mat, int nrow, int ncol, FILE *outfp) {
    for (int r = 0; r < nrow; r++)
        fwrite (mat[r], sizeof(float), ncol, outfp);
}

void Globals::tofile(float **mat, int nrow, int ncol, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(mat, nrow, ncol, outfp);
    fclose(outfp);
}

void Globals::tofile(vector<int> vec, FILE *outfp) {
    fwrite (vec.data(), sizeof(int), vec.size(), outfp);
}

void Globals::tofile(vector<int> vec, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(vec, outfp);
    fclose(outfp);
}

void Globals::tofile(vector<float> vec, FILE *outfp) {
    fwrite (vec.data(), sizeof(float), vec.size(), outfp);
}

void Globals::tofile(vector<float> vec, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(vec, outfp);
    fclose(outfp);
}

void Globals::tofile(vector<vector<int> > mat, FILE *outfp) {
    for (int r = 0; r < mat.size(); r++)
        fwrite (mat[r].data(), sizeof(float), mat[r].size(), outfp);
}

void Globals::tofile(vector<vector<int> > mat, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(mat, outfp);
    fclose(outfp);
}

void Globals::tofile(vector<vector<float> > mat, FILE *outfp) {
    for (float r = 0; r < mat.size(); r++)
        fwrite (mat[r].data(), sizeof(float), mat[r].size(), outfp);
}

void Globals::tofile(vector<vector<float> > mat, string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    tofile(mat, outfp);
    fclose(outfp);
}

void Globals::reset() {
    simstep = 0;
    simtime = 0;
}

void Globals::gsetseed(long seed) {
    RndGen::grndgen->setseed(seed);
}

long Globals::ggetseed() {
    return RndGen::grndgen->getseed();
}

void Globals::gsetnormalparams(double mean, double std) {
    RndGen::grndgen->setnormalparams(mean, std);
}

void Globals::gsetpoissonmean(double mean) {
    RndGen::grndgen->setpoissonmean(mean);
}

int Globals::gnextint() {
    return RndGen::grndgen->nextint();
}

float Globals::gnextfloat() {
    return RndGen::grndgen->nextfloat();
}

float Globals::gnextnormal() {
    return RndGen::grndgen->nextnormal();
}

int Globals::gnextpoisson() {
    return RndGen::grndgen->nextpoisson();
}

vector<int> Globals::gshuffle(vector<int> pidx) {
    return RndGen::grndgen->doshuffle(pidx);
}

float *Globals::binarizepat(float *pat, int H, int M) {
    int N = H * M;
    float *bxpat = new float[2 * N];
    for (int n = 0; n < N; n++) {
        bxpat[2 * n] = 1 - pat[n];
        bxpat[2 * n + 1] = pat[n];
    }
    return bxpat;
}

vector<float> Globals::binarizepat(vector<float> pat, int H, int M) {
    int N = H * M;
    vector<float> bxpat = vector<float>(2 * N);
    for (int n = 0; n < N; n++) {
        bxpat[2 * n] = 1 - pat[n];
        bxpat[2 * n + 1] = pat[n];
    }
    return bxpat;
}

vector<float> Globals::unbinarizepat(vector<float> bxpat, int newH) {
    int N = bxpat.size();
    if (N % 2 != 0)
        error("PatternFactory::unbinarizepat", "Illegal: patlen not even");
    if ((N / 2) % newH != 0)
        error("PatternFactory::unbinarizepat", "Illegal: (N/2)%newH!=0");
    int newM = N / 2 / newH, newN = newH * newM;
    vector<float> pat = vector<float>(newN);
    for (int n = 0; n < newN; n++)
        pat[n] = bxpat[2 * n + 1];
    return pat;
}

void Globals::advance() {
    Logging::dologging();
    Probing::doprobing();
    simstep += 1;
    simtime = simstep * timestep;
}

int Globals::argmin(float *vec, int i1, int n) {
    int mini = i1;
    float minv = vec[mini];
    for (int i = i1 + 1; i < i1 + n; i++)
        if (vec[i] < minv) {
            mini = i;
            minv = vec[i];
        }
    return mini;
}

int Globals::argmin(vector<float> vec, int i1, int n) {
    int mini = i1;
    float minv = vec[mini];
    for (int i = i1 + 1; i < i1 + n; i++)
        if (vec[i] < minv) {
            mini = i;
            minv = vec[i];
        }
    return mini;
}

float Globals::vlen(vector<float> vec) {
    float vlen = 0;
    for (size_t i = 0; i < vec.size(); i++)
        vlen += vec[i] * vec[i];
    return sqrt(vlen);
}

float Globals::vdiff(vector<float> vec1, vector<float> vec2) {
    if (vec1.size() != vec2.size())
        error("Globals::vdiff", "vec1 -- vec2 length mismatch");
    float vdiff = 0;
    for (size_t i = 0; i < vec1.size(); i++)
        vdiff += (vec2[i] - vec1[i]) * (vec2[i] - vec1[i]);
    return sqrt(vdiff);
}

float Globals::vl1(vector<float> vec1, vector<float> vec2) {
    if (vec1.size() != vec2.size())
        error("Globals::vl1", "vec1 -- vec2 length mismatch");
    float vl1 = 0;
    for (size_t i = 0; i < vec1.size(); i++)
        vl1 += abs(vec2[i] - vec1[i]);
    return vl1;
}

float Globals::vmean(vector<float> vec) {
    float vsum = 0;
    for (size_t i = 0; i < vec.size(); i++)
        vsum += vec[i];
    return vsum / vec.size();
}

float Globals::vstd(vector<float> vec, float vmn) {
    if (vec.size() < 2)
        error("Globals::vstd", "Illegl: veclen<2");
    float vsqsum = 0;
    for (size_t i = 0; i < vec.size(); i++)
        vsqsum += (vec[i] - vmn) * (vec[i] - vmn);
    return sqrt(vsqsum / (vec.size() - 1));
}

vector<vector<float> > Globals::readpats(int plen, string filename, int npat) {
    FILE *infp = fopen(filename.c_str(), "rb");
    vector<vector<float> > pats;
    if (infp == nullptr)
        error("Globals::readpats", "Could not open file: [" + filename + "]");
    else {
        fseek(infp, 0, SEEK_END); // non-portable
        long nbyte = ftell(infp);
        fseek(infp, 0, SEEK_SET);
        if (nbyte % sizeof(float) != 0)
            error("Globals::readpats", "Byte format error");
        int nflt = nbyte / sizeof(float);
        // printf("nflt = %d npat = %d plen = %d\n", nflt, npat, plen);
        if (npat > 0) {
            if (nflt < npat * plen)
                error("Globals::readpats", "'npat' too big : " + filename);
            else
                nflt = npat * plen;
        } else { // npat == 0 (default)
            // Read all pats
            if (nflt % plen != 0)
                error("Globals::readpats", "Pat float format error: " + to_string(nflt) + " -- " + to_string(plen));
            else
                npat = nflt / plen;
        }
        float *patdata = (float *)calloc(nflt, sizeof(float));
        int nread = fread(patdata, sizeof(float), nflt, infp);

        if (nread != nflt)
            error("Globals::readpats", "Read float error: " + to_string(nread) + " -- " + to_string(nflt));
        fclose(infp);
        if (nflt % plen != 0)
            error("Globals::readpats", "Pat float format error: " + to_string(nflt) + " -- " + to_string(plen));
        int npat = nflt / plen;
        pats = vector<vector<float> >(npat, vector<float>(plen, 0));
        for (int p = 0, i = 0; p < npat; p++)
            for (int c = 0; c < plen; c++)
                pats[p][c] = patdata[i++];
    }
    return pats;
}

int Globals::grandint(int i) {
    return gnextint() % i;
}

RndGen *RndGen::grndgen = new RndGen();

RndGen::RndGen(long seedoffs) {
    uniformfloatdistr = uniform_real_distribution<float> (0.0, 1.0);
    uniformintdistr = uniform_int_distribution<int>();
    normaldistr = normal_distribution<double>(0.01, 0.001);
    poissondistr = poisson_distribution<int>(1);
    setseed(4711174, seedoffs);
}

void RndGen::setseed(long seed, int hcuid) { // hcuid <--> seedoffs
    // seed==0 gives random seed
    if (seed == 0)
        this->seed = random_device{}();
    else
        this->seed = seed;
    generator.seed(this->seed);
}

void RndGen::setnormalparams(double mean, double std) {
    normaldistr = normal_distribution<double>(mean, std);
}

void RndGen::setpoissonmean(float mean) {
    poissondistr = poisson_distribution<int>(mean);
}

long RndGen::getseed() {
    return seed;
}

int RndGen::nextint() {
    return uniformintdistr(generator);
}

float RndGen::nextfloat() {
    return uniformfloatdistr(generator);
}

float RndGen::nextnormal() {
    return normaldistr(generator);
}

int RndGen::nextpoisson() {
    return poissondistr(generator);
}

vector<int> RndGen::doshuffle(vector<int> pidx) {
    shuffle(begin(pidx), end(pidx), RndGen::generator);
    return pidx;
}
