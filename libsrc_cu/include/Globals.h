/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2023-09-08     Modified: 2023-10-27

*****************************************************************/
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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/

#ifndef __Globals_included
#define __Globals_included

#include <string>
#include <random>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <algorithm> // shuffle
#include <iostream>
#include <limits>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <fstream>

/**************************************************************
                   BCPNNSim2.0 enum space
**************************************************************/

namespace Globals {

    enum LIF_T { ALIF = 0, ADEX, ADEXS, NOLIF };    
 
    const std::vector<std::string> LIF_T_STRING = { "ALIF", "ADEX", "ADEXS", "NOLIF" };

    enum FIELD {
        LGI = 0, INP, BWSUP, BWSUPINF, SUP, ACT, ADA, SADA,
        EXP, WTA, SOFTMAX, STCWTA, SPK, FULLNORM, HALFNORM,
        HDON, HDOFF, HDONL,
        BCP, WILL, HEBB, COV,
        CHJHI, HIHJHI, CHJHIX, CJI, IJI, CJIX,
        ZI, ZJ, P, ZIX,
        EI, EJ, EJI, EIX,
        PI, PJ, PJI, PJPI, BJ, PJIX, WJIX, PIX,
        KBJ, WJI, RAWWJI,
        MIHJHI, RAWMISC, CUV, RAWCUV, MIHJHIX, MIJI, MIJIX, NMIJI, NMIJIX,
        XDHITOHI, AXOACT,
        DENACT, RAWDENACT, HIFANOUT, HJFANIN, IFANOUT, DENACTX,
        HRAND, ORTHO,
        ENERGY
    };

    const std::vector<std::string> FIELD_STRING = {
        "LGI", "INP", "BWSUP", "BWSUPINF", "SUP", "ACT", "ADA", "SADA",
        "EXP", "WTA", "SOFTMAX", "STCWTA", "SPK", "FULLNORM", "HALFNORM",
        "HDON", "HDOFF", "HDONL",
        "BCP", "WILL", "HEBB", "COV",
        "CHJHI", "HIHJHI", "CHJHIX", "CJI", "IJI", "CJIX",
        "ZI", "ZJ", "P", "ZIX",
        "EI", "EJ", "EJI", "EIX", 
        "PI", "PJ", "PJI", "PJPI", "BJ", "PJIX", "WJIX", "PIX",
        "KBJ", "WJI", "RAWWJI",
        "MIHJHI", "RAWMISC", "CUV", "RAWCUV", "MIHJHIX", "MIJI", "MIJIX", "NMIJI", "NMIJIX",
        "XDHITOHI", "AXOACT",
        "DENACT", "RAWDENACT", "HIFANOUT", "HJFANIN", "IFANOUT", "DENACTX",
        "HRAND", "ORTHO",
        "ENERGY"
    };

    extern int simstep;
    extern float timestep, simtime;
    extern float EPS;
    extern int libverbosity;
    extern int verbosity;
    extern float gNaN;
    extern int MAXINT;
    extern float MAXFLT, LOWESTFLT;

    void error(std::string errloc, std::string errstr, int errcode = -1);
    void warning(std::string warnloc, std::string warnstr);

    float getDiffTime(struct timeval start_time);
    LIF_T stringtoliftype(std::string liftype);
    std::string fieldtostring(int field);
    const char *fieldtocstr(int field);
    int stringtofield(std::string field);
    void clear();
    int *delete1i(int *intvec);
    float *delete1f(float *fltvec);
    int **delete2i(int **intmat, int nrow);
    float **delete2f(float **fltmat, int nrow);
    int *alloc1i(int n, int intval = 0);
    float *alloc1f(int n, float fltval = 0);
    int **alloc2i(int nrow, int ncol, int intval = 0);
    float **alloc2f(int nrow, int ncol, float fltval = 0);
    void fill1i(int *intvec, int n, int intval = 0);
    void fill1f(float *fltvec, int n, float fltval = 0);
    void fill2i(int **intmat, int nrow, int ncol, int intval = 0);
    void fill2f(float **fltmat, int nrow, int ncol, float fltval = 0);
    float *flattenf(float **fltmat, int nrow, int ncol);
    int *flatteni(int **intmat, int nrow, int ncol);

    void tofile(int *vec, int n, FILE *outfp);
    void tofile(int *vec, int n, std::string filename);
    void tofile(float *vec, int n, FILE *outfp);
    void tofile(float *vec, int n, std::string filename);
    void tofile(int **mat, int nrow, int ncol, FILE *outfp);
    void tofile(int **mat, int nrow, int ncol, std::string filename);
    void tofile(float **mat, int nrow, int ncol, FILE *outfp);
    void tofile(float **mat, int nrow, int ncol, std::string filename);
    void tofile(std::vector<int> vec, FILE *outfp);
    void tofile(std::vector<int> vec, std::string filename);
    void tofile(std::vector<float> vec, FILE *outfp);
    void tofile(std::vector<float> vec, std::string filename);
    void tofile(std::vector<std::vector<int> > mat, FILE *outfp);
    void tofile(std::vector<std::vector<int> > mat, std::string filename);
    void tofile(std::vector<std::vector<float> > mat, FILE *outfp);
    void tofile(std::vector<std::vector<float> > mat, std::string filename);

    void reset();
    void advance();
    void gsetseed(long seed);
    long ggetseed();
    void gsetnormalparams(double mean, double std);
    void gsetpoissonmean(double mean);
    int gnextint();
    float gnextfloat();
    float gnextnormal();
    int gnextpoisson();
    std::vector<int> gshuffle(std::vector<int> pidx);
    float *binarizepat(float *pat, int H, int M);
    std::vector<float> binarizepat(std::vector<float> pat, int H, int M);
    std::vector<float> unbinarizepat(std::vector<float> bxpat, int newH);
    int grandint(int i);
    int argmin(float *vec, int i1, int n);
    int argmin(std::vector<float> vec, int i1, int n);
    float vlen(std::vector<float> vec);
    float vdiff(std::vector<float> vec1, std::vector<float> vec2);
    float vl1(std::vector<float> vec1, std::vector<float> vec2);
    float vmean(std::vector<float> vec);
    float vstd(std::vector<float> vec, float vmean);
    std::vector<std::vector<float> > readpats(int plen, std::string filename, int npat = 0);
    void allogoff();
    void dologall();
    void dosaveall();

    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    template <typename T>
    int argmax(T *vec, int i1, int n) {
        int maxi = i1;
        T maxv = vec[maxi];
        for (int i = i1 + 1; i < i1 + n; i++)
            if (vec[i] > maxv) {
                maxi = i;
                maxv = vec[i];
            }
        return maxi;
    }

    template <typename T>
    int argmax(std::vector<T> vec, int i1, int n) {
        return argmax(vec.data(), i1, n);
    }
};

class RndGen {
    protected:
        long seed;
        std::uniform_real_distribution<float> uniformfloatdistr;
        std::uniform_int_distribution<int> uniformintdistr;
        std::poisson_distribution<int> poissondistr;
        std::normal_distribution<double> normaldistr;

    public:
        std::mt19937_64 generator;
        static RndGen *grndgen;
        RndGen(long seedoffs = -1);
        void setseed(long seed, int hcuid = -1);
        void setnormalparams(double mean, double std);
        void setpoissonmean(float mean);
        long getseed();
        int nextint();
        float nextfloat();
        float nextnormal();
        int nextpoisson();
        std::vector<int> doshuffle(std::vector<int> pidx);
};

#endif // __Globals_included
