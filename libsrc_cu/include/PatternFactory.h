/*****************************************************************

  Author: Anders Lansner

  Created: 2023-09-13     Modified: 2024-01-07
  Modified from: /home/ala/OurPrograms/BCPNNH/Pats.h

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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/

#ifndef __PatternFactory_included
#define __PatternFactory_included

#include "Globals.h"

class PatternFactory {

    public:
        int id;
        int Hx, Mx, Nx; // patlen = pattern length = Nx, Hx = N:o modules, Mx = module size
        int patype;
        int npat, ninst;
        std::string filename;
        bool binarizing;
        std::vector<std::vector<float> > pats, bxpats, insts, bxinsts;

    public:
        static std::vector<PatternFactory *> patfacs;
        static RndGen *rndgen;
        static void clear();
        static RndGen *mkrndgen(long seed);
        static std::vector<float> complpat(std::vector<float> pat, int Hx, int Mx);
        static std::vector<float> hblank(std::vector<float> pat, int H, int M, float nhblank);
        static std::vector<float> hflip(std::vector<float> pat, int H, int M, float nhflip);
        static std::vector<float> binarize(std::vector<float> pat);
        static std::vector<float> hnormalize(std::vector<float> pat, int Hx);
        static void prpat(std::vector<float> pat, int ndec = 2, int npos = 0, std::string endl = "\n");
        static void prpats(std::vector<std::vector<float> > pats, int ndec = 2, int npos = -1,
                           std::string endl = "\n");
        static void prpat(float *pat, int n, int ndec = 2, int npos = 0, std::string endl = "\n");
        static void prpats(float *pats, int n, int npat, int ndec = 2, int npos = 0, std::string endl = "\n");

        PatternFactory(int patlen, int Hx, std::string patype = "hrand");
        explicit PatternFactory(PatternFactory *patfac, bool complm = false);
        ~PatternFactory();
        void binarizepats();
        int getnpat();
        int getninst();
        void mkpats(int npat);
        void mkinsts(int ninst);
        void readpats(std::string filename, int npat = 0);
        std::vector<std::vector<float> > binarizepats(std::vector<std::vector<float> > pats);
        std::vector<float> getpat(int p);
        std::vector<std::vector<float> > getpats(int p0 = 0, int npat = 0);
        std::vector<float> getinst(int p);
        std::vector<std::vector<float> > getinsts(int p0 = 0, int npat = 0);
        float *getfpat(int p);
        float *getfinst(int p);
        void writepats(std::string filename);
        void writeinsts(std::string filename);
        void hblankpats(float nhblank);
        void hflippats(float nhflip);
        void hblankinsts(float nhblank);
        void hflipinsts(float nhflip);
};

#endif // __PatternFactory_included
