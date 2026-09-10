/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2023-09-08     Modified: 2024-05-03

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

#ifndef __Pop_included
#define __Pop_included

#include "Globals.h"
#include "GPUGlobals.cuh"

class Pop;
class Axo;
class AxoDelbuf;

class Pop {
//********** Static variables and methods **********
    protected:
        static std::vector<Pop *> pops;

    public:
        static void clear();
        static bool popnamexists(std::string name);
        static int npop();
        static void resetall(bool resetaxodelbuf = true);
        static void resetbwsupall();
        static void updsupall();
        static void updactall();
        static ulong nallocbyteall();

//************** Class variables and methods **************

        int H, M, N;
        int id, actfn, normfn;
        Globals::LIF_T liftype;
        std::string name;
        float again, igain, nampl, nfreq, taumdt, tauadt, adgain, tausadt, sadgain, maxfq, fgain, bwgain;
        float *lgi, *bwsup, *supinf, *sup, *act, *ada, *sada, *cumsum, *popdata;
        uint *pnoise;
        int *nimax, *himax;
        float *hfmax, *hfsum, *spkthres;
        std::vector<Axo *> axos;
        AxoDelbuf *axodelbuf;
        float hx, hy;

        std::poisson_distribution<int> poissondistr;
        std::uniform_real_distribution<float> uniformdistr;
        curandGenerator_t gen_cu;
        curandStatus_t istat;

        Pop(int H, int M, std::string name = "");
        virtual ~Pop();
        virtual ulong nallocbyte();
        void setactfn(std::string actfn);
        void setnormfn(std::string normfn);
        virtual void settaum(float taum);
        void setagain(float again);
        void setigain(float igain);
        void setbwgain(float bwgain);
        void setnampl(float nampl);
        virtual void setnfreq(float nfreq);
        void settaua(float taua);
        virtual void setadgain(float adgain);
        void settausa(float tausa);
        virtual void setsadgain(float sadgain);
        void setmaxfq(float maxfq);
        virtual void reset(bool resetaxodelbuf = true);
        virtual void setinput(float input);
        virtual void setinput(float *input = nullptr);
        void setlgi(float *lgi = nullptr);
        virtual void setclamp(float *input, float clampval = 1e3);
        void unsetclamp();
        void resetbwsup();
        virtual void updsup();
        void normact();
        virtual void updact();
        virtual int getnelem(int field);
        virtual float *getfieldf(int field);
        void prnfieldf(int field, std::string filename = "");
        void prnfield(int field, std::string filename = "");
        void prnfield(std::string field, std::string filename = "");

};

#endif // __Pop_included
