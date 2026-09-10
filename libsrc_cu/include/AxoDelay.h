/*****************************************************************

  Created: 2024-05-01     Modified: 2024-05-01

  Author: Anders Lansner

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

#ifndef __AxoDelay_included
#define __AxoDelay_included

#include "Globals.h"

class Pop;
class Prjbase;
class AxoDeltap;

class Axo {

    public:
        Pop *pop;
        Prjbase *prj;
        AxoDeltap *axodeltap;
        int maxidelay;

        Axo(Pop *pop, Prjbase *prj);
        ~Axo();
        void setdelays(float delay, float spread);
        void setdelays(std::vector<std::vector<float> > delaymat);
        void updstate();

};

class AxoDelbuf {

    public:

        friend class Pop;
        friend class AxoDeltap;

        int N, maxidelay, now;

        float *popact, *axodelbuf;

    public:

        AxoDelbuf(Pop *pop, int maxidelay);

        ~AxoDelbuf();

        int nallocbyte();
        void reset();
        void updstate();
        int getdslot(int idelay);
        float getdelact(int m, int idelay);
        float *getdelacts(int idelay);
        void prn(FILE *outfp = stderr);

};

class AxoDeltap {

    public:

        friend class Pop;

        Pop *pop;
        Prjbase *prj;
        int Ni;
        float *axoact;
        int *axodeltap;

        AxoDeltap(Pop *pop, Prjbase *prj);
        ~AxoDeltap();

        int nallocbyte();
        void setidelays(float delay, float spread, int *maxidelay);
        void setidelays(std::vector<std::vector<float> > delaymat, int Hi, int Hj, int Nj,
                        int *maxidelay);
        void updstate();

};

#endif // __AxoDelay_included
