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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/

#ifndef __Analys_included
#define __Analys_included

#include "Globals.h"

class Analys {
    public:
        int H, M, N;

    public:
        Analys(int H, int M);
        bool iscorr(float *rec, float *fac);
        bool iscorr(std::vector<float> rec, std::vector<float> fac);
        void update_confusion(float *rec, float *fac, float *confusion);
        float corrfrac(std::vector<std::vector<float> > rec, std::vector<std::vector<float> > fac,
                       int offs = 0, int nelem = 1);
        float mean(std::vector<float> data);
        float std(std::vector<float> data);
        float sem(std::vector<float> data);
        std::vector<float> vmean(std::vector<std::vector<float> > data, int offs = 0, int nelem = 0);
};

#endif // __Analys_included
