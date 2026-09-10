/*****************************************************************

  Author: Anders Lansner

  Created: 2024-08-05     Modified: 2024-08-10

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

#ifndef __Prj_cu_included
#define __Prj_cu_included

#include "Globals.h"
#include "GPUGlobals.cuh"

void upddenact_cu(float *axoact, int *Hihjhi, int *Chjhi, int Hj, int denHi, int Mi, float *denact);

void updzitrc_cu(float *denact, int Hj, int denNi, float fgain, float eps, float tauzidt, float *Zi);


// void updpitrc_cu(float *Zi, int Hj, int denNi, float prntaupdt, float *Pi);


// void updzpjtrc_cu(float *trgact, int Nj, float fgain, float eps, float tauzjdt, float prntaupdt,
//                   float *Zj, float *Pj);


// void updpjitrc_cu(float *Zi, float *Zj, int Hj, int Nj, int denNi, int Mj, float prntaupdt, float *Pji);


void updtraces_cu(float *denact, float *trgact, float prn, bool frozen,
                  int Hj, int Nj, int Mj, int denNi,
                  float fgain, float eps, float tauzidt, float tauzjdt, float taupdt,
                  float *Zj, float *Zi, float *Pj, float *Pi, float *Pji);

void updbw_cu(int lrule, int Nj, int Mj, int denHi, int denNi, int Mi,
              int *Chjhi,
              float *Pj, float *Pi, float *Pji, float *Bj, float *Wji,
              float eps, float bgain, float wgain, float ewgain, float iwgain,
              bool bdebias = false);

void updbwsup_cu(float *Zi, float *Bj, float *Wji, int Hj, int Mj, int denNi, float tauzidt,
                 float *bwsupinf, float *bwsup);


void contribute_cu(float *bwsup, float *trgpopbwsup, int Nj);


void updMIsc_cu(float *Pj, float *Pi, float *Pji, float eps, int *Hihjhi, int *Hifanout, int Hj, int Nj, int Mj,
                int denHi, int denNi, int Mi, float *MIhjhi);

void nrmMIsc_cu(int *Hihjhi, int *Hifanout, 
                int Hj, int Nj, int denHi, int denNi, 
                float *MIhjhi, float *nMIhjhi);

// void swaponeconn_cu(int *Chjhi, float *MIhjhi, int *Hhjhi, int Hj, int Mj, int denHi, int Mi,
//                     float LOWESTFLT, float MAXFLT, float eps, float swaprthr,
//                     int *Hifanout, int *nswapped, int *gnswapped,
//                     float *Pj, float *Pi, float *Pji, float *Bj, float *Wji);

// void reploneconn_cu(int *Chjhi, float *MIhjhi, int *Hihjhi, uint *rnduints,
//                     float eps,
//                     int Hj, int Mj, int denHi, int axoHi, int Mi,
//                     float MAXFLT, float replthr, int *Hifanout, int *nrepled, int *gnrepled,
//                     float *Pj, float *Pi, float *Pji, float *Bj, float *Wji);

#endif // __Prj_cu_included
