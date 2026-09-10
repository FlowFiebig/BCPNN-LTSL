/*****************************************************************

  Author: Anders Lansner

  Created: 2026-02-19

  Copyright (c) 2025 Anders Lansner

******************************************************************/

#ifndef __TDPop_included
#define __TDPop_included

#include "Pop.h"

class TDPop : public Pop {

 public:

    // int K, My; // H == K, M = My
    float *s, *a, *b;
    float fs, tau0, taumax, c;
    int K_mode; // 0 = logarithmic, 1 = linear
    float* d_input;

 public:

    TDPop(int K, int My, std::string name, float fs, float taumin, float taumax, int K_mode);
    ~TDPop() override;
    ulong nallocbyte() override;
    void reset(bool dum) override;
    using Pop::setinput;
    void setinput(float *input) override;
    
};

#endif // __TDPop_included
