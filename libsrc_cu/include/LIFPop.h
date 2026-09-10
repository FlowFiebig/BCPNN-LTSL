/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-08-11     Modified: 2024-08-11

  Copyright (c) 2023 Anders Lansner, Naresh Ravichandran

******************************************************************/

#ifndef __LIFPop_included
#define __LIFPop_included

#include "Globals.h"
#include "GPUGlobals.cuh"
#include "Pop.h"

class LIFPop : public Pop {

 public:

    float C, gL, EL, DT, VR, VT, reft, taum;
    float *inp;
    int ireft, ispkwid;
    int *spkstep;

 public:

    LIFPop(int H, int M, Globals::LIF_T liftype = Globals::ALIF, std::string name = "");
    ~LIFPop();
    ulong nallocbyte();

    void setCgL(float C, float gL, bool verbose = false);
    void setEL(float EL);
    void setDT(float DT); // if (not actfn_t==ALIF) 
    void setVR(float VR);
    void setVT(float VT);
    void settaum(float taum) override;
    void setadgain(float adgain) override;
    void setsadgain(float adgain) override;
    void setreft(float reft);
    void setspkwid(float spkwid);
    void setnfreq(float nfreq) override;
    void reset(bool resetaxodelbuf = true) override;

    void setinput(float input) override;
    void setinput(float *input = nullptr) override;
    void setclamp(float *input, float clampval = 1) override;
    // void setlgi(float *lgi = nullptr); // Missing

    void updada();
    void updsup() override;
    void updact() override;

    int getnelem(int field) override;
    float *getfieldf(int field) override;

};

#endif // __LIFPop_included
