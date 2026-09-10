/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2026-01-14

  Copyright (c) 2026 Anders Lansner

*****************************************************************/

#ifndef __Prjbase_included
#define __Prjbase_included

#define ABSENT 0
#define SILENT 1
#define ACTIVE 2

#include "Globals.h"

class Pop;
class Axo;

class Prjbase {

//********** Static variables and methods **********

    public:
        static std::vector<Prjbase *> baseprjs;

    public:
        static bool prjnamexists(std::string name);
        static void clear();
        static ulong nallocbyteall();
        static void resetall();
        static void upddenactall();
        static void updtracesall(std::vector<float> prns = std::vector<float>(0));
        static void updbwall(bool force = false);
        static void updbwsupall();
        static void contributeall();
        static void updconnsall();

//*********** Class variables and methods **********

public:
    std::string name;
    int id, lrule;
    bool recurrent, needsupdbw, frozen, bdebias;
    Pop *srcpop, *trgpop;
    Axo *axo;
    int srcN, srcM, Mi, axoHi, axoNi, denHi, denNi, Hj, Mj, Nj, selfc;
    int nactQi, nsilQi;
    float *Zi, *Zj, *Zji, *Ei, *Ej, *Eji, *Pi, *Pj, *Bj, *Pji, *Wji;
    float *srcpopact, *trgpopact, *trgpopbwsup, *axoact, *denact;
    float *trgact2; // For Prj2
    float *bwsupinf, *bwsup;
    float eps, taupdt, tauedt, tauzidt, tauzjdt, wgain, bgain, ewgain, iwgain, fgain,
        nswap, nrepl, swaprthr, knrepl, replkthr, Vrev;
    int gnswapped, gnrepled;

    Prjbase(Pop *srcpop, Pop *trgpop, int nactQi, int nsilQi, std::string name);
    virtual ~Prjbase() = default;

    virtual void initialize() = 0;
    virtual void reinitialize() = 0;
    virtual void reset() = 0;
    virtual void allocmem() = 0;
    virtual ulong nallocbyte() = 0;
    virtual bool ISABSENT(int nj, int ni) = 0;
    virtual void initconns(int nactQi, int nsilQi) = 0;
    virtual void upddenact() = 0;
    virtual void putsrcact(float *srcact);
    virtual void updzitrc() = 0;
    virtual void updtraces(float prn = 0.) = 0;
    virtual void updbwsup() = 0;
    virtual void updbw(bool force = false) = 0;
    virtual void contribute() = 0;
    virtual void updMIsc() = 0;
    virtual void nrmMIsc() = 0;
    virtual float miscsum(std::string filename = "") = 0;
    virtual void miscsumx(float& actmiscsum, float& silmiscsum, std::string filename = "") = 0;
    virtual void swapconns() = 0;
    virtual void replconns() = 0;
    virtual int *expandfieldi(int field) = 0;
    virtual float *expandfieldf(int field) = 0;
    virtual float *expandfieldf1(int field) = 0;
    virtual int getnelem(int field) = 0;
    virtual int *getfieldi(int field) = 0;
    virtual float *getfieldf(int field) = 0;
    virtual void prnfield(int field, std::string filename = "") = 0;
    virtual void prnfield(std::string field, std::string filename = "") = 0;

    void setlrule(std::string lrule);
    void seteps(float eps);
    void setbdebias(bool bdebias);
    void settauzi(float tauzi);
    void settauzj(float tauzj);
    void settaue(float taue);
    void settaup(float taup);
    void setVrev(float Vrev); // Not yet used
    void setselfc(std::string selfc);
    void setrandPji();
    void pscrambleps(float kN = 1);
    void setBj(float *Bj);
    void setWji(float *Wji);
    void setbgain(float bgain);
    void setwgain(float wgain);
    void setwgainx(float ewgain, float iwgain);
    void setbwgain(float bwgain);
    void setnswap(float nswap);
    void setnrepl(float nrepl);
    void setknrepl(float knrepl);
    void setreplkthr(float replkthr);
    void setswaprthr(float swaprthr);
    void setdelays(float delay, float spread);
    void setdelays(std::vector<std::vector<float> > delaymat);

};

#endif // __Prjbase_included
