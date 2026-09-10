/*****************************************************************

  Author: Anders Lansner

  Created: 2026-01-14

  Copyright (c) 2026 Anders Lansner

260210 ALa: This code has structural plasticity at level of pre-N --
post-H. Unclear what benefits it has, but seems like the biologically
most plausible configuration.

*****************************************************************/

#ifndef __Prj1_included
#define __Prj1_included

#include "Globals.h"
#include "Prjbase.h"
#include "Prj1.h"
#include "Pop.h"

class Pop;
class Axo;

class Prj1 : public Prjbase {

//************** Class variables and methods **************

    public:
        Axo *axo;
        int nactNi, nsilNi, denHi;
        int *Cji, *Iji;
        uint *rnduints;
        float P0; // nbrav
        float *MIji, *nMIji;
        int *Ifanout;

        int *nilist, *nilistoffs;

        /********** Variables for monitoring ***********/
        float gminactsc, gmaxactsc, gminsilsc, gmaxsilsc;
        int *nswapped, *nrepled;
        bool *spmask;
        /********** Variables for printout and logging ***********/
        int *xfieldi_NjNi, *xfieldi_HjNi;
        float *xfieldf_NjNi, *xfieldf_HjNi;
    public:
        Prj1(Pop *srcpop, Pop *trgpop, int nactHi, int nsilHi, std::string name = "");
        Prj1(Pop *srcpop, Pop *trgpop, std::string name = "");
        ~Prj1();
    protected:
        void initialize();
        void allocmem() override;
        void reinitialize();
        void reset();
        bool ISABSENT(int nj, int ni);
        void fixupselfc(std::vector<int> shuffled);
        void initIji(int nactNi, int nsilNi);
        void initconns(int nactNi, int nsilNi);
        void updIfanout();
        ulong nallocbyte();
    public:
        void upddenact();
        void upddenact_cu(float* denact, float* axoact, const int* Iji,
                          int axoNi, int denNi, int Hj,
                          cudaStream_t stream = 0);
        void putsrcact(float *srcact);
        void updzitrc();
        void updzitrc_cu(float* Zi, const float* denact, int Nj, int denNi, float fgain, float eps, float tauzidt,
                         cudaStream_t stream = 0);
        void updtraces(float *trgact, float prn = 1);
        void updtraces_cu(const float* trgact, float prn, bool frozen, int Hj, int Mj, int Nj, int denNi, float fgain, float eps,
                          float tauzjdt, float taupdt, const float* Zi, float* Zj, float* Pi, float* Pj, float* Pji,
                          cudaStream_t stream = 0);
        void updtraces(float prn = 0);
        void updbw(bool force = false);
        void updbw_cu(float* Bj, float* Wji, const float* Pj, const float* Pi, const float* Pji, const int* Cji,
                      const int* Iji, int Hj, int Mj, int Nj, int denNi, float bgain, float wgain, float ewgain, float iwgain,
                      float nactNi, float axoNi, bool recurrent, int selfc, cudaStream_t stream);
        void updbwsup();
        void updbwsup_cu(float* bwsupinf, float* bwsup, const float* Bj, const float* Zi, const float* Wji,
                         int Hj, int Mj, int Nj, int denNi, float tauzidt, cudaStream_t stream = 0);
        void contribute();
        void contribute_cu(float* trgpopbwsup, const float* bwsup, int Nj, cudaStream_t stream = 0);
        void updMIsc();
        void updMIsc_cu(float* MIji, const float* Pi, const float* Pj, const float* Pji,
                        const int* Iji,   // can be nullptr if not used
                        int Hj, int Mj, int Nj, int denNi, bool recurrent, int selfcisHDOFF,
                        cudaStream_t stream = 0);
        void nrmMIsc();
        void nrmMIsc_cu(float* MIji, const int* Iji,  // can be nullptr if not used
                        int *Ifanout, float* nMIji, int HjdenNi, cudaStream_t stream = 0);
        void resetcmonitor();
        float miscsum(std::string filename = "");
        void miscsumx(float& actmiscsum, float& silmiscsum, std::string filename = "");
        void swapconns();
        void swapconns_cu(bool *spmask, int *nswapped, cudaStream_t stream = 0);
        void replconns();
        void replconns_cu(bool *spmask, int *nrepled, cudaStream_t stream = 0);
        int *expandfieldi(int field);
        float *expandfieldf(int field);
        float *expandfieldf1(int field);
        int getnelem(int field);
        int *getfieldi(int field);
        float *getfieldf(int field);
        void prnfield(int field, std::string filename = "");
        void prnfield(std::string field, std::string filename = "");
};

#endif // __Prj1_included
