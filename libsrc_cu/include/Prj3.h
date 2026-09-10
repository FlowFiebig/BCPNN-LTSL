/*****************************************************************

  Author: Anders Lansner

  Created: 2026-02-10

  Copyright (c) 2026 Anders Lansner

260210 ALa:
This code has structural plasticity at level of pre-N -- post-N. Seems
to allow efficient sparsity for recurrent networks.

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
        /********** Variables for monitoring ***********/
        float gminactsc, gmaxactsc, gminsilsc, gmaxsilsc;
        int *nswapped, *nrepled;
        int gnswapped, gnrepled;
        /********** Variables for printout and logging ***********/
        int *xfieldi_NjNi;
        float *xfieldf_NjNi;
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
        void setreplkthr(float replkthr);
        void upddenact();
        void putsrcact(float *srcact);
        void updzitrc();
        void updtraces(float *denact, float *trgact, float prn = 1);
        void updtraces(float prn = 0);
        void updbw(bool force = false);
        void updbwsup();
        void contribute();
        void updMIsc();
        void nrmMIsc();
        void resetcmonitor();
        float miscsum(std::string filename = "");
        void miscsumx(float& actmiscsum, float& silmiscsum, std::string filename = "");
        bool ondiag(int nj, int ni);
        void swapconns();
        void replconns();
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
