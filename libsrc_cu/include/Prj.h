/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-07-10     Modified: 2024-07-10

*****************************************************************/

#ifndef __Prj_included
#define __Prj_included

#include "Globals.h"
#include "Prjbase.h"
#include "Pop.h"

class Axo;

class Prj : public Prjbase {

//************** Class variables and methods **************

    public:
        int nactHi, nsilHi;
        int *Chjhi, *Hihjhi;
        float *MIhjhi, *nMIhjhi;
        uint *rnduints;
        float P0; // nbrav
        int *Hifanout;
        int *hilist, *hilistoffs;
        /********** Variables for monitoring ***********/
        float gminactsc, gmaxactsc, gminsilsc, gmaxsilsc;
        int *nswapped, *nrepled;
        bool *spmask;
        /********** Variables for printout and logging ***********/
        int *xfieldi_HjHi;
        float *xfieldf_HjHi, *xfieldf_NjNi, *xfieldf_HjNi;

    public:
        Prj(Pop *srcpop, Pop *trgpop, std::string name = "", bool doinit = true);
        Prj(Pop *srcpop, Pop *trgpop, int nactHi, int nsilHi, std::string name = "", bool doinit = true);
        ~Prj();

    public:

        void initialize() override;
        void allocmem() override;
        void reinitialize() override;
        void reset() override;
        ulong nallocbyte() override;
        bool ISABSENT(int nj, int ni) override;
        void setWjix(float *Wjix);
        void fixupselfc(std::vector<int> shuffled);
        void initconns(int nactHi, int nsilHi) override;

        void calcrfpos(int hj, float *rfpos);
        void prnrfpos(std::string filename);
        virtual void upddenact();
        void updzitrc();
    protected:
        void updtraces(float *denact, float *trgact, float prn = 1);
    public:
        void updtraces(float prn = 0);
        void updbw(bool force = false);
        void updbwsup();
        void contribute();
        float miscsum(std::string filename = "");
        void miscsumx(float& actmiscsum, float& silmiscsum, std::string filename = "");
        void updMIsc();
        void nrmMIsc();
        void updhifanout();
        void swapconns();
        void renorm_hi(int hi);
        int swapconns_cu();
        void replconns();
        void resetcmonitor();
        virtual bool onhdiag(int hj, int hihjhi);
        virtual int nhdiagoff();

        int *expandfieldi(int field);
        float *expandfieldf(int field);
        float *expandfieldf1(int field);
        int getnelem(int field);
        int *getfieldi(int field);
        float *getfieldf(int field);
        void prnfield(int field, std::string filename = "");
        void prnfield(std::string field, std::string filename = "");

};

#endif // __Prj_included
