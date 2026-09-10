/*****************************************************************

  Created: 2024-01-25  Modified: 2024-01-25

  Authors: Anders Lansner

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

#include "Probe.h"

using namespace std;
using namespace Globals;
using namespace Probing;

vector<Probe *> Probe::probes;
bool Probing::enabled = true;

void Probing::clearprobes() {
    Probe::probes.clear();
}

void Probing::enableprobes(bool value) {
    enabled = value;
}

void Probing::doprobing() {
    for (size_t p = 0; p < Probe::probes.size(); p++)
        Probe::probes[p]->doprobe();
}

void Probing::closeprobes() {
    for (size_t p = 0; p < Probe::probes.size(); p++)
        Probe::probes[p]->doclose();
}

//************** Class variables and methods **************

Probe::Probe(Pop *pop, string fieldstr, string filename, int n) {
    this->pop = pop;
    this->prj = nullptr;
    field = stringtofield(fieldstr);
    outfp = fopen(filename.c_str(), "w");
    nelem = pop->N;
    if (n < 0 or n > nelem)
        error("Probe::Probe(Pop)", "n not in [0,nelem[");
    this->n = n;
    probeint = 1;
    ison = true;
    probes.push_back(this);
}

Probe::Probe(Prj *prj, string fieldstr, string filename, int k) {
    this->pop = nullptr;
    this->prj = prj;
    field = stringtofield(fieldstr);
    outfp = fopen(filename.c_str(), "w");
    switch (field) {
        case AXOACT:
            if (k < 0 or prj->axoNi <= k)
                error("Probe::Probe(Prj)", "i not in [0,axoNi[");
            break;
        case BWSUPINF:
        case BWSUP:
            if (k < 0 or prj->Nj <= k)
                error("Probe::Probe(Prj)", "j not in [0,Nj[");
            break;
        default:
            error("Probe::Probe(Prj)", "No such field: '" + fieldtostring(field) + "'");
    }
    this->k = k;
    probeint = 1;
    ison = true;
    probes.push_back(this);
}

Probe::Probe(Prj *prj, string fieldstr, string filename, int dhi, int hj, int mi, int mj) {
    this->pop = nullptr;
    this->prj = nullptr;
    field = stringtofield(fieldstr);
    outfp = fopen(filename.c_str(), "w");
    if (hj < 0 or prj->Hj <= hj)
        error("Probe::Probe(Prj)", "hj not in [0,Hj[");
    this->hj = hj;
    if (dhi < 0 or prj->denHi <= dhi)
        error("Probe::Probe(Prj)", "dhi not in [0,denHi[");
    this->dhi = dhi;
    if (mj < 0 or prj->Mj <= mj)
        error("Probe::Probe(Prj)", "mj not in [0,Mj[");
    this->mj = mj;
    if (mi < 0 or prj->Mi <= mi)
        error("Probe::Probe(Prj)", "mi not in [0,Mi[");
    this->mi = mi;
    probeint = 1;
    ison = true;
    probes.push_back(this);
}

void Probe::setoffs(int probeoffs) {
    this->probeoffs = probeoffs;
}

void Probe::setint(int probeint) {
    this->probeint = probeint;
}

void Probe::on() {
    ison = true;
}

void Probe::off() {
    ison = false;
}

void Probe::doclose() {
    off();
    fclose(outfp);
}

void Probe::doprobepop() {
    if (not enabled or not ison or (simstep - probeoffs) % probeint != 0)
        return;
    float *dataf = pop->getfieldf(field);
    fprintf(outfp, "%d %e\n", simstep, dataf[n]);
}

void Probe::doprobeprj() {
    if (not enabled or not ison or simstep % probeint != 0)
        return;
    float dataf;
    int nj, ni;
    switch (field) {
        case AXOACT:
            dataf = prj->getfieldf(field)[k];
            break;
        case BWSUPINF:
        case BWSUP:
            dataf = prj->getfieldf(field)[k];
            break;
        case MIHJHI:
            dataf = prj->MIhjhi[hj * prj->denHi + dhi];
            break;
        case ZJ:
            dataf = prj->Zj[hj * prj->Mj + mj];
            break;
        case EJ:
            dataf = prj->Ej[hj * prj->Mj + mj];
            break;
        case PJ:
            dataf = prj->Pj[hj * prj->Mj + mj];
            break;
        case BJ:
            dataf = prj->Bj[hj * prj->Mj + mj];
            break;
        case ZI:
            dataf = prj->Zi[hj * prj->denNi + dhi * prj->Mi + mi];
            break;
        case EI:
            dataf = prj->Ei[hj * prj->denNi + dhi * prj->Mi + mi];
            break;
        case PI:
            dataf = prj->Pi[hj * prj->denNi + dhi * prj->Mi + mi];
            break;
        case EJI:
            nj = hj * prj->Mj + mj;
            ni = dhi * prj->Mi + mi;
            dataf = prj->Eji[nj * prj->denNi + ni];
            break;
        case PJI:
            nj = hj * prj->Mj + mj;
            ni = dhi * prj->Mi + mi;
            dataf = prj->Pji[nj * prj->denNi + ni];
            break;
        case PJPI:
            dataf = prj->Pj[hj * prj->Mj + mj] * prj->Pi[hj * prj->denNi + dhi * prj->Mi + mi];
            break;
        case WJI:
            nj = hj * prj->Mj + mj;
            ni = dhi * prj->Mi + mi;
            dataf = prj->Wji[nj * prj->denNi + ni];
            break;
        default:
            error("Probe::doprobeprj", "No such field: '" + fieldtostring(field) + "'");
    }
    fprintf(outfp, "%d %e\n", simstep, dataf);
}

void Probe::doprobe() {
    if (pop != nullptr)
        doprobepop();
    else if (prj != nullptr)
        doprobeprj();
}
