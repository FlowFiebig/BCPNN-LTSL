/*****************************************************************

  Created: 2023-09-10  Modified: 2024-02-02

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

#include "LIFPop.h"
#include "Pop.h"
#include "TDPop.h"
#include "Prj.h"
#include "Prj2.h"
#include "Prj1.h"
#include "Netw.h"

using namespace std;
using namespace Globals;

//********** Static variables and methods **********

vector<Netw *> Netw::netws = vector<Netw *>(0);


bool Netw::netwnamexists(string name) {
    for (size_t p = 0; p < Netw::netws.size(); p++)
        if (netws[p]->name == name)
            return true;
    return false;
}


void Netw::clear() {
    netws.clear();
}

//************** Class variables and methods **************

Netw::Netw(string name) {
    if (name != "" and netwnamexists(name))
        error("Netw::Netw", "Netw-name " + name + " exists");
    this->name = name;
    netws.push_back(this);
}


Netw::~Netw() {
    for (auto* p : prjs) delete p;
    prjs.clear();

    for (auto* p : pops) delete p;
    pops.clear();
}


Pop *Netw::add(Pop *pop, bool isinpop, bool isutpop) {
    pops.push_back(pop);
    if (isinpop)
        inpops.push_back(pop);
    if (isutpop)
        utpops.push_back(pop);
    return (Pop *)pops.back();
}

TDPop *Netw::add(TDPop *pop, bool isinpop, bool isutpop) {
    pops.push_back(pop);
    if (isinpop)
        inpops.push_back(pop);
    if (isutpop)
        utpops.push_back(pop);
    return (TDPop *)pops.back();
}

LIFPop *Netw::add(LIFPop *pop, bool isinpop, bool isutpop) {
    pops.push_back(pop);
    if (isinpop)
        inpops.push_back(pop);
    if (isutpop)
        utpops.push_back(pop);
    return (LIFPop *)pops.back();
}


Prj *Netw::add(Prj *prj) {
    prjs.push_back(prj);
    return (Prj *)prjs.back();
}


Prj2 *Netw::add(Prj2 *prj) {
    prjs.push_back(prj);
    return (Prj2 *)prjs.back();
}


Prj1 *Netw::add(Prj1 *prj) {
    prjs.push_back(prj);
    return (Prj1 *)prjs.back();
}


void Netw::setinpopinput(vector<float> Xi) {
    if ((int)Xi.size() != inpops[0]->N)
        error("setinpopinputs", "Xi.size() -- N mismatch");
    inpops[0]->setinput(Xi.data());
}


void Netw::setinpopinputs(vector<vector<float> > Xis) {
    for (size_t p = 0; p < Xis.size(); p++)
        setinpopinput(Xis[p]);
}


void Netw::setutpopinput(vector<float> Xj) {
    if ((int)Xj.size() != utpops[0]->N)
        error("setutpopinputs", "Xj.size() -- N mismatch");
    utpops[0]->setinput(Xj.data());
}


void Netw::setutpopinputs(vector<vector<float> > Xjs) {
    for (size_t p = 0; p < Xjs.size(); p++)
        setutpopinput(Xjs[p]);
}


void Netw::popsreset(bool resetaxdelbuf) {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->reset(resetaxdelbuf);
}


void Netw::popsresetbwsup() {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->resetbwsup();
}


void Netw::popsupdsup() {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->updsup();
}


void Netw::popsupdact() {
    for (size_t p = 0; p < pops.size(); p++)
        pops[p]->updact();
}


void Netw::prjsupdact() {
    for (size_t p = 0; p < prjs.size(); p++)
        prjs[p]->upddenact();
}


void Netw::prjsupdzitrcs() {
    for (size_t p = 0; p < prjs.size(); p++)
        prjs[p]->updzitrc();
}


void Netw::prjsupdbwsup() {
    for (size_t p = 0; p < prjs.size(); p++)
        prjs[p]->updbwsup();
}


void Netw::prjscontribute() {
    for (size_t p = 0; p < prjs.size(); p++)
        prjs[p]->contribute();
}


void Netw::prjsupdtraces(vector<float> prns) {
    if (prns.size() != prjs.size())
        error("Netw::prjsupdtraces", "prns -- prjs size mismatch");
    for (size_t p = 0; p < prjs.size(); p++)
        prjs[p]->updtraces(prns[p]);
}


void Netw::prjsupdbw(bool force) {
    for (size_t p = 0; p < prjs.size(); p++)
        prjs[p]->updbw();
}


void Netw::prjsupdconns() {
    Prj::updconnsall();
}


void Netw::reset() {
    Pop::resetall();
    Prj::resetall();
}


void Netw::updstate(vector<float> prns, bool doupdbw) {
    popsupdsup();
    popsupdact();
    prjsupdact();
    prjsupdzitrcs();
    popsresetbwsup();
    prjsupdbwsup();
    prjscontribute();
    prjsupdtraces(prns);
    if (doupdbw)
        prjsupdbw();
}


void Netw::updstate(float prn, bool doupdbw) {
    vector<float> prns = vector<float>(prjs.size());
    fill(prns.begin(), prns.end(), prn);
    updstate(prns, doupdbw);
}

