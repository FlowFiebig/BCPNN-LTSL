/*****************************************************************

  Created: 2025-11-11

  Authors: Anders Lansner

  Copyright (c) 2025 Anders Lansner

******************************************************************/

#include <algorithm>
#include <string>
#include <cctype>

#include "Globals.h"
#include "Pop.h"
#include "Pop.cuh"
#include "Prj.h"
#include "Netw.h"
#include "PatternFactory.h"
#include "LTSL.h"
#include "LTSLNetw.h"

using namespace std;
using namespace Globals;

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

LTSLNetw::LTSLNetw(std::string name, int K, int My, float fs, float taumin, float taumax, int K_mode) {
    // K <--> Ht, My <--> M
    this->name = name;
    this->K = K;
    this->My = My;
    
    ltslnetw = new Netw(name);
    in_ypop = ltslnetw->add(new Pop(K, My, ""));
    in_ypop->setactfn("SOFTMAX");
    out_ypop = ltslnetw->add(new Pop(1, My, ""));
    out_ypop->setactfn("SOFTMAX");

    outoffs = 1;

    inpatfact = nullptr;

    y_yprj = (Prj *)ltslnetw->add(new Prj(in_ypop, out_ypop, ""));

    for (int my = 0; my < My; my++) {
        ltsls.push_back(new LTSL(fs, taumin, taumax, K, K_mode));
        ltsls.back()->reset();
    }
    
    in_ypop_ys = new float[My*K];

    setmode("train");

}

Prj *LTSLNetw::addinprj(LTSLNetw *srcnetw) {
    Prj *tmp = new Prj(srcnetw->in_ypop, out_ypop);
    srcnetw->ltslnetw->add(tmp);
    return tmp;
}

void LTSLNetw::loaddataset(string filename) {
    inpatfact = new PatternFactory(My, 1);
    inpatfact->readpats(filename);
}

void LTSLNetw::setoutoffs(int outoffs) {
    this->outoffs = outoffs;
}

void LTSLNetw::settaup(float taup) {
    y_yprj->settaup(taup);
}

void LTSLNetw::setoutagain(float again) {
    out_ypop->setagain(again);
}

void LTSLNetw::setmode(string mode) {
    this->mode = mode;
    if (mode == "train")
        y_yprj->setbwgain(0);
    else if (mode == "test" or mode == "ustest" or mode == "generate")
        y_yprj->setbwgain(1);
    else
        error("LTSLNetw", "No such mode");
}

void LTSLNetw::reset() {
    for (int my = 0; my < My; my++) {
        ltsls[my]->reset();
        memset(in_ypop_ys, 0, K*my);
    }
    ltslnetw->reset();
}

void LTSLNetw::setinput(int p) {
    float *ltslin;
    if (mode == "generate")
        ltslin = out_ypop->act;
    else
        ltslin = inpatfact->getfpat(p);
    for (int my = 0; my < My; my++) {
        ys = ltsls[my]->step(ltslin[my]);
        for (int k = 0; k < K; k++)
            in_ypop_ys[k*My + my] = ys[K-k-1];
    }
    in_ypop->setinput(in_ypop_ys);
    if (mode == "train")
        out_ypop->setinput(inpatfact->getfpat(p + outoffs));
    else
        out_ypop->setinput();

}

void LTSLNetw::updstate(float prn, bool doupdbw) {
    ltslnetw->updstate(prn, doupdbw);
}

int LTSLNetw::addInPopLog(string field, string tag) {
    if (tag == "")
        tag = mode;
    inpoplogs.push_back(vector<Logger *>());
    inpoplogs.back().push_back(new Logger(this->in_ypop, field, name + "_inpop_" + tag + "_" +
                                          to_lower(field) + ".bin"));
    return inpoplogs.size()-1;
}

int LTSLNetw::addInPopLog(vector<string> fields, string tag) {
    if (tag == "")
        tag = mode;
    inpoplogs.push_back(vector<Logger *>());
    for (auto field : fields)
        inpoplogs.back().push_back(new Logger(this->in_ypop, field, name + "_inpop_" + tag + "_" +
                                              to_lower(field) + ".bin"));
    return inpoplogs.size()-1;
}

void LTSLNetw::InPopLogsOff(int logid) {
    for (auto logger : inpoplogs[logid])
        logger->off();
}

void LTSLNetw::InPopLogsOn(int logid) {
    for (auto logger : inpoplogs[logid])
        logger->on();
}

int LTSLNetw::addOutPopLog(string field, string tag) {
    if (tag == "")
        tag = mode;
    outpoplogs.push_back(vector<Logger *>());
    outpoplogs.back().push_back(new Logger(this->out_ypop, field, name + "_outpop_" + tag + "_" +
                                           to_lower(field) + ".bin"));
    return outpoplogs.size()-1;
}

int LTSLNetw::addOutPopLog(vector<string> fields, string tag) {
    if (tag == "")
        tag = mode;
    outpoplogs.push_back(vector<Logger *>());
    for (auto field : fields)
        outpoplogs.back().push_back(new Logger(this->out_ypop, field, name + "_outpop_" + tag + "_" +
                                               to_lower(field) + ".bin"));
    return outpoplogs.size()-1;
}

void LTSLNetw::OutPopLogsOff(int logid) {
    for (auto logger : outpoplogs[logid])
        logger->off();
}

void LTSLNetw::OutPopLogsOn(int logid) {
    for (auto logger : outpoplogs[logid])
        logger->on();
}

