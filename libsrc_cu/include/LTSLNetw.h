/*****************************************************************

  Author: Anders Lansner

  Created: 2025-11-11

  Copyright (c) 2025 Anders Lansner

******************************************************************/

#ifndef __LTSLNetw_included
#define __LTSLNetw_included

#include "Netw.h"
#include "Pop.h"
#include "PatternFactory.h"
#include "Logger.h"
#include "LTSL.h"

class LTSLNetw {

 public:

    std::string name, mode;
    PatternFactory *inpatfact;
    int K, My, outoffs; // Ht == K
    std::vector<LTSL *> ltsls;     // One per y-bin
    Netw *ltslnetw;
    Pop *in_ypop, *out_ypop;
    Prj *y_yprj;
    float *ys, *in_ypop_ys;
    std::vector<std::vector<Logger *>> inpoplogs, outpoplogs;

 public:

    LTSLNetw(std::string name, int K, int My, float fs, float taumin, float taumax, int K_mode = 0);
    Prj *addinprj(LTSLNetw *srcnetw);
    void loaddataset(std::string filename);
    void setoutoffs(int offs);
    void settaup(float taup);
    void setoutagain(float again);
    void setmode(std::string mode);
    void reset();
    void setinput(int p = -1);
    void updstate(float prn, bool doupdbw = true);

     int addInPopLog(std::string field, std::string tag = "");
     int addInPopLog(std::vector<std::string> fields, std::string tag = "");
    void InPopLogsOff(int logid);
    void InPopLogsOn(int logid);
     int addOutPopLog(std::string field, std::string tag = "");
     int addOutPopLog(std::vector<std::string> fields, std::string tag = "");
    void OutPopLogsOff(int logid);
    void OutPopLogsOn(int logid);
};

#endif // __LTSLNetw_included
