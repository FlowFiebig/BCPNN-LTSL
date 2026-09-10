/*****************************************************************

  Created: 2024-01-28  Modified: 2024-01-28

  Authors: Anders Lansner, Naresh Ravchandran

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

#include "Pop.h"
#include "Prjbase.h"
#include "Logger.h"

using namespace std;
using namespace Globals;
using namespace Logging;

//********** Static variables and methods **********

vector<Logger *> Logger::loggers;
bool Logging::enabled = true;

//************** Class variables and methods **************

void Logging::clearlogs() {
    Logger::loggers.clear();
}

void Logging::enablelogs(bool boolval) {
    enabled = boolval;
}

void Logging::dologging() {
    for (size_t p = 0; p < Logger::loggers.size(); p++)
        Logger::loggers[p]->dolog();
}

void Logging::savelogs() {
    for (size_t p = 0; p < Logger::loggers.size(); p++)
        Logger::loggers[p]->dosave();
}

void Logging::closelogs() {
    for (size_t p = 0; p < Logger::loggers.size(); p++)
        Logger::loggers[p]->doclose();
}

Logger *Logger::findlog(string logname) {
    for (size_t p = 0; p < loggers.size(); p++)
        if (loggers[p]->logname == logname)
            return loggers[p];
    return nullptr;
}

Logger::Logger(Pop *pop, string field, string logname) {
    this->pop = pop;
    this->prj = nullptr;
    this->field = stringtofield(field);
    ison = true;
    len = 0;
    nelem = pop->getnelem(this->field);
    pop->getfieldf(this->field);
    this->logname = logname;
    this->logoffs = 0;
    this->logint = 1;
    logfp = fopen(logname.c_str(), "wb");
    loggers.push_back(this);
}

Logger::Logger(Prjbase *prj, string fieldstr, string logname) {
    pop = nullptr;
    this->prj = prj;
    field = stringtofield(fieldstr);
    ison = true;
    len = 0;
    nelem = prj->getnelem(field);
    this->logname = logname;
    this->logoffs = 0;
    this->logint = 1;
    logfp = fopen(logname.c_str(), "wb");
    loggers.push_back(this);
}

Logger::~Logger() {
    // Need not destroy std::vector objects (logdata)
}

void Logger::setoffs(int logoffs) {
    this->logoffs = logoffs;
}

void Logger::setint(int logint) {
    if (logint <= 0)
        error("Logger::setlogint", "Illegal: logint <= 0");
    this->logint = logint;
}

void Logger::on() {
    ison = true;
}

void Logger::off() {
    ison = false;
}

void Logger::dopoplog() {
    float *dataf;
    dataf = pop->getfieldf(field);
    if (logdataf.size() <= len) {
        fltvec.assign(dataf, dataf + nelem);
        logdataf.push_back(fltvec); 
    } else
        logdataf[len].assign(dataf, dataf + nelem);
    len++;
    if (nelem * len > 1e7)
        dopopsave();
}

void Logger::doprjlog() {
    if (field == HIFANOUT or field == HIHJHI or field == CHJHIX or
        field == IFANOUT or field == IJI or field == CJIX) {
        int *datai;
        switch (field) {
            case CHJHI:
            case CJI:
            case HIHJHI:
            case HIFANOUT:
                datai = prj->getfieldi(field);
                break;
            case CHJHIX:
            case CJIX:
                datai = prj->expandfieldi(field);
                break;
            default:
                error("Logger::doprjlog", "No such field: '" + fieldtostring(field) + "'");
        }
        if (not datai)
            error("Logger::doprjlog", "Field " + fieldtostring(field) + " is invalid");
        if (logdatai.size() <= len) {
            intvec.assign(datai, datai + nelem);
            logdatai.push_back(intvec);
        } else
            logdatai[len].assign(datai, datai + nelem);
        len++;
        if (nelem * len > 1e7)
            doprjsave();
        return;
    }
    float *dataf;
    switch (field) {
        case DENACTX:
        case MIJI:
        case ZIX:
        case EIX:
        case PIX:
        case PJIX:
        case WJIX:
        case MIHJHIX:
        case MIJIX:
            dataf = prj->expandfieldf(field);
            break;
        case MIHJHI:
            dataf = prj->expandfieldf1(field);
            break;
        case AXOACT:
        case DENACT:
        case BWSUPINF:
        case BWSUP:
        case ZI:
        case EI:
        case PI:
        case ZJ:
        case EJ:
        case PJ:
        case BJ:
        case EJI:
        case PJI:
        case PJPI:
        case WJI:
            dataf = prj->getfieldf(field);
            break;
        default:
            error("Logger::doprjlog", "No such field: '" + fieldtostring(field) + "'");
    }
    if (not dataf)
        error("Logger::doprjlog", "Field " + fieldtostring(field) + " is invalid");
    if (logdataf.size() <= len) {
        fltvec.assign(dataf, dataf + nelem);
        logdataf.push_back(fltvec);
    } else
        logdataf[len].assign(dataf, dataf + nelem);
    len++;
    if (nelem * len > 1e7)
        doprjsave();
}

void Logger::dolog(bool force) {
    if (enabled and ison and (simstep - logoffs) % logint == 0) {
        if (pop != nullptr)
            dopoplog();
        else if (prj != nullptr)
            doprjlog();
        else
            error("Logger::dolog", "Missing object to log");
    }
}

vector<vector<int> > Logger::getdatai() {
    return logdatai;
}

vector<vector<float> > Logger::getdataf() {
    return logdataf;
}

void Logger::dopopsave() {
    string logfilename = logname;
    if (logfilename.find(".log") == string::npos)
        logfilename.append(".log");
    if (logdatai.size() > 0) {
        for (size_t p = 0; p < logdatai.size(); p++)
            fwrite(logdatai[p].data(), sizeof(int), logdatai[p].size(), logfp);
    }
    if (logdataf.size() > 0) {
        for (size_t p = 0; p < logdataf.size(); p++) {
            fwrite(logdataf[p].data(), sizeof(float), logdataf[p].size(), logfp);
        }
        fflush(logfp);
    }
    logdatai.resize(0);
    logdataf.resize(0);
    len = 0;
    // Leaving logfile open
}

void Logger::doprjsave() {
    string logfilename = logname;
    if (logfilename.find(".log") == string::npos)
        logfilename.append(".log");
    if (logdatai.size() > 0) {
        for (size_t p = 0; p < len; p++)
            fwrite(logdatai[p].data(), sizeof(int), logdatai[p].size(), logfp);
    }
    if (logdataf.size() > 0) {
        for (size_t p = 0; p < len; p++)
            fwrite(logdataf[p].data(), sizeof(float), logdataf[p].size(), logfp);
    }
    fflush(logfp);
    logdatai.resize(0);
    logdataf.resize(0);
    len = 0;
    // Leaving logfile open.
}

void Logger::dosave() {
    if (pop != nullptr)
        dopopsave();
    else if (prj != nullptr)
        doprjsave();
    else
        error("Logger::dosave", "Missing object to log");
}

void Logger::doclose() {
    dosave();
    off();
    if (logfp != nullptr)
        fclose(logfp);
}
