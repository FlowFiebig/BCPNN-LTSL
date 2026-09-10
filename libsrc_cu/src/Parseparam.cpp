/*****************************************************************

  Author: Anders Lansner

  Included: 2021-08-18           Modified: 2021-08-19

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

#include "Globals.h"
#include "Parseparam.h"

using namespace std;

void Parseparam::error(string errloc, string errstr) {
    fprintf(stderr, "ERROR in Parseparam::%s: %s\n", errloc.c_str(), errstr.c_str());
    exit(0);
}

Parseparam::Parseparam(string paramfile) {
    _paramfile = paramfile;
    time(&_oldmtime);
}

void Parseparam::postparam(string paramstring, void *paramvalue, Value_t paramtype) {
    _paramstring.push_back(paramstring);
    _paramvalue.push_back(paramvalue);
    _paramtype.push_back(paramtype);
}

int Parseparam::findparam(string paramstring) {
    /* Find index in _paramstring */
    for (int i = 0; i < (int)_paramstring.size(); i++)
        if (strcmp(paramstring.c_str(), _paramstring[i].c_str()) == 0)
            return i;
    return -1;
}

void Parseparam::doparse() {
    char *str1, *str2;
    int k;
    string linestr, str;
    std::ifstream paramf(_paramfile.c_str());
    if (!paramf) {
        fprintf(stderr, "paramfile \"%s\" not found!\n", _paramfile.c_str());
        error("Parseparam::doparse", "Could not open paramfile: " + _paramfile);
    }
    bool comment;
    int lineno = 0;
    while (getline(paramf, linestr)) {
        comment = false;
        for (int i = 0; i < (int)linestr.size(); i++) {
            if (linestr[i] == '#')
                comment = true;
            if (comment)
                linestr[i] = (char)0;
        }
        if (strlen(linestr.c_str()) == 0)
            continue;
        str1 = strtok((char *)linestr.c_str(), " \t");
        str2 = strtok(nullptr, " \t");
        if (str1 == nullptr || str2 == nullptr)
            error("Parseparam::doparse", "Illegal str1 or str2: " + to_string(lineno));
        k = findparam(string(str1));
        if (k < 0)
            continue;
        switch (_paramtype[k]) {
            case Int:
                *(int *)_paramvalue[k] = atoi(str2);
                break;
            case Long:
                *(long *)_paramvalue[k] = atol(str2);
                break;
            case Float:
                *(float *)_paramvalue[k] = atof(str2);
                break;
            case Boole:
                if (strcmp(str2, "true") == 0)
                    *(bool *)_paramvalue[k] = true;
                else if (strcmp(str2, "false") == 0)
                    *(bool *)_paramvalue[k] = false;
                else
                    error("Parseparam::doparse", "Illegal boolean value");
                break;
            case String:
                *(string *)_paramvalue[k] = string(str2);
                break;
            default:
                error("Parseparam::doparse", "Illegal paramtype");
        }
        lineno++;
    }
    paramf.close();
}

bool Parseparam::haschanged() {
    struct stat file_stat;
    bool haschd = false;
    int err = stat(_paramfile.c_str(), &file_stat);
    if (err != 0) {
        perror("Parseparam::haschanged");
        exit(9);
    }
    if (file_stat.st_mtime > _oldmtime) {
        haschd = true;
        time(&_oldmtime);
    }
    return haschd;
}


void Parseparam::padwith0(string &str, int len) {
    while (str.size() < len)
        str.assign("0" + str);
}


string Parseparam::timestamp() {
    // current date/time based on current system
    time_t now = time(0);
    tm *ltm = localtime(&now);
    string yearstr = to_string(1900 + ltm->tm_year);
    string monthstr = to_string(1 + ltm->tm_mon);
    padwith0(monthstr, 2);
    string daystr = to_string(ltm->tm_mday);
    padwith0(daystr, 2);
    string hourstr = to_string(ltm->tm_hour);
    padwith0(hourstr, 2);
    string minstr = to_string(ltm->tm_min);
    padwith0(minstr, 2);
    string secstr = to_string(ltm->tm_sec);
    padwith0(secstr, 2);
    string timestamp = yearstr + monthstr + daystr + hourstr + minstr + secstr;
    return timestamp;
}


string Parseparam::dolog(string paramlogfile, bool usetimestamp) {
    _timestamp = timestamp();
    if (usetimestamp)
        _paramlogfile += "_" + _timestamp;
    _paramlogfile += ".sav";
    printf("Parameter logfile = %s\n", _paramlogfile.c_str());
    FILE *logf = fopen(_paramlogfile.c_str(), "w");
    if (usetimestamp)
        fprintf(logf, "%-30s%s\n", "TIMESTAMP", _timestamp.c_str());
    for (size_t p = 0; p < _paramstring.size(); p++) {
        fprintf(logf, "%-30s", _paramstring[p].c_str());
        switch (_paramtype[p]) {
            case Int:
                fprintf(logf, "%d", *(int *)_paramvalue[p]);;
                break;
            case Long:
                fprintf(logf, "%ld", *(long *)_paramvalue[p]);;
                break;
            case Float:
                fprintf(logf, "%f", *(float *)_paramvalue[p]);;
                break;
            case Boole:
                if (*(bool *)_paramvalue[p])
                    fprintf(logf, "true");
                else
                    fprintf(logf, "false");
                break;
            case String:
                fprintf(logf, "%s", (*(string *)_paramvalue[p]).c_str());;
                break;
            default:
                error("Parseparam::doparse", "Illegal paramtype");
        }
        fprintf(logf, "\n");
    }
    fclose(logf);
    return _timestamp;
}
