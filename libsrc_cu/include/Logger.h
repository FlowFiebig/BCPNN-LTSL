/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-01-27     Modified: 2024-01-27

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
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/

#ifndef __Logger_included
#define __Logger_included

#include "Globals.h"

class Pop;
class Prjbase;

class Logger {

// ********** Static variables and methods **********

    protected:
        friend class Pop;
        friend class Prjbase;

        static Logger *findlog(std::string logname);

    public:
        static std::vector<Logger *> loggers;
        // static void clearlogs();
        // static void enablelogs(bool value = true);
        // static void stoplog(std::string logname);
        // static void startlog(std::string logname);
        // static void closelog(std::string logname);

// ************** Class variables and methods **************

        Pop *pop;
        Prjbase *prj;
        FILE *logfp;
        std::string logname;
        std::vector<int> intvec;
        std::vector<float> fltvec; 
        std::vector<std::vector<int> > logdatai;
        std::vector<std::vector<float> > logdataf;
        int field, logoffs, logint, len, nelem;
        bool ison;
        Logger(Pop *pop, std::string field, std::string logname = "");
        Logger(Prjbase *prj, std::string field, std::string logname = "");
        ~Logger();
        void on();
        void off();
        void setoffs(int logoffs);
        void setint(int logint);
        void dopoplog();
        void doprjlog();
        void dolog(bool force = false);
        std::vector<std::vector<int> > getdatai();
        std::vector<std::vector<float> > getdataf();
        void dopopsave();
        void doprjsave();
        void dosave();
        void doclose();
};

namespace Logging {

    extern  bool enabled;

    void clearlogs();
    void enablelogs(bool set = true);
    void dologging();
    void savelogs();
    void closelogs();

}

#endif // __Logger_included
