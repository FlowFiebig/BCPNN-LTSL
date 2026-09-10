/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-01-24     Modified: 2024-01-24

*****************************************************************/
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

#ifndef __Probe_included
#define __Probe_included

#include "Globals.h"
#include "Pop.h"
#include "Prj.h"

class Pop;
class Prj;

class Probe {

// ********** Static variables and methods **********
    public:
        friend class Pop;
        friend class Prj;

        static std::vector<Probe *> probes;

// ************** Class variables and methods **************
        Prj *prj;
        Pop *pop;
        FILE *outfp;
        int field, nelem, probeoffs, probeint, n, k, hj, dhi, mj, mi;
        bool ison;

        Probe(Pop *pop, std::string fieldstr, std::string filename, int n = 0);
        Probe(Prj *prj, std::string fieldstr, std::string filename, int k = 0);
        Probe(Prj *prj, std::string fieldstr, std::string filename, int dhi, int hj, int mi, int mj);
        ~Probe();
        void on();
        void off();
        void setoffs(int probeoffs);
        void setint(int probeint);
        void doprobepop();
        void doprobeprj();
        void doprobehcconn();
        void doprobe();
        void doclose();
};

namespace Probing {

    extern bool enabled;

    void doprobing();
    void saveprobes();
    void closeprobes();
    void clearprobes();
    void enableprobes(bool value = true);

}

#endif // __Probe_included
