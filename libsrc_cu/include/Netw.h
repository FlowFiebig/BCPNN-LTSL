/*****************************************************************

  Author: Anders Lansner

  Created: 2023-09-08     Modified: 2024-02-02

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

#ifndef __Netw_included
#define __Netw_included

#include "Globals.h"
#include "Prjbase.h"

class Pop;
class TDPop;
class LIFPop;
class Prj;
class Prj2;
class Prj1;

class Netw {
//********** Static variables and methods **********
    protected:
        static std::vector<Netw *> netws;

    public:
        static bool netwnamexists(std::string name);
        static void clear();

        friend class Pop;
        friend class Prj;
        friend class Prj2;
        friend class Prj1;

// ************** Class variables and methods **************
        std::string name;
        int id;
        std::vector<Pop *> pops;
        std::vector<Pop *> inpops;
        std::vector<Pop *> utpops;
        std::vector<Prjbase *> prjs;

        Netw(std::string name = "");
        ~Netw();
        Pop *add(Pop *pop, bool isinpop = false, bool isutpop = false);
        TDPop *add(TDPop *pop, bool isinpop = false, bool isutpop = false);
        LIFPop *add(LIFPop *pop, bool isinpop = false, bool isutpop = false);        
        Prj *add(Prj *prj);
        Prj2 *add(Prj2 *prj);
        Prj1 *add(Prj1 *prj);
        void setinpopinput(std::vector<float> Xi = {});
        void setinpopinputs(std::vector<std::vector<float> > Xis = {});
        void setutpopinput(std::vector<float> Xj = {});
        void setutpopinputs(std::vector<std::vector<float> > Xjs = {});
        void popsreset(bool resetaxdelbuf = true);
        void popsresetbwsup();
        void popsupdsup();
        void popsupdact();
        void prjsupdact();
        void prjsreset();
        void prjsupdbwsup();
        void prjscontribute();
        void prjsupdzitrcs();
        void prjsupdtraces(std::vector<float> prns = {});
        void prjsupdbw(bool force = false);
        void prjsupdconns();
        void reset();
        void updstate(std::vector<float> prns, bool doupdbw = true);
        void updstate(float prn = 0, bool doupdbw = true);

};

#endif // __Netw_included
