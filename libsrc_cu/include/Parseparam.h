/*****************************************************************

  Author: Anders Lansner

  Included: 2021-08-18           Modified: 2024-06-13

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

#ifndef __Parseparam_INCLUDED__
#define __Parseparam_INCLUDED__

#include "Globals.h"

enum Value_t { Int = 0, Long, Float, Boole, String } ;

class Parseparam {

    public:

        Parseparam(std::string paramfile) ;

        void error(std::string errloc, std::string errstr) ;

        void postparam(std::string paramstring, void *paramvalue, Value_t paramtype) ;

        int findparam(std::string paramstring) ;

        void doparse() ;

        bool haschanged() ;

        void padwith0(std::string &str, int len) ;

        std::string timestamp() ;

        std::string dolog(std::string paramlogfile, bool usetimestamp) ;

    protected:

        std::string _paramlogfile, _paramfile, _timestamp;
        std::vector<std::string> _paramstring;
        std::vector<void *> _paramvalue;
        std::vector<Value_t> _paramtype;
        time_t _oldmtime;

} ;

#endif // __Parseparam_INCLUDED__
