/*****************************************************************

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2024-10-13

  Copyright (c) 2024 Anders Lansner

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

#ifndef __Prj2_included
#define __Prj2_included

#include "Globals.h"
#include "Prj.h"
#include "Pop.h"

class Prj2 : public Prj {

    //********** Static variables and methods **********

public:
    // also use Prj::prjs

    //************** Class variables and methods **************

public:
    // std::string name;
    Prj2(Pop *srcpop, Pop *trgpop, int nactHi, int nsilHi, std::string name = "");

    // void fixupselfc(std::vector<int> shuffled) override;
    bool onhdiag(int hj, int hihjhi) override;
    int nhdiagoff() override;
    void upddenact() override;
    void updtrgact2();
    void updtraces(float prn = 0) override;
        
};

#endif // __Prj2_included
