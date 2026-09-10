# BCPNNSim2.2 Release 20251205

This release contains a static library file lib/libbcpnn-2.2.a,
header file include/bcpnn.h, and this README.md file.

The BCPNNSim2.2 API calls cuda:

    Required CUDA version: ≥ 12.3
    Required GPU compute capability: ≥ 7.0
    Installation path assumption: /usr/local/cuda

To create and run a main.cpp program that calls the BCPNNSim2.2 API do the following:

1) Assume you have unzipped BCPNNSim2.2.zip

2) Write a main program, for instace:

#include <vector>
#include <string>
#include <sys/time.h>

#include "bcpnn.h"

using namespace std;
using namespace Globals;
using namespace Logging;

struct timeval total_time;

int main(void) {

    int nloop = 100;

    gsetseed(4711);

    Netw *netw1 = new Netw("netw1");

    Pop *pop1 = netw1->add(new Pop(10, 20, "pop1"));
    pop1->setnampl(0.5);

    Pop *pop2 = netw1->add(new Pop(20, 10, "pop2"));
    pop2->setnampl(0.5);

    Prj *prj12 = netw1->add(new Prj(pop1, pop2, "prj12"));

    Logger *act1 = new Logger(pop1, "ACT", "act1.bin");
    Logger *act2 = new Logger(pop2, "ACT", "act2.bin");
    Logger *sup2 = new Logger(pop2, "SUP", "sup2.bin");
    vector<float> prns {1.0};
    for (int loop = 0; loop < nloop; loop++) {
        if (loop == nloop - 1) {
           prns[0] = 1;
        } else {
           prns[0] = 0;
        }
        netw1->updstate(prns);
        advance();
     }
    closelogs();
    printf("Max memory footprint: Pops = %.1f (kB) Prjs = %.1f (MB)\n",
           Pop::nallocbyteall() / 1000.0, Prj::nallocbyteall() / 1000000.0);
    printf("Total simsteps = %d, time elapsed = %.3f sec\n",
           simstep, getDiffTime(total_time) / 1000);

    return 0;
}

3)  Compile and link the program:

Assuming main.cpp is in BCPNNSim2.2/:

nvcc main.cpp -I./include -L./lib -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
 -lbcpnn-2.2 -lcudart -lcurand -lcublas -o main

4) Run as ./main


There is currently no additional documentation beyond the *.h files in the include/ directory.

Questions and suggestions regarding this code should be addressed to
ala@kth.se and nbrav@kth.se.
