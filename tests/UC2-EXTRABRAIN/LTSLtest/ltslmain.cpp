/*

  Author: Anders Lansner
  Created: 2025-11-21

      Modified by: Florian Fiebig, 2026-03-23
      to avoid naming output files by particular parameter values, which complicates more general work

  This mgmain2_md1_m main program has one outpop, outpop_md1, and uses it internally and for output.

*/

#include <vector>
#include <string>
#include <tuple>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <sys/time.h>

#include "bcpnn.h"

using namespace std;
using namespace Globals;
using namespace Logging;


Netw *tdnetw;
TDPop *ltslpop; // In essence using only the LTSL machinery on top of the K * My inpop.
Pop *outpop_md1;
Prj *ltslprj_md1;
PatternFactory *inpatfact_md1 = nullptr;
string infilename_md = "";
// Optional sidecar schedule file (one row per block: "trnpat vanpat tenpat").
// Empty -> classic single-block path used by Mains.general() / Mains.multi().
// Non-empty -> multi-block walk-forward path used by Mains.multislice():
// network state is preserved across blocks because the whole schedule runs
// inside this one process.
string schedule_file = "";

int K = 10, mean_nbin = 40 , deriv1st_nbin = 31, mean_deriv1st_nbin = mean_nbin * deriv1st_nbin,
    trnpat = 750, vanpat = 0, tenpat = 250, offs = 1, ngenstep = 100, warmup_pat = 0;
float taumin = 0.001, fs = 1./taumin, taumax = 1., outagain_gen = 1;
string K_mode_str = "logarithmic";
int K_mode = 0;
float taup = -1, eps = -1;
int bdebias = 1;

struct timeval total_time;

void parseparams(std::string paramfile) {
    Parseparam *parseparam = new Parseparam(paramfile);
    
    parseparam->postparam("mean_nbin", &mean_nbin, Int);
    parseparam->postparam("deriv1st_nbin", &deriv1st_nbin, Int);
    // Aliases for alternative feature sets (e.g. feature_set=return_variance).
    // The network only needs the output-marginal dimension (mean_nbin) and the
    // inner factor (deriv1st_nbin); the meaning of the channels is irrelevant
    // here, so return_nbin/variance_nbin map onto the same two variables. A
    // .par file supplies exactly one naming convention; the unused names simply
    // never match a line and keep their defaults.
    parseparam->postparam("return_nbin", &mean_nbin, Int);
    parseparam->postparam("variance_nbin", &deriv1st_nbin, Int);
    parseparam->postparam("K", &K, Int);
    parseparam->postparam("K_mode", &K_mode_str, String);
    parseparam->postparam("taumin", &taumin, Float);
    parseparam->postparam("taumax", &taumax, Float);
    parseparam->postparam("offs", &offs, Int);
    parseparam->postparam("outagain_gen", &outagain_gen, Float);
    parseparam->postparam("trnpat", &trnpat, Int);
    parseparam->postparam("vanpat", &vanpat, Int);
    parseparam->postparam("tenpat", &tenpat, Int);
    parseparam->postparam("ngenstep", &ngenstep, Int);
    parseparam->postparam("warmup_pat", &warmup_pat, Int);
    parseparam->postparam("infilename_md", &infilename_md, String);

    parseparam->postparam("taup", &taup, Float);
    parseparam->postparam("eps", &eps, Float);
    parseparam->postparam("bdebias", &bdebias, Int);

    parseparam->postparam("schedule_file", &schedule_file, String);

    parseparam->doparse();

    fs = 1/taumin;
    mean_deriv1st_nbin = mean_nbin * deriv1st_nbin;
    K_mode = (K_mode_str == "linear") ? 1 : 0;
}

void makeinpats() {
    inpatfact_md1 = new PatternFactory(mean_deriv1st_nbin, 1);
    inpatfact_md1->readpats(infilename_md, trnpat + vanpat + tenpat);
}

// def pats_to_dists(pats, mean_nbin, deriv1st_bin) :
//     ydists = []
//     for pat in pats :
//         ydists.append(np.sum(pat.reshape(mean_nbin, deriv1st_nbin), 1))
//     return np.array(ydists)

vector<float> pat_to_dist(vector<float> pat, int mean_nbin, int deriv1st_nbin) {
    vector<float> dist(mean_nbin);
    for (int m_nbin = 0; m_nbin < mean_nbin; m_nbin++)
        for (int d_nbin = 0; d_nbin < deriv1st_nbin; d_nbin++)
            dist[m_nbin] += pat[m_nbin * deriv1st_nbin + d_nbin];
    return dist;    
}

vector<vector<float>> pats_to_dists(vector<vector<float>> pats, int mean_nbin, int deriv1st_nbin) {
    vector<vector<float>> dists(pats.size(), vector<float>(mean_nbin));
    for (int p = 0; p < pats.size(); p++)
        dists[p] = pat_to_dist(pats[p], mean_nbin, deriv1st_nbin);
    return dists;
}

// Read a float32 activations file produced by a Pop Logger back into [npat, nelem].
// Used instead of Logger::getdataf() because the Logger auto-flushes and clears its
// in-memory buffer once nelem*len > 1e7, so getdataf() only returns the tail after
// the last flush. Reading from disk gives the full stream regardless of size.
vector<vector<float>> readpatfile(const string &filename, int nelem) {
    vector<vector<float>> pats;
    FILE *f = fopen(filename.c_str(), "rb");
    if (f == nullptr) {
        printf("readpatfile: cannot open %s\n", filename.c_str());
        return pats;
    }
    vector<float> buf(nelem);
    while (fread(buf.data(), sizeof(float), nelem, f) == (size_t)nelem)
        pats.push_back(buf);
    fclose(f);
    return pats;
}

// Reduce the outer (mean x deriv1st) activation file into a marginal mean
// distribution file. The Pop Logger has already written the .act.bin stream
// to disk (with Logger::dosave() flushing the final in-memory chunk), so we
// just read it back, marginalize over deriv1st_nbin, and write the result.
void write_dist_from_act(const string &act_filename, const string &dist_filename,
                         int mean_nbin, int deriv1st_nbin) {
    int nelem = mean_nbin * deriv1st_nbin;
    vector<vector<float>> pats = readpatfile(act_filename, nelem);
    vector<vector<float>> dists = pats_to_dists(pats, mean_nbin, deriv1st_nbin);
    tofile(dists, dist_filename);
}

// Read a multi-block schedule file with one row per block:
//   trnpat_b  vanpat_b  tenpat_b
// (whitespace-separated ints). Comments starting with '#' and blank lines are
// ignored. Sums are also returned so the caller can sanity-check against
// trnpat/vanpat/tenpat from the .par file.
vector<tuple<int,int,int>> read_schedule(const string &filename,
                                         int &sum_trn, int &sum_van, int &sum_ten) {
    vector<tuple<int,int,int>> schedule;
    sum_trn = sum_van = sum_ten = 0;
    ifstream sf(filename.c_str());
    if (!sf) {
        fprintf(stderr, "ERROR: schedule_file '%s' not found\n", filename.c_str());
        exit(1);
    }
    string line;
    int lineno = 0;
    while (getline(sf, line)) {
        lineno++;
        size_t hash = line.find('#');
        if (hash != string::npos)
            line = line.substr(0, hash);
        istringstream iss(line);
        int tp, vp, ep;
        if (!(iss >> tp >> vp >> ep))
            continue; // skip blank / comment-only lines
        if (tp < 0 || vp < 0 || ep < 0) {
            fprintf(stderr, "ERROR: negative count on schedule line %d\n", lineno);
            exit(1);
        }
        schedule.emplace_back(tp, vp, ep);
        sum_trn += tp;
        sum_van += vp;
        sum_ten += ep;
    }
    return schedule;
}

// Run one multi-block phase as N_b equal ticker segments.  For k > 0 the
// first warmup_pat patterns are forward-only (updstate(0), logger off).
// learn_mode: 1 = training (weight updates on scored suffix), 0 = eval.
// log_on: when true, logger is on only during each segment's scored suffix.
void run_multislice_phase(int phase_total, int per_t_len, int cursor,
                          int learn_mode, Logger *logger, bool log_on) {
    if (phase_total <= 0 || per_t_len <= 0)
        return;
    int n_seg = phase_total / per_t_len;
    for (int k = 0; k < n_seg; k++) {
        int seg = cursor + k * per_t_len;
        int w = (k > 0 && warmup_pat > 0) ? warmup_pat : 0;
        int scored = per_t_len - w;
        if (log_on && logger)
            logger->off();
        for (int p = 0; p < w; p++) {
            ltslpop->setinput(inpatfact_md1->getfpat(seg + p));
            if (learn_mode)
                outpop_md1->setinput(inpatfact_md1->getfpat(seg + p + offs));
            tdnetw->updstate(0);
            advance();
        }
        if (log_on && logger)
            logger->on();
        for (int p = 0; p < scored - offs; p++) {
            ltslpop->setinput(inpatfact_md1->getfpat(seg + w + p));
            if (learn_mode)
                outpop_md1->setinput(inpatfact_md1->getfpat(seg + w + p + offs));
            tdnetw->updstate(learn_mode ? 1 : 0);
            advance();
        }
        if (log_on && logger)
            logger->off();
    }
}

void run_multislice_phase_flat(int phase_total, int cursor, int learn_mode,
                               Logger *logger) {
    if (phase_total <= 0)
        return;
    if (learn_mode) {
        for (int p = 0; p < phase_total - offs; p++) {
            ltslpop->setinput(inpatfact_md1->getfpat(cursor + p));
            outpop_md1->setinput(inpatfact_md1->getfpat(cursor + p + offs));
            tdnetw->updstate(1);
            advance();
        }
    } else {
        if (logger)
            logger->on();
        for (int p = 0; p < phase_total - offs; p++) {
            ltslpop->setinput(inpatfact_md1->getfpat(cursor + p));
            tdnetw->updstate(0);
            advance();
        }
        if (logger)
            logger->off();
    }
}

int main(int argc, char **args) {
    gettimeofday(&total_time, 0);

    string paramfile = "mgmain2_mean_1stderiv.par";
    if (argc > 1)
        paramfile = args[1];
    printf("paramfile = %s\n", paramfile.c_str());

    parseparams(paramfile);

    // Per-ticker phase lengths from .par (multislice); schedule totals override
    // trnpat/vanpat/tenpat below but these per-ticker values stay for segmenting.
    int trnpat_per = trnpat, vanpat_per = vanpat, tenpat_per = tenpat;

    // When schedule_file is set, Python (Mains.general / multi / multislice)
    // writes the sidecar before invoking ltslmain. The schedule is the source
    // of truth for total trnpat/vanpat/tenpat (one row for general/multi,
    // many rows for multislice). Values in the .par file may carry different
    // semantics (e.g. per-block sizes in multislice) and are overridden here.
    vector<tuple<int,int,int>> schedule;
    if (!schedule_file.empty()) {
        int sum_trn = 0, sum_van = 0, sum_ten = 0;
        schedule = read_schedule(schedule_file, sum_trn, sum_van, sum_ten);
        if (schedule.empty()) {
            fprintf(stderr, "ERROR: schedule_file '%s' contains no blocks\n",
                    schedule_file.c_str());
            exit(1);
        }
        printf("schedule_file = %s  (%zu blocks; sum trn=%d van=%d ten=%d)\n",
               schedule_file.c_str(), schedule.size(), sum_trn, sum_van, sum_ten);
        trnpat = sum_trn;
        vanpat = sum_van;
        tenpat = sum_ten;
        // Autoregression across stock / block boundaries is ill-defined.
        if (schedule.size() > 1)
            ngenstep = 0;
    }

    if (warmup_pat > 0)
        printf("warmup_pat = %d  (forward-only at ticker transitions k>0)\n",
               warmup_pat);

    if (taup < 0) {
        taup = trnpat * timestep;
        printf("taup = %.3f\n", taup);
    }
    if (eps < 0) {
        eps = 1. / (1 + trnpat);
        printf("eps = %.1e\n", eps);
    }

    printf("K_mode = %s\n", K_mode_str.c_str());
    tdnetw = new Netw("tdnetw");

    ltslpop = tdnetw->add(new TDPop(K, mean_deriv1st_nbin, "ltslpop", fs, taumin, taumax, K_mode));
    ltslpop->setactfn("SOFTMAX");
    outpop_md1 = tdnetw->add(new Pop(1, mean_deriv1st_nbin, "outpop_md1"));
    outpop_md1->setactfn("SOFTMAX");
    ltslprj_md1 = tdnetw->add(new Prj(ltslpop, outpop_md1, "ltslprj_md1"));

    makeinpats();

    if (trnpat == -1)
        trnpat = inpatfact_md1->getnpat();
    ltslprj_md1->settaup(taup);
    // Forward the parsed/defaulted eps to the projection. Without this call the
    // Prj silently keeps the library default EPS=1e-7, so the P-trace floors do
    // not regularize rare-bin log-odds weights at all (the printed eps was a lie).
    ltslprj_md1->seteps(eps);
    // Invert the eps trace floor when computing Bj so the recalled pmf is not
    // inflated with ~M*eps uniform mass (critical for tail/VaR calibration).
    ltslprj_md1->setbdebias(bdebias != 0);
    printf("bdebias = %d\n", bdebias);

    if (schedule_file.empty()) {
        // ---------------- Classic single-block path -----------------
        struct timeval setinput_time;
        float setinputime = 0.;
        // Train
        Logger *troutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_tr_act.bin");
        printf("TRAINING from simstep = %d offs = %d\n", simstep, offs);
        outpop_md1->setigain(1.);
        ltslprj_md1->setbwgain(0.);
        for (int p = 0; p < trnpat - offs; p++) {
            gettimeofday(&setinput_time, 0);
            ltslpop->setinput(inpatfact_md1->getfpat(p));
            setinputime += getDiffTime(setinput_time);
            outpop_md1->setinput(inpatfact_md1->getfpat(p + offs));
            tdnetw->updstate(1);
            advance();
        }
        printf("setinputime elapsed = %.3f sec\n",setinputime / 1000.);
        troutacts->off();

        Logger *inacts = new Logger(ltslpop, "ACT", "netw1_ltslpop_tetr_act.bin");
        Logger *insups = new Logger(ltslpop, "SUP", "netw1_ltslpop_tetr_sup.bin");
        // Test on training data
        ltslprj_md1->setbwgain(1.);
        Logger *tetroutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_tetr_act.bin");
        Logger *tetroutsups = new Logger(outpop_md1, "SUP", "netw1_outpop_tetr_sup.bin");
        outpop_md1->setigain(0.);
        ltslprj_md1->setbwgain(1.);
        printf("TESTING on training data from simstep = %d\n", simstep);
        for (int p = 0; p < trnpat - offs; p++) {
            ltslpop->setinput(inpatfact_md1->getfpat(p));
            tdnetw->updstate(0);
            advance();
        }
        inacts->off();
        insups->off();
        tetroutacts->off();
        tetroutsups->off();

        // Validation (held-out patterns right after training; no learning)
        Logger *vanoutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_van_act.bin");
        Logger *vanoutsups = new Logger(outpop_md1, "SUP", "netw1_outpop_van_sup.bin");
        if (vanpat > 0)
            printf("VALIDATING on held-out data from simstep = %d\n", simstep);
        for (int p = trnpat; p < trnpat + vanpat - offs; p++) {
            ltslpop->setinput(inpatfact_md1->getfpat(p));
            tdnetw->updstate(0);
            advance();
        }
        vanoutacts->off();
        vanoutsups->off();

        // Test on unseen data
        Logger *teusoutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_teus_act.bin");
        if (-1 <= tenpat) {
            if (tenpat == -1)
                tenpat = inpatfact_md1->getnpat() - trnpat - vanpat;
            if (tenpat > 0)
                printf("TESTING on unseen data from simstep = %d\n", simstep);
            for (int p = trnpat + vanpat; p < trnpat + vanpat + tenpat - offs; p++) {
                ltslpop->setinput(inpatfact_md1->getfpat(p));
                tdnetw->updstate(0);
                advance();
            }
            printf("\n");
        }
        teusoutacts->off();

        // Generation
        outpop_md1->setigain(0.);
        outpop_md1->setagain(outagain_gen);
        Logger *genoutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_gen_act.bin");
        if (ngenstep > 0) {
            printf("GENERATING from simstep = %d\n", simstep);
            for (int step = 0; step < ngenstep; step++) {
                ltslpop->setinput(outpop_md1->act);
                tdnetw->updstate(0);
                advance();
            }
        }
        genoutacts->off();

        // Flush each act Logger's remaining in-memory buffer to disk so that
        // the .act.bin files contain the full stream. Reading the files back
        // below then marginalizes the full pattern set into *_dist.bin, instead
        // of only the tail the Logger still happens to hold in RAM.
        troutacts->dosave();
        tetroutacts->dosave();
        vanoutacts->dosave();
        teusoutacts->dosave();
        genoutacts->dosave();

        write_dist_from_act("netw1_outpop_tr_act.bin",   "netw1_outpop_tr_dist.bin",   mean_nbin, deriv1st_nbin);
        write_dist_from_act("netw1_outpop_tetr_act.bin", "netw1_outpop_tetr_dist.bin", mean_nbin, deriv1st_nbin);
        write_dist_from_act("netw1_outpop_van_act.bin",  "netw1_outpop_van_dist.bin",  mean_nbin, deriv1st_nbin);
        write_dist_from_act("netw1_outpop_teus_act.bin", "netw1_outpop_teus_dist.bin", mean_nbin, deriv1st_nbin);
        write_dist_from_act("netw1_outpop_gen_act.bin",  "netw1_outpop_gen_dist.bin",  mean_nbin, deriv1st_nbin);
    } else {
        // ---------------- Multi-block walk-forward path -----------------
        // One Logger per phase, kept alive for the whole run. The constructor
        // opens the file in "wb" mode, and dolog() only fires when ison is
        // true (see Logger.cpp). Toggling on()/off() at phase boundaries
        // therefore yields a single concatenated stream per phase across
        // the whole schedule.
        Logger *troutacts   = new Logger(outpop_md1, "ACT", "netw1_outpop_tr_act.bin");
        Logger *tetroutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_tetr_act.bin");
        Logger *vanoutacts  = new Logger(outpop_md1, "ACT", "netw1_outpop_van_act.bin");
        Logger *teusoutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_teus_act.bin");
        troutacts->off(); tetroutacts->off();
        vanoutacts->off(); teusoutacts->off();

        int cursor = 0;
        for (size_t b = 0; b < schedule.size(); b++) {
            int trnp = std::get<0>(schedule[b]);
            int vanp = std::get<1>(schedule[b]);
            int tenp = std::get<2>(schedule[b]);
            printf("BLOCK %zu/%zu  trnpat=%d vanpat=%d tenpat=%d  "
                   "(simstep=%d)\n",
                   b + 1, schedule.size(), trnp, vanp, tenp, simstep);

            // --- training (learning ON) then test-on-train (same patterns) ---
            outpop_md1->setigain(1.);
            ltslprj_md1->setbwgain(0.);
            if (warmup_pat > 0) {
                run_multislice_phase(trnp, trnpat_per, cursor, 1, troutacts, true);
            } else {
                troutacts->on();
                run_multislice_phase_flat(trnp, cursor, 1, nullptr);
                troutacts->off();
            }
            outpop_md1->setigain(0.);
            ltslprj_md1->setbwgain(1.);
            if (warmup_pat > 0)
                run_multislice_phase(trnp, trnpat_per, cursor, 0, tetroutacts, true);
            else
                run_multislice_phase_flat(trnp, cursor, 0, tetroutacts);
            cursor += trnp;

            // --- validation (learning OFF) ---
            if (warmup_pat > 0)
                run_multislice_phase(vanp, vanpat_per, cursor, 0, vanoutacts, true);
            else
                run_multislice_phase_flat(vanp, cursor, 0, vanoutacts);
            cursor += vanp;

            // --- test on unseen (learning OFF) ---
            if (warmup_pat > 0)
                run_multislice_phase(tenp, tenpat_per, cursor, 0, teusoutacts, true);
            else
                run_multislice_phase_flat(tenp, cursor, 0, teusoutacts);
            cursor += tenp;
        }

        troutacts->dosave();
        tetroutacts->dosave();
        vanoutacts->dosave();
        teusoutacts->dosave();

        write_dist_from_act("netw1_outpop_tr_act.bin",   "netw1_outpop_tr_dist.bin",   mean_nbin, deriv1st_nbin);
        write_dist_from_act("netw1_outpop_tetr_act.bin", "netw1_outpop_tetr_dist.bin", mean_nbin, deriv1st_nbin);
        write_dist_from_act("netw1_outpop_van_act.bin",  "netw1_outpop_van_dist.bin",  mean_nbin, deriv1st_nbin);
        write_dist_from_act("netw1_outpop_teus_act.bin", "netw1_outpop_teus_dist.bin", mean_nbin, deriv1st_nbin);

        // Generation after the scheduled train/val/test blocks (e.g. Mains.general()
        // with a one-row schedule). Skipped when ngenstep==0 (multi / multislice).
        outpop_md1->setigain(0.);
        outpop_md1->setagain(outagain_gen);
        Logger *genoutacts = new Logger(outpop_md1, "ACT", "netw1_outpop_gen_act.bin");
        if (ngenstep > 0) {
            printf("GENERATING from simstep = %d\n", simstep);
            for (int step = 0; step < ngenstep; step++) {
                ltslpop->setinput(outpop_md1->act);
                tdnetw->updstate(0);
                advance();
            }
        }
        genoutacts->off();
        genoutacts->dosave();
        write_dist_from_act("netw1_outpop_gen_act.bin",  "netw1_outpop_gen_dist.bin",  mean_nbin, deriv1st_nbin);
    }

    printf("Total simsteps = %d, time elapsed = %.3f sec\n",
           simstep, getDiffTime(total_time) / 1000);

    closelogs();

    return 0;
}
