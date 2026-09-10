/*****************************************************************

  Author: Anders Lansner

  Created: 2023-09-13     Modified: 2024-01-07
  Modified from: /home/ala/OurPrograms/BCPNNH/Pats.h

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
#include "PatternFactory.h"

using namespace std;
using namespace Globals;

//********** Static variables and methods **********

vector<PatternFactory *> PatternFactory::patfacs;
RndGen *PatternFactory::rndgen = nullptr;

void PatternFactory::clear() {
    patfacs.clear();
}

RndGen *PatternFactory::mkrndgen(long seed) {
    rndgen = new RndGen();
    rndgen->setseed(seed);
    return rndgen;
}

vector<float> PatternFactory::complpat(vector<float> pat, int Hx, int Mx) {
    int Nx = Hx * Mx;
    for (int i = 0; i < Nx; i++)
        pat[i] = 1 - pat[i];
    for (int h = 0; h < Hx; h++) {
        float hsum = 0;
        for (int m = h * Mx; m < (h + 1)*Mx; m++)
            hsum += pat[m];
        if (hsum != 0)
            for (int m = h * Mx; m < (h + 1)*Mx; m++)
                pat[m] /= hsum;
    }
    return pat;
}

vector<float> PatternFactory::hblank(vector<float> pat, int Hx, int Mx, float nhblank) {
    if (rndgen == nullptr) rndgen = RndGen::grndgen;
    int rnd01 = rndgen->nextfloat() < (nhblank - floor(nhblank)), ndis = rnd01 + floor(nhblank);
    vector<int> tmp(Hx, 0);
    for (int d = 0, h; d < ndis; d++) {
        h = rndgen->nextint() % Hx;
        while (tmp[h] != 0)
            h = rndgen->nextint() % Hx;
        tmp[h] = 1;
        for (int m = 0; m < Mx; m++)
            pat[h * Mx + m] = 0;
    }
    return pat;
}

vector<float> PatternFactory::hflip(vector<float> pat, int Hx, int Mx, float nhflip) {
    if (rndgen == nullptr) rndgen = RndGen::grndgen;
    int rnd01 = rndgen->nextfloat() < (nhflip - floor(nhflip)), ndis = rnd01 + floor(nhflip);
    vector<int> tmp(Hx, 0);
    for (int d = 0, h, m1, mn; d < ndis; d++) {
        h = rndgen->nextint() % Hx;
        while (tmp[h] != 0)
            h = rndgen->nextint() % Hx;
        tmp[h] = 1;
        if (Mx > 1) {
            m1 = argmax(pat, h * Mx, Mx) - h * Mx;
            for (int m = 0; m < Mx; m++)
                pat[h * Mx + m] = 0;
            mn = rndgen->nextint() % Mx;
            while (mn == m1)
                mn = rndgen->nextint() % Mx;
            pat[h * Mx + mn] = 1;
        } else
            pat[h] = 1;
    }
    return pat;
}

vector<float> PatternFactory::binarize(vector<float> pat) {
    vector<float> bxpat(2 * pat.size());
    for (size_t i = 0; i < pat.size(); i++) {
        bxpat[2 * i] = 1 - pat[i];
        bxpat[2 * i + 1] = pat[i];
    }
    return bxpat;
}

vector<float> PatternFactory::hnormalize(vector<float> pat, int Hx) {
    float hsum;
    int Mx = pat.size() / Hx;
    for (int h = 0; h < Hx; h++) {
        hsum = 0;
        for (int i = h * Mx; i < (h + 1) * Mx; i++)
            hsum += pat[i];
        if (hsum > 0)
            for (int i = h * Mx; i < (h + 1) * Mx; i++)
                pat[i] /= hsum;
    }
    return pat;
}

void PatternFactory::prpat(vector<float> pat, int ndec, int npos, string endl) {
    for (size_t i = 0; i < pat.size(); i++)
        if (npos < 0)
            fprintf(stderr, "%.*f ", ndec, pat[i]);
        else
            fprintf(stderr, "%*.*f ", npos, ndec, pat[i]);
    fprintf(stderr, "%s", endl.c_str());
}

void PatternFactory::prpats(vector<vector<float> > pats, int ndec, int npos, string endl) {
    for (size_t r = 0; r < pats.size(); r++)
        prpat(pats[r], ndec, npos, "\n");
    fprintf(stderr, "%s", endl.c_str());
}

void PatternFactory::prpat(float *pat, int n, int ndec, int npos, string endl) {
    for (size_t i = 0; (int)i < n; i++)
        if (npos < 0)
            fprintf(stderr, "%.*f ", ndec, pat[i]);
        else
            fprintf(stderr, "%*.*f ", npos, ndec, pat[i]);
    fprintf(stderr, "%s", endl.c_str());
}

void PatternFactory::prpats(float *pats, int n, int npat, int ndec, int npos, string endl) {
    for (int p = 0; p < npat; p++) {
        for (size_t i = 0; (int)i < n; i++)
            if (npos < 0)
                fprintf(stderr, "%.*f ", ndec, pats[p * n + i]);
            else
                fprintf(stderr, "%*.*f ", npos, ndec, pats[p * n + i]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

//********** Class variables and methods ***********

PatternFactory::PatternFactory(int patlen, int Hx, string patype) {
    if (patlen % Hx != 0)
        error("PatternFactory::PatternFactory", "patlen%Hx!=0");
    this->Nx = patlen;
    this->Hx = Hx;
    Mx = Nx / Hx;
    Nx = Hx * Mx;
    filename = "";
    if (patype == "hrand")
        this->patype = HRAND;
    else if (patype == "ortho")
        this->patype = ORTHO;
    else
        error("PatternFactory::PatternFactory", "Unknown patype");
    id = patfacs.size();
    binarizing = false;
    patfacs.push_back(this);
}

PatternFactory::PatternFactory(PatternFactory *cpatfac, bool complm) {
    Hx = cpatfac->Hx;
    Mx = cpatfac->Mx;
    Nx = cpatfac->Nx;
    patype = cpatfac->patype;
    filename = cpatfac->filename;;
    binarizing = cpatfac->binarizing;
    if (cpatfac->getnpat() > 0) {
        for (size_t p = 0; p < cpatfac->pats.size(); p++) {
            if (not complm) {
                pats.push_back(cpatfac->pats[p]);
                if (bxpats.size() > 0)
                    bxpats.push_back(cpatfac->pats[p]);
            } else {
                pats.push_back(complpat(cpatfac->pats[p], Hx, Mx));
                if (bxpats.size() > 0)
                    bxpats.push_back(binarize(complpat(cpatfac->pats[p], Hx, Mx)));
            }
        }
    }
    if (cpatfac->getninst() > 0) {
        for (int p = 0; p < (int)cpatfac->insts.size(); p++) {
            if (not complm) {
                insts.push_back(cpatfac->insts[p]);
                if (bxinsts.size() > 0)
                    bxinsts.push_back(cpatfac->insts[p]);
            } else {
                insts.push_back(complpat(cpatfac->insts[p], Hx, Mx));
                if (bxinsts.size() > 0)
                    bxinsts.push_back(binarize(complpat(cpatfac->insts[p], Hx, Mx)));
            }
        }
    }
    id = patfacs.size();
    patfacs.push_back(this);
}

void PatternFactory::mkpats(int npat) {
    if (rndgen == nullptr) rndgen = RndGen::grndgen;
    if (Hx == Nx)
        error("PatternFactory::mkpats", "Illegal: Hx == Nx (Mx == 1)");
    pats = vector<vector<float> > (npat, vector<float> (Nx, 0));
    uint un;
    for (int p = 0; p < npat; p++) {
        for (int h = 0; h < Hx; h++) {
            if (patype == ORTHO)
                pats[p][h * Mx + p % Mx] = 1;
            else if (patype == HRAND) {
                un = rndgen->nextint();
                un = un % Mx;
                pats[p][h * Mx + un] = 1;
            } else
                error("PatternFactory::mkpats", "No such patype: " + patype);
        }
    }
}

void PatternFactory::readpats(string filename, int npat) {
    this->filename = filename;
    pats = Globals::readpats(Nx, filename, npat);
}

void PatternFactory::mkinsts(int ninst) {
    int npat = pats.size();
    insts = vector<vector<float> >(npat * ninst, vector<float>(Nx, 0));
    this->ninst = ninst;
    for (int p = 0; p < npat; p++)
        for (int i = 0; i < ninst; i++)
            insts[p * ninst + i] = pats[p];
}

vector<vector<float> > PatternFactory::binarizepats(vector<vector<float> > pats) {
    bxpats = vector<vector<float> >(pats.size());
    if (pats.size() > 0)
        for (size_t p = 0; p < pats.size(); p++)
            bxpats[p] = binarize(pats[p]);
    binarizing = true;
    return bxpats;
}

int PatternFactory::getnpat() {
    return pats.size();
}

int PatternFactory::getninst() {
    return insts.size();
}

vector<float> PatternFactory::getpat(int p) {
    if (p < 0)
        error("PatternFactory::getpat", "Illegal p = " + to_string(p));
    if ((int)pats.size() <= p)
        return vector<float>(0);
    if (binarizing)
        return bxpats[p];
    return pats[p];
}

vector<vector<float> > PatternFactory::getpats(int p0, int npat) {
    if (p0 < 0 or (int)pats.size() <= p0 + npat)
        error("PatternFactory::getpats", "Illegal range");
    if (binarizing)
        return { bxpats.begin() + p0, bxpats.begin() + p0 + npat };
    return { pats.begin() + p0, pats.begin() + p0 + npat };
}

vector<float> PatternFactory::getinst(int p) {
    if (p < 0)
        error("PatternFactory::getinst", "Illegal p = " + to_string(p));
    if ((int)insts.size() <= p)
        return vector<float>(0);
    if (binarizing)
        return bxinsts[p];
    return insts[p];
}

vector<vector<float> > PatternFactory::getinsts(int p0, int npat) {
    if (p0 < 0 or (int)insts.size() <= p0 + npat)
        error("PatternFactory::getinsts", "Illegal range");
    if (binarizing)
        return { bxinsts.begin() + p0, bxinsts.begin() + p0 + npat };
    return { insts.begin() + p0, insts.begin() + p0 + npat };
}

float *PatternFactory::getfpat(int p) {
    if (pats.size() < p)
        return nullptr;
    if (binarizing)
        return bxpats[p].data();
    return pats[p].data();
}

float *PatternFactory::getfinst(int p) {
    if ((int)insts.size() < p) {
        error("PatternFactory::getfinst", "insts.size() < p");
        return nullptr;
    }
    if (binarizing)
        return bxinsts[p].data();
    return insts[p].data();
}

void PatternFactory::writepats(std::string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    for (int p = 0; p < getnpat(); p++)
        if (binarizing)
            fwrite(bxpats[p].data(), sizeof(float), 2 * Nx, outfp);
        else
            fwrite(pats[p].data(), sizeof(float), Nx, outfp);
    fclose(outfp);
}

void PatternFactory::writeinsts(std::string filename) {
    FILE *outfp = fopen(filename.c_str(), "wb");
    for (int p = 0; p < getninst(); p++)
        if (binarizing)
            fwrite(bxinsts[p].data(), sizeof(float), 2 * Nx, outfp);
        else
            fwrite(insts[p].data(), sizeof(float), Nx, outfp);
    fclose(outfp);
}

void PatternFactory::hblankpats(float nhblank) {
    for (size_t p = 0; p < pats.size(); p++)
        pats[p] = hblank(pats[p], Hx, Mx, nhblank);
    if (binarizing)
        bxpats = binarizepats(pats);
}

void PatternFactory::hflippats(float nhflip) {
    for (size_t p = 0; p < pats.size(); p++)
        pats[p] = hflip(pats[p], Hx, Mx, nhflip);
    if (binarizing)
        bxpats = binarizepats(pats);
}

void PatternFactory::hblankinsts(float nhblank) {
    for (size_t p = 0; p < insts.size(); p++)
        insts[p] = hblank(insts[p], Hx, Mx, nhblank);
    if (binarizing)
        bxinsts = binarizepats(insts);
}

void PatternFactory::hflipinsts(float nhflip) {
    for (size_t p = 0; p < insts.size(); p++)
        insts[p] = hflip(insts[p], Hx, Mx, nhflip);
    if (binarizing)
        bxinsts = binarizepats(insts);
}
