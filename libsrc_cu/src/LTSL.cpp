/*****************************************************************

  Created: 2025-11-11

  Author: Anders Lansner

  Copyright (c) 2025 Anders Lansner

******************************************************************/

#include "LTSL.h"
#include <cmath>
#include <algorithm>

LTSL::LTSL(float fs, float taumin, float taumax, int K, int K_mode)
    : fs(fs), tau0(taumin), taumax(taumax), c(0.0f), K(K), K_mode(K_mode) {
    if (fs <= 0.0f) throw std::invalid_argument("fs must be > 0");
    if (K <= 0) throw std::invalid_argument("K must be > 0");
    allocate_();
    compute_coeffs_();
    reset();
}

LTSL::LTSL(float fs, float taumin, float taumax, int K, int K_mode, bool initialize)
    : fs(fs), tau0(taumin), taumax(taumax), c(0.0f), K(K), K_mode(K_mode) {
    if (fs <= 0.0f) throw std::invalid_argument("fs must be > 0");
    if (K <= 0) throw std::invalid_argument("K must be > 0");
    allocate_();
    if (initialize) {
        compute_coeffs_();
        reset();
    }
}

LTSL::~LTSL() { deallocate_(); }

LTSL::LTSL(const LTSL& other)
    : fs(other.fs), tau0(other.tau0), taumax(other.taumax), c(other.c), K(other.K), K_mode(other.K_mode) {
    allocate_();
    std::copy(other.taus_, other.taus_ + K, taus_);
    std::copy(other.a_, other.a_ + K, a_);
    std::copy(other.b_, other.b_ + K, b_);
    std::copy(other.s_, other.s_ + K, s_);
    std::copy(other.y_, other.y_ + K, y_);
}

LTSL& LTSL::operator=(const LTSL& other) {
    if (this == &other) return *this;
    if (K != other.K) {
        deallocate_();
        fs = other.fs;
        tau0 = other.tau0;
        taumax = other.taumax;
        c = other.c;
        K = other.K;
        K_mode = other.K_mode;
        allocate_();
    } else {
        fs = other.fs;
        tau0 = other.tau0;
        taumax = other.taumax;
        c = other.c;
        K_mode = other.K_mode;
    }
    std::copy(other.taus_, other.taus_ + K, taus_);
    std::copy(other.a_, other.a_ + K, a_);
    std::copy(other.b_, other.b_ + K, b_);
    std::copy(other.s_, other.s_ + K, s_);
    std::copy(other.y_, other.y_ + K, y_);
    return *this;
}

LTSL::LTSL(LTSL&& other) noexcept
    : fs(other.fs), tau0(other.tau0), taumax(other.taumax), c(other.c), K(other.K), K_mode(other.K_mode),
      taus_(other.taus_), a_(other.a_), b_(other.b_), s_(other.s_), y_(other.y_) {
    other.taus_ = other.a_ = other.b_ = other.s_ = other.y_ = nullptr;
    other.K = 0;
}

LTSL& LTSL::operator=(LTSL&& other) noexcept {
    if (this == &other) return *this;
    deallocate_();
    fs = other.fs;
    tau0 = other.tau0;
    taumax = other.taumax;
    c = other.c;
    K = other.K;
    K_mode = other.K_mode;
    taus_ = other.taus_; a_ = other.a_; b_ = other.b_; s_ = other.s_; y_ = other.y_;
    other.taus_ = other.a_ = other.b_ = other.s_ = other.y_ = nullptr;
    other.K = 0;
    return *this;
}

ulong LTSL::nallocbyte() {
    return 5 * K * sizeof(float);
}

void LTSL::allocate_() {
    taus_ = new float[K];
    a_    = new float[K];
    b_    = new float[K];
    s_    = new float[K];
    y_    = new float[K];
}

void LTSL::deallocate_() {
    delete[] taus_; taus_ = nullptr;
    delete[] a_;    a_    = nullptr;
    delete[] b_;    b_    = nullptr;
    delete[] s_;    s_    = nullptr;
    delete[] y_;    y_    = nullptr;
}

void LTSL::reset() {
    for (int k = 0; k < K; ++k) { s_[k] = 0.0f; y_[k] = 0.0f; }
}

float* LTSL::step(float x_n) {
    // Stage 0
    s_[0] = a_[0] * s_[0] + b_[0] * x_n;
    y_[0] = s_[0];
    // Remaining stages
    for (int k = K-1; k > 0; k--) {
        s_[k] = a_[k] * s_[k] + b_[k] * y_[k - 1];
        y_[k] = s_[k];
    }
    return y_;
}

void LTSL::run(const float* X, std::size_t N, float* Y_row_major) {
    // reset();
    for (std::size_t n = 0; n < N; ++n) {
        const float* yk = step(X[n]);
        // copy y_ into output row n
        float* row = Y_row_major + static_cast<std::ptrdiff_t>(n) * K;
        for (int k = 0; k < K; ++k) {
            row[k] = yk[k];
        }
    }
}

const float* LTSL::taus() const { return taus_; }
const float* LTSL::a() const { return a_; }
const float* LTSL::b() const { return b_; }

void LTSL::compute_coeffs_() {
    const float dt = 1.0f / fs;
    if (K_mode == 0) {
        // Logarithmic (geometric) spacing
        c = std::pow(taumax / tau0, 1.0f / (K - 1.0f));
        float tau_k = tau0;
        for (int k = 0; k < K; ++k) {
            taus_[k] = tau_k;
            a_[k] = static_cast<float>(std::exp(-dt / tau_k));
            b_[k] = 1.0f - a_[k];
            tau_k *= c;
        }
    } else {
        // Linear spacing
        float dtau = (taumax - tau0) / (K - 1.0f);
        float tau_k = tau0;
        for (int k = 0; k < K; ++k) {
            taus_[k] = tau_k;
            a_[k] = static_cast<float>(std::exp(-dt / tau_k));
            b_[k] = 1.0f - a_[k];
            tau_k += dtau;
        }
    }
}
