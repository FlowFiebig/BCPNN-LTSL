/*****************************************************************

  Created: 2025-11-11

  Author: Anders Lansner

  Copyright (c) 2025 Anders Lansner

******************************************************************/
#pragma once
#include <cstddef>
#include <stdexcept>

class LTSL {
public:
    explicit LTSL(float fs, float taumin, float taumax, int K, int K_mode = 0);
    ~LTSL();


    // Rule of five: ensure deep copies of internal arrays

public:
    LTSL(float fs, float taumin, float taumax, int K, int K_mode, bool initialize);
    LTSL(const LTSL& other);
    LTSL& operator=(const LTSL& other);
    LTSL(LTSL&& other) noexcept;
    LTSL& operator=(LTSL&& other) noexcept;

    void reset();

    // One explicit time step. Returns pointer to y_[0], length K.
    float* step(float x_n);

    // return the allocated number of bytes
    ulong nallocbyte();

    // Run over N inputs in X. Outputs Y in row-major [n*K + k].
    void run(const float* X, std::size_t N, float* Y_row_major);

    // Accessors (do not expose std::vector)
    const float* taus() const; // length K
    const float* a() const; // length K
    const float* b() const; // length K

public:
    float fs, tau0, taumax, c;
    int K;
    int K_mode; // 0 = logarithmic, 1 = linear
private:
    float* taus_ = nullptr;
    float* a_ = nullptr;
    float* b_ = nullptr;
    float* s_ = nullptr;
    float* y_ = nullptr;

    void allocate_();
    void deallocate_();
    void compute_coeffs_();
};
