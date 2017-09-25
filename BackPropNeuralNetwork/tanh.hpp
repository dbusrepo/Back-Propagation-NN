#pragma once

#include "common.h"
#include <cmath>

/** interpolation boundaries, precision and interpolation step */
#ifdef USE_FLOAT
static const RealType tanh_xmax = 12.5f * logf(2);
static const RealType tanh_xmin = -tanh_xmax;
static const RealType tanh_err = 1e-8f;
static const RealType tanh_step = sqrtf(tanh_err * 6 * sqrtf(3.));
static const RealType tanh_istep = 1.f / tanh_step;
#else
static const RealType tanh_xmax = 12.5 * log(2);
static const RealType tanh_xmin = -tanh_xmax;
static const RealType tanh_err = 1e-10;
static const RealType tanh_step = sqrt(tanh_err * 6 * sqrt(3.));
static const RealType tanh_istep = 1. / tanh_step;
#endif


/** interpolation table size */
static const size_t tanh_size = (size_t)((tanh_xmax - tanh_xmin) / tanh_step) + 2;
static RealType * tanh_table = nullptr;

/**
* table initialization
*/
void tanh_inter_init()
{
	if (nullptr == tanh_table) {
		tanh_table = new RealType[2 * tanh_size];
		// tanh() samples
		for (size_t i = 0; i<tanh_size; i++)
#ifdef USE_FLOAT
			tanh_table[2 * i] = tanhf(tanh_xmin + i * tanh_step);
#else
			tanh_table[2 * i] = tanh(tanh_xmin + i * tanh_step);
#endif
		// precomputed sample differences
		// results are interlaced to minimize cache miss
		for (size_t i = 0; i<tanh_size - 1; i++)
			tanh_table[2 * i + 1] = tanh_table[2 * i + 2] - tanh_table[2 * i];
	}
}

/**
* interpolated tanh() approximation
*
*/
inline RealType tanh_inter(RealType x)
{
	// position of x in the table
	RealType fpos = (x - tanh_xmin) * tanh_istep;

	// position of x in the interpolation table
	size_t ipos = static_cast<size_t>(fpos);

	// t in [0..1] is the continuous position of x
	// between step ipos and step ipos+1
	RealType t = fpos - static_cast<RealType>(ipos);

#ifdef USE_FLOAT
	if (fabsf(x) > tanh_xmax)
#else
	if (abs(x) > tanh_xmax)
#endif
		// out of bounds, tanh(x) = +-1 at this precision
		return (x > 0.f ? 1.f : -1.f);
	else
		// table[2*ipos] is the i-th sample
		// table[2*ipos+1] is the difference between i-th sample and the next
		return tanh_table[2 * ipos] + t * tanh_table[2 * ipos + 1];
}

/**
* tanh() from libc
*/
inline RealType tanh_libc(RealType x)
{
#ifdef USE_FLOAT
	return tanhf(x);
#else
	return tanh(x);
#endif
}
