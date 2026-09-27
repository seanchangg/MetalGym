#include <metal_stdlib>

#pragma once
using namespace metal;

//Per-env random numbers. The rng buffer holds one uint per env, seeded
//nonzero by the host. Environments call these from reset() and step().
//xorshift32
inline uint rngNext(thread uint& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

//uniform float in [0, 1)
inline float rngUniform(thread uint& s) {
    return float(rngNext(s) >> 8) * (1.0f / 16777216.0f);
}

//uniform float in [lo, hi)
inline float rngUniform(thread uint& s, float lo, float hi) {
    return lo + (hi - lo) * rngUniform(s);
}

//uniform int in [0, n)
inline uint rngInt(thread uint& s, uint n) {
    return rngNext(s) % n;
}

template <typename Env>
kernel void rollout(
	device typename Env::State* states [[buffer(0)]],
	device const int* actions [[buffer(1)]],
	device float* obs [[buffer(2)]],
	device float* reward [[buffer(3)]],
	device uchar* done [[buffer(4)]],
	device uint* rng [[buffer(5)]],
	uint tid [[thread_position_in_grid]]
) {
	typename Env::State s = states[tid];
	uint rg = rng[tid];
	float r = reward[tid];
	uchar d = done[tid];
	if (done[tid]) Env::reset(s, rg);
	Env::step(s, actions[tid], obs + tid*Env::OBS_DIM, r, d);
	states[tid] = s;
	reward[tid] = r;
	rng[tid] = rg;
	done[tid] = d;
}
