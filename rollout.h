#include <metal_stdlib>

using namespace metal;

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
