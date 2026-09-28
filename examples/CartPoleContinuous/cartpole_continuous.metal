//CartPole with a continuous force. The policy outputs one float; the env
//clamps it to [-1, 1] and scales it to the force magnitude.
#include <metal_stdlib>
#include "rollout.h"
using namespace metal;

struct CartPoleContinuous {
    struct State {
        float x;
        float xDot;
        float theta;
        float thetaDot;
        uint steps;
    };
    struct Action {
        float force; //Gaussian sample, unbounded; clamped in step()
    };
    struct Obs {
        float x;
        float xDot;
        float theta;
        float thetaDot;
    };

    constant static constexpr uint OBS_DIM = 4;
    constant static constexpr uint ACT_DIM = 1;

    constant static constexpr float GRAVITY = 9.8f;
    constant static constexpr float MASS_CART = 1.0f;
    constant static constexpr float MASS_POLE = 0.1f;
    constant static constexpr float TOTAL_MASS = MASS_CART + MASS_POLE;
    constant static constexpr float LENGTH = 0.5f;
    constant static constexpr float POLE_MASS_LENGTH = MASS_POLE * LENGTH;
    constant static constexpr float FORCE_MAG = 10.0f;
    constant static constexpr float TAU = 0.02f;
    constant static constexpr float THETA_THRESHOLD = 12.0f * 2.0f * M_PI_F / 360.0f;
    constant static constexpr float X_THRESHOLD = 2.4f;
    constant static constexpr uint MAX_STEPS = 500;

    static void reset(thread State& s, thread uint& rng) {
        s.x        = rngUniform(rng, -0.05f, 0.05f);
        s.xDot     = rngUniform(rng, -0.05f, 0.05f);
        s.theta    = rngUniform(rng, -0.05f, 0.05f);
        s.thetaDot = rngUniform(rng, -0.05f, 0.05f);
        s.steps    = 0;
    }

    static void observe(thread const State& s, thread Obs& obs) {
        obs.x        = s.x;
        obs.xDot     = s.xDot;
        obs.theta    = s.theta;
        obs.thetaDot = s.thetaDot;
    }

    static void step(thread State& s, Action action, thread Obs& obs,
                     thread float& reward, thread uchar& done) {
        float force = clamp(action.force, -1.0f, 1.0f) * FORCE_MAG;
        float cosTheta = cos(s.theta);
        float sinTheta = sin(s.theta);

        float temp = (force + POLE_MASS_LENGTH * s.thetaDot * s.thetaDot * sinTheta) / TOTAL_MASS;
        float thetaAcc = (GRAVITY * sinTheta - cosTheta * temp)
                       / (LENGTH * (4.0f / 3.0f - MASS_POLE * cosTheta * cosTheta / TOTAL_MASS));
        float xAcc = temp - POLE_MASS_LENGTH * thetaAcc * cosTheta / TOTAL_MASS;

        s.x        += TAU * s.xDot;
        s.xDot     += TAU * xAcc;
        s.theta    += TAU * s.thetaDot;
        s.thetaDot += TAU * thetaAcc;
        s.steps    += 1;

        bool terminated = fabs(s.x) > X_THRESHOLD || fabs(s.theta) > THETA_THRESHOLD;
        bool truncated  = s.steps >= MAX_STEPS;

        observe(s, obs);
        reward = 1.0f;
        done = terminated || truncated;
    }
};

template [[host_name("rollout_cartpole_continuous")]]
kernel void rollout<CartPoleContinuous>(
	device CartPoleContinuous::State*,
	device const CartPoleContinuous::Action*,
	device CartPoleContinuous::Obs*,
	device float*,
	device uchar*,
	device uint*,
	uint
);
