#include <metal_stdlib>
#include "../../rollout.h"
using namespace metal;

//Goal-conditioned navigation over generated terrain, after minetrix's bridge
//band. A 32 x 32 heightmap with one chasm; spawn on one side, goal on the
//other; the agent walks, jumps, or places cobble to cross. Observation is a
//depth camera of 8 x 4 rays plus the goal in the agent's frame.
struct VoxelNav {
    constant static constexpr uint W = 32;
    constant static constexpr uint RAY_COLS = 8;
    constant static constexpr uint RAY_ROWS = 4;
    constant static constexpr uint N_RAYS = RAY_COLS * RAY_ROWS;
    constant static constexpr uint OBS_DIM = N_RAYS + 8;
    constant static constexpr uint ACT_DIM = 6;
    constant static constexpr uint MAX_STEPS = 250;
    constant static constexpr float FOV = 90.0f * M_PI_F / 180.0f;
    constant static constexpr float MAX_DEPTH = 24.0f;
    constant static constexpr float EYE = 1.6f;
    constant static constexpr float GOAL_RADIUS = 1.5f;
    constant static constexpr uint START_COBBLE = 16;

    struct State {
        float x, z, yaw;
        float gx, gz;
        float prevDist;
        uint steps;
        uint cobble;
        uchar height[W * W]; //terrain height per cell, 0 = void
    };
    using Action = int; //0 forward, 1 back, 2 turn left, 3 turn right, 4 jump forward, 5 place cobble
    struct Obs {
        float ray[N_RAYS];
        float goalForward, goalRight, goalUp, goalDist;
        float sinYaw, cosYaw, cobble, height;
    };

    static uint cell(int cx, int cz) { return uint(cx) * W + uint(cz); }
    static bool inside(int cx, int cz) { return cx >= 0 && cz >= 0 && cx < int(W) && cz < int(W); }
    static float heightAt(thread const State& s, float x, float z) {
        int cx = int(floor(x)), cz = int(floor(z));
        return inside(cx, cz) ? float(s.height[cell(cx, cz)]) : 0.0f;
    }
    static float dist(thread const State& s) {
        float dx = s.gx - s.x, dz = s.gz - s.z;
        return sqrt(dx * dx + dz * dz);
    }

    static void reset(thread State& s, thread uint& rng) {
        //rolling hills: 5 x 5 random knots, bilinear between them
        float knot[25];
        for (uint i = 0; i < 25; i++) knot[i] = rngUniform(rng, 3.0f, 9.0f);
        for (uint cx = 0; cx < W; cx++) {
            float fx = float(cx) / float(W - 1) * 4.0f;
            uint ix = min(uint(fx), 3u); float tx = fx - float(ix);
            for (uint cz = 0; cz < W; cz++) {
                float fz = float(cz) / float(W - 1) * 4.0f;
                uint iz = min(uint(fz), 3u); float tz = fz - float(iz);
                float h = mix(mix(knot[ix * 5 + iz], knot[(ix + 1) * 5 + iz], tx),
                              mix(knot[ix * 5 + iz + 1], knot[(ix + 1) * 5 + iz + 1], tx), tz);
                s.height[cell(cx, cz)] = uchar(round(h));
            }
        }
        //one chasm along z, 2 to 5 cells wide; the far side may be lifted
        uint width = 2 + rngInt(rng, 4);
        uint left = 12 + rngInt(rng, 6);
        uint lift = rngInt(rng, 3);
        for (uint cx = 0; cx < W; cx++) {
            for (uint cz = 0; cz < W; cz++) {
                uint k = cell(cx, cz);
                if (cx >= left && cx < left + width) s.height[k] = 0;
                else if (cx >= left + width) s.height[k] = uchar(min(uint(s.height[k]) + lift, 15u));
            }
        }
        s.x = 2.5f + rngUniform(rng) * float(left - 4);
        s.z = 2.5f + rngUniform(rng) * float(W - 5);
        s.gx = float(left + width) + 2.5f + rngUniform(rng) * float(W - 5 - (left + width));
        s.gz = 2.5f + rngUniform(rng) * float(W - 5);
        s.yaw = rngUniform(rng, -M_PI_F, M_PI_F);
        s.prevDist = dist(s);
        s.steps = 0;
        s.cobble = START_COBBLE;
    }

    static void observe(thread const State& s, thread Obs& o) {
        const float pitches[RAY_ROWS] = {-45.0f, -20.0f, 0.0f, 15.0f};
        float h = heightAt(s, s.x, s.z);
        float ey = h + EYE;
        for (uint c = 0; c < RAY_COLS; c++) {
            float a = s.yaw + (float(c) / float(RAY_COLS - 1) - 0.5f) * FOV;
            for (uint r = 0; r < RAY_ROWS; r++) {
                float p = pitches[r] * M_PI_F / 180.0f;
                float dx = cos(p) * cos(a), dy = sin(p), dz = cos(p) * sin(a);
                float depth = MAX_DEPTH;
                for (float t = 0.5f; t < MAX_DEPTH; t += 0.5f) {
                    float px = s.x + t * dx, py = ey + t * dy, pz = s.z + t * dz;
                    int cx = int(floor(px)), cz = int(floor(pz));
                    if (!inside(cx, cz)) { depth = t; break; }
                    if (py <= float(s.height[cell(cx, cz)])) { depth = t; break; }
                }
                o.ray[c * RAY_ROWS + r] = depth / MAX_DEPTH;
            }
        }
        float gx = s.gx - s.x, gz = s.gz - s.z;
        float ca = cos(s.yaw), sa = sin(s.yaw);
        o.goalForward = (gx * ca + gz * sa) / float(W);
        o.goalRight = (-gx * sa + gz * ca) / float(W);
        o.goalUp = (heightAt(s, s.gx, s.gz) - h) / 8.0f;
        o.goalDist = sqrt(gx * gx + gz * gz) / float(W);
        o.sinYaw = sa;
        o.cosYaw = ca;
        o.cobble = float(s.cobble) / float(START_COBBLE);
        o.height = h / 16.0f;
    }

    //move to (nx, nz) if the target cell is solid and at most one block up.
    //Returns 1 if the move ended in the void.
    static bool tryMove(thread State& s, float nx, float nz) {
        int cx = int(floor(nx)), cz = int(floor(nz));
        if (!inside(cx, cz)) return false;
        float here = heightAt(s, s.x, s.z);
        float there = float(s.height[cell(cx, cz)]);
        if (there == 0.0f) { s.x = nx; s.z = nz; return true; }
        if (there > here + 1.0f) return false;
        s.x = nx; s.z = nz;
        return false;
    }

    static void step(thread State& s, Action action, thread Obs& obs,
                     thread float& reward, thread uchar& done) {
        float ca = cos(s.yaw), sa = sin(s.yaw);
        bool fell = false;
        switch (action) {
            case 0: fell = tryMove(s, s.x + ca, s.z + sa); break;
            case 1: fell = tryMove(s, s.x - ca, s.z - sa); break;
            case 2: s.yaw -= M_PI_F / 6.0f; break;
            case 3: s.yaw += M_PI_F / 6.0f; break;
            case 4: fell = tryMove(s, s.x + 2.0f * ca, s.z + 2.0f * sa); break;
            case 5: {
                int cx = int(floor(s.x + ca)), cz = int(floor(s.z + sa));
                if (s.cobble > 0 && inside(cx, cz)) {
                    uint k = cell(cx, cz);
                    uint here = uint(heightAt(s, s.x, s.z));
                    if (s.height[k] == 0) { s.height[k] = uchar(here); s.cobble--; }
                    else if (uint(s.height[k]) <= here) { s.height[k]++; s.cobble--; }
                }
                break;
            }
        }
        if (s.yaw > M_PI_F) s.yaw -= 2.0f * M_PI_F;
        if (s.yaw < -M_PI_F) s.yaw += 2.0f * M_PI_F;
        s.steps++;

        float d = dist(s);
        reward = -0.05f + 0.2f * clamp(s.prevDist - d, -1.5f, 1.5f);
        s.prevDist = d;
        bool success = d < GOAL_RADIUS;
        if (success) reward += 10.0f;
        if (fell) reward -= 5.0f;
        done = success || fell || s.steps >= MAX_STEPS;
        observe(s, obs);
    }
};

template [[host_name("rollout_voxelnav")]]
kernel void rollout<VoxelNav>(
    device VoxelNav::State*,
    device const VoxelNav::Action*,
    device VoxelNav::Obs*,
    device float*,
    device uchar*,
    device uint*,
    uint
);
