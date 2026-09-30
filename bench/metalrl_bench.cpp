#include "gym.h"
#include "model/ppo.h"
#include <chrono>
#include <cstdio>
struct CartPole {
    struct State { float x, xDot, theta, thetaDot; uint32_t steps; };
    using Action = int;
    struct Obs { float x, xDot, theta, thetaDot; };
    static constexpr uint32_t OBS_DIM = 4;
    static constexpr uint32_t ACT_DIM = 2;
    static constexpr const char* KERNEL = "rollout_cartpole";
};
using clk = std::chrono::steady_clock;
static double sec(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
int main(int argc, char** argv) {
    const uint32_t N = argc > 1 ? std::atoi(argv[1]) : 4096, T = 32; const int L = argc > 2 ? std::atoi(argv[2]) : 2;
    const int ITERS = 20;
    Gym<CartPole> gym(N);
    PPOModel<CartPole> model(gym, 128, 4, L, 3e-4f, T);
    std::vector<float> reward(N);
    //warm-up
    model.collect(); model.train();
    //env only: gym.run in a loop with the actions left in the buffer
    auto t0 = clk::now();
    for (int i = 0; i < 1000; i++) gym.run(reward);
    double envOnly = sec(t0, clk::now());
    //collect only
    double collectT = 0, trainT = 0;
    for (int it = 0; it < ITERS; it++) {
        auto a = clk::now(); model.collect(); auto b = clk::now(); model.train(); auto c = clk::now();
        collectT += sec(a, b); trainT += sec(b, c);
    }
    double steps = (double)ITERS * N * T;
    std::printf("N=%u T=%u layers=%d\n", N, T, L);
    std::printf("env step only (gym.run loop):      %10.0f steps/s\n", 1000.0 * N / envOnly);
    std::printf("collect (policy + env + GAE):      %10.0f steps/s\n", steps / collectT);
    std::printf("collect + train (4 epochs):        %10.0f steps/s\n", steps / (collectT + trainT));
    std::printf("per iteration: collect %.3fs  train %.3fs\n", collectT / ITERS, trainT / ITERS);
}
