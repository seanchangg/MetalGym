//Continuous-action CartPole. Env::Action is a struct of ACT_DIM floats, so
//Model selects the Gaussian head and learns a shared log_std.
#include "gym.h"
#include "model/model.h"
#include <cstdio>

struct CartPoleContinuous {
    struct State {
        float x;
        float xDot;
        float theta;
        float thetaDot;
        uint32_t steps;
    };
    struct Action { float force; };
    struct Obs { float x, xDot, theta, thetaDot; };
    static constexpr uint32_t OBS_DIM = 4;
    static constexpr uint32_t ACT_DIM = 1;
    static constexpr const char* KERNEL = "rollout_cartpole_continuous";
};

int main() {
    const uint32_t N = 4096;
    const uint32_t T = 32;

    Gym<CartPoleContinuous> gym(N);
    Model<CartPoleContinuous> model(gym, 128, 4, 2, 3e-4f, T);

    for (int it = 0; it < 60; it++) {
        model.collect();
        model.train();
        if (it % 10 == 0 || it == 59) {
            //rollout steps per finished episode, a quick proxy for the episode length
            uint32_t ended = 0;
            for (uint32_t t = 0; t < T; t++) {
                const uint8_t* done = model.slotDones(t);
                for (uint32_t i = 0; i < N; i++) ended += done[i];
            }
            const float meanLen = ended ? (float)(N * T) / (float)ended : (float)(N * T);
            LossStats s = model.lastStats();
            std::printf("iter %3d  steps/done %6.1f  std %.3f  entropy %.3f  kl %.4f\n",
                        it, meanLen, std::exp(model.logStdValue(0)), s.entropy, s.approxKl);
        }
    }
    return 0;
}
