#include "gym.h"
#include "model/model.h"
#include <cstdio>

struct CartPole {
    struct State {
        float x;
        float xDot;
        float theta;
        float thetaDot;
        uint32_t steps;
    };
    using Action = int;
    struct Obs { float x, xDot, theta, thetaDot; };
    static constexpr uint32_t OBS_DIM = 4;
    static constexpr uint32_t ACT_DIM = 2;
    static constexpr const char* KERNEL = "rollout_cartpole";
};

int main() {
    const uint32_t N = 4096; //environments
    const uint32_t T = 32;   //steps per rollout

    Gym<CartPole> gym(N);
    Model<CartPole> model(gym, 128, 4, 2, 3e-4f, T);

    for (int it = 0; it < 40; it++) {
        model.collect();
        model.train();
        if (it % 10 == 0) {
            LossStats s = model.lastStats();
            std::printf("iter %3d  entropy %.3f  kl %.4f\n", it, s.entropy, s.approxKl);
        }
    }
    return 0;
}
