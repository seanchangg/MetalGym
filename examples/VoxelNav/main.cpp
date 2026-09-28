#include "gym.h"
#include "model/model.h"
#include <cstdio>

struct VoxelNav {
    static constexpr uint32_t W = 32;
    static constexpr uint32_t N_RAYS = 32;
    struct State {
        float x, z, yaw;
        float gx, gz;
        float prevDist;
        uint32_t steps;
        uint32_t cobble;
        uint8_t height[W * W];
    };
    using Action = int;
    struct Obs {
        float ray[N_RAYS];
        float goalForward, goalRight, goalUp, goalDist;
        float sinYaw, cosYaw, cobble, height;
    };
    static constexpr uint32_t OBS_DIM = N_RAYS + 8;
    static constexpr uint32_t ACT_DIM = 6;
    static constexpr const char* KERNEL = "rollout_voxelnav";
};

int main() {
    const uint32_t N = 4096;
    const uint32_t T = 32;

    Gym<VoxelNav> gym(N);
    Model<VoxelNav> model(gym, 128, 4, 2, 3e-4f, T);

    for (int it = 0; it < 200; it++) {
        model.collect();
        model.train();
        if (it % 10 == 0) {
            //an episode that ends with reward above 5 reached the goal
            uint64_t episodes = 0, successes = 0;
            for (uint32_t t = 0; t < T; t++) {
                const uint8_t* done = model.slotDones(t);
                const float* reward = model.slotRewards(t);
                for (uint32_t i = 0; i < N; i++) {
                    if (done[i]) { episodes++; successes += reward[i] > 5.0f; }
                }
            }
            LossStats s = model.lastStats();
            std::printf("iter %3d  success %.3f  episodes %5llu  entropy %.3f\n",
                        it, episodes ? (double)successes / episodes : 0.0,
                        (unsigned long long)episodes, s.entropy);
        }
    }
    return 0;
}
