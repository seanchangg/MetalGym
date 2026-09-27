#include "gym.h"
#include <random>

struct CartPole {
	struct State {
        float x;
        float xDot;
        float theta;
        float thetaDot;
        uint32_t steps;
    };
	static constexpr uint32_t OBS_DIM = 4;
    static constexpr uint32_t ACT_DIM = 2;
    static constexpr const char* KERNEL = "rollout_cartpole";
};

//Random policy check. Gymnasium CartPole-v1 with uniform random actions
//gives a mean episode length of about 22.
int main() {
	const uint32_t N = 4096;
	const int STEPS = 1000;
	Gym<CartPole> gym(N);
	std::vector<float> reward(N, 0);
	std::vector<uint32_t> length(N, 0);
	std::mt19937 rng(42);
	std::bernoulli_distribution coin(0.5);

	double totalLength = 0;
	uint64_t episodes = 0;
	for (int t = 0; t < STEPS; t++) {
		int* act = gym.actions();
		for (uint32_t i = 0; i < N; i++) act[i] = coin(rng) ? 1 : 0;
		gym.run(reward);
		uint8_t* done = gym.done();
		for (uint32_t i = 0; i < N; i++) {
			length[i]++;
			if (done[i]) {
				//the kernel resets and steps in the same call, so every
				//call counts as one real step of the episode
				totalLength += length[i];
				episodes++;
				length[i] = 0;
			}
		}
	}
	std::cout << "episodes: " << episodes
	          << "  mean length: " << (episodes ? totalLength / episodes : 0) << "\n";
	return 0;
}
