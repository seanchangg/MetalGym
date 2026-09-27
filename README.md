# metalRL

A reinforcement learning framework on Apple Metal. Each environment runs one
instance per GPU thread. The framework handles the rollout kernel, the random
numbers, and the host buffers. You write the environment.

## Environment template

### Metal side

```metal
#include <metal_stdlib>
#include "rollout.h"
using namespace metal;

struct Environment {
    struct State {};

    constant static constexpr uint OBS_DIM;
    constant static constexpr uint ACT_DIM;

    //any additional constants

    static void reset(thread State& s, thread uint& rng) {  }

    static void observe(thread const State& s, device float* obs) {  }

    static void step(thread State& s, int action, device float* obs,
                     thread float& reward, thread uchar& done) {  }
};

//instantiate the metal rollout function
template [[host_name("{kernel_name}")]]
kernel void rollout<Environment>(
    device Environment::State*,
    device const int*,
    device float*,
    device float*,
    device uchar*,
    device uint*,
    uint
);
```

### Host side

```cpp
struct Environment {
    struct State { };
    static constexpr uint32_t OBS_DIM;
    static constexpr uint32_t ACT_DIM;
    static constexpr const char* KERNEL; //make sure this matches the kernel name
};
```

## Build

Requirements: Xcode command line tools, CMake 3.20 or newer, and the
[metal-cpp](https://developer.apple.com/metal/cpp/) headers. CMake looks for
metal-cpp in `~/.local/include/metal-cpp`. Set `METAL_CPP_DIR` to use a
different path.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/metalRL
```

The build compiles every `.metal` file in the project root into
`default.metallib` next to the executable. The shader list is a glob, so run
the first `cmake` command again after you add or remove a `.metal` file.

## Usage

`Gym<Env>` owns the Metal device, the pipeline, and one buffer per kernel
argument. The host writes actions, calls `run`, and reads the results.

```cpp
#include "gym.h"

Gym<CartPole> gym(4096);          //4096 environments, one per GPU thread
std::vector<float> reward(4096);

int* actions = gym.actions();     //one int per env, written before each step
for (int i = 0; i < 4096; i++) actions[i] = 0;

gym.run(reward);                  //one step for every env

float* obs = gym.obs();           //OBS_DIM floats per env
uint8_t* done = gym.done();       //1 when the episode ended on this step
```

The `done` buffer starts at 1, so the first `run` resets every environment.
When an environment finishes an episode, the next `run` resets it and steps
once with the action given.

`main.cpp` runs CartPole with a random policy and prints the mean episode
length. Gymnasium CartPole-v1 gives about 22 with the same policy, which
checks the physics.

## Files

| File | Contents |
|---|---|
| `rollout.h` | The rollout kernel template and the per-env random number helpers |
| `cartpole.metal` | The CartPole environment and its kernel instantiation |
| `gym.h` | The host side: device setup, buffers, and the `run` loop |
| `main.cpp` | The host `CartPole` struct and the random policy check |
