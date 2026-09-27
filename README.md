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
    struct State {};          //per-env state, kept on the GPU between steps
    struct Action {};       //what the policy outputs
    struct Obs {};  //what the policy reads

    constant static constexpr uint OBS_DIM;
    constant static constexpr uint ACT_DIM; //policy output width

    //any additional constants

    static void reset(thread State& s, thread uint& rng) {  }

    static void step(thread State& s, Action action, thread Obs& obs,
                     thread float& reward, thread uchar& done) {  }
};

//instantiate the metal rollout function
template [[host_name("{kernel_name}")]]
kernel void rollout<Environment>(
    device Environment::State*,
    device const Environment::Action*,
    device Environment::Obs*,
    device float*,
    device uchar*,
    device uint*,
    uint
);
```

### Host side

```cpp
struct Environment {
    struct State {};         //same fields and order as the Metal State
    struct Action {};       //same as the Metal Action
    struct Obs {};  		//same as the Metal Obs
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

CartPole::Action* actions = gym.actions(); //one Action per env, written before each step
for (int i = 0; i < 4096; i++) actions[i] = 0;

gym.run(reward);                  //one step for every env

CartPole::Obs* obs = gym.obs();   //one Obs per env
float* flat = gym.obsFlat();      //the same buffer as OBS_DIM floats per env
uint8_t* done = gym.done();       //1 when the episode ended on this step
```

`State`, `Action`, and `Obs` are plain structs of 4-byte scalars, so the
host and Metal layouts match. Both sides check that `Obs` is exactly
`OBS_DIM` floats, which lets the policy read the obs buffer as a flat matrix.

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
