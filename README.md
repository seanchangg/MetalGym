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

## Minimum training loop

```cpp
#include "gym.h"
#include "model/model.h"

Gym<Environment> gym(4096);                    //4096 environments, one per GPU thread
Model<Environment> model(gym, 128, 4, 2, 3e-4f, 32); //embed 128, mlp x4, 2 blocks, lr, 32 steps per rollout

for (int it = 0; it < 100; it++) {
    model.collect(); //T env steps + GAE, one GPU submission
    model.train();   //4 shuffled PPO epochs over the rollout, one GPU submission per minibatch
}
```

`Model<Env>` borrows the gym's device, queue, rng, and obs buffer. Nothing is
copied between the two. The policy is a residual MLP with a softmax head, so
`Env::Action` must be a 4-byte integer and actions are indices below `ACT_DIM`.

`train(epochs)` takes the epoch count; the default is 4. Both calls block
until the GPU finishes.

## Observability

Everything below is a read of a shared buffer. None of it runs unless called.

### After `collect()`

One entry per environment, for step `t` of the last rollout, `0 <= t < T`.

| Method | Type | Meaning |
|---|---|---|
| `model.slotObs(t)` | `Env::Obs*` | the obs the policy saw at step `t` |
| `model.slotActions(t)` | `Env::Action*` | the action it took |
| `model.slotRewards(t)` | `float*` | the reward the env returned |
| `model.slotDones(t)` | `uint8_t*` | 1 if the episode ended on this step |
| `model.slotValues(t)` | `float*` | the critic's value estimate |
| `model.slotAdvantages(t)` | `float*` | GAE advantage |
| `model.slotReturns(t)` | `float*` | value target, advantage + value |

Mean episode length, the usual CartPole score, is a loop over `slotDones`.

### After `train()` or `trainSlot(t)`

| Method | Type | Meaning |
|---|---|---|
| `model.lastStats()` | `LossStats` | batch means from the most recent minibatch |
| `model.headValue(row, col)` | `float` | raw head output of the last forward: logit at `col < ACT_DIM`, value at `col == ACT_DIM` |

`LossStats` fields: `policy` (clipped surrogate), `value`, `entropy`,
`approxKl` (mean `oldLogProb - newLogProb`), `clipFrac` (share of rows outside
the clip band), `gradNorm` (before clipping).

### Hyperparameters

Public fields on the model, read on every call: `clip` 0.2, `valueCoef` 0.5,
`entropyCoef` 0.01, `maxGradNorm` 0.5 (0 disables), `gamma` 0.99,
`lambda` 0.95.

### Manual stepping

For evaluation or a hand-written policy, drive the gym one step at a time.

| Call | Effect |
|---|---|
| `model.forward()` | policy forward on `gym.obs()`; head in the model |
| `model.act(rng)` | sample one action per env into `gym.actions()`; also fills `model.values()` and `model.logProbs()` |
| `model.actGreedy()` | argmax action per env into `gym.actions()` |
| `gym.run(reward)` | one env step; `reward` gets one float per env |
| `gym.obs()` | `Env::Obs*`, the live obs |
| `gym.actions()` | `Env::Action*`, written before `run` |
| `gym.done()` | `uint8_t*`, 1 when the episode ended on this step |

A random policy is a loop that writes `gym.actions()` and calls `gym.run()`.

The `done` buffer starts at 1, so the first step resets every environment.
When an environment finishes an episode, the next step resets it and steps
once with the action given.

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

The build compiles every `.metal` file it globs into `default.metallib` next
to the executable. Run the first `cmake` command again after you add or remove
a `.metal` file.

The network width is a compile-time constant in `model/config.h`
(`N_EMBED_CFG`, default 128), because the Metal matmul descriptors need it.
The `Model` constructor checks its `embedDim` against it.

## Files

| File | Contents |
|---|---|
| `rollout.h` | The rollout kernel template and the per-env random number helpers |
| `gym.h` | The env runtime: device, buffers, `run`, and `encodeStep` |
| `model/model.h` | `Model<Env>`: the policy, `collect`, `train`, and the accessors above |
| `model/config.h` | `N_EMBED_CFG`, shared by the shaders and the host |
| `model/*.metal` | The network kernels: input projection, layernorm, mlp, head, PPO loss, sampler, GAE, Adam |
| `examples/CartPole/` | The CartPole environment and the minimum training loop |
