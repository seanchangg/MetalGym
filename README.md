# MetalGym

The **fastest** reinforcement learning framework on Apple Metal. You write the environment, and the framework handles the rollout kernel, the random
numbers, and a fused MLP inference. 

Performance comparisons:
| | env steps/s | with policy | full PPO loop |
|---|---|---|---|
| CartPole, MetalGym | 27 M | 3.8 M | 296 K |
| CartPole, JAX + jax-metal + gymnax | 1.5 M | 1.7 M | 180 K |
| CartPole, PufferLib 2.0.6, torch CPU | 0.9 M | 0.17 M | 56 K |
| VoxelNav, MetalGym | 7.9 M | 2.4 M | 265 K |
| VoxelNav, JAX on CPU (jax-metal cannot run it, see below) | 0.44 M | 0.27 M | 47 K |

Configuration and the other rows are in [Bench](#bench).

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
#include "model/ppo.h"

Gym<Environment> gym(4096);                    //4096 environments, one per GPU thread
PPOModel<Environment> model(gym, 128, 4, 2, 3e-4f, 32); //embed 128, mlp x4, 2 blocks, lr, 32 steps per rollout

for (int it = 0; it < 100; it++) {
    model.collect(); //T env steps + GAE, one GPU submission
    model.train();   //4 shuffled PPO epochs over the rollout, one GPU submission per minibatch
}
```

`PPOModel<Env>` borrows the gym's device, queue, rng, and obs buffer. Nothing is
copied between the two. The policy is a residual MLP. The type of `Env::Action`
selects the head at compile time:

| `Env::Action` | Head | Action |
|---|---|---|
| a 4-byte integer, e.g. `int` | softmax over `ACT_DIM` logits | an index below `ACT_DIM` |
| a struct of `ACT_DIM` floats, e.g. `struct Action { float torque, bend; };` | Gaussian with `ACT_DIM` means and a learned, shared `log_std` | `ACT_DIM` unbounded floats |

The Gaussian sample is not bounded. Clamp or `tanh` it inside `step()`.
`log_std` starts at 0 (std 1); `model.setLogStd(v)` overwrites it and
`model.logStdValue(k)` reads it. `examples/CartPoleContinuous` is the
continuous template.

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
| `model.headValue(row, col)` | `float` | raw head output of the last forward: logit or mean at `col < ACT_DIM`, value at `col == ACT_DIM` |
| `model.logStdValue(k)` | `float` | continuous only: the shared log standard deviation of action dim `k` |

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
| `model.actGreedy()` | argmax action (discrete) or the mean (continuous) per env into `gym.actions()` |
| `gym.run(reward)` | one env step; `reward` gets one float per env |
| `gym.obs()` | `Env::Obs*`, the live obs |
| `gym.actions()` | `Env::Action*`, written before `run` |
| `gym.done()` | `uint8_t*`, 1 when the episode ended on this step |

A random policy is a loop that writes `gym.actions()` and calls `gym.run()`.

The `done` buffer starts at 1, so the first step resets every environment.
When an environment finishes an episode, the next step resets it and steps
once with the action given.

## Benchmarks

Apple M4, 10 GPU cores, macOS 26. CartPole, 4096 environments, 32-step
rollouts, PPO with 4 epochs and a minibatch of 4096. Steps per second means
environment steps. "Collect" includes the policy forward and GAE. "Full loop"
includes training. Higher is better.

| | env step only | collect | full loop |
|---|---|---|---|
| MetalGym, 1 block | 27 M | 6.7 M | 496 K |
| MetalGym, 2 blocks | 27 M | 3.8 M | 296 K |
| JAX 0.4.34 + jax-metal + gymnax, plain 128-512-128 MLP | 1.5 M (6.2 M scanned) | 2.4 M | 399 K |
| JAX 0.4.34 + jax-metal + gymnax, same 2-block policy | 1.5 M (6.2 M scanned) | 1.7 M | 180 K |
| PufferLib 2.0.6, Gymnasium CartPole, 8 workers, torch CPU | 0.9 M | 0.17 M | 56 K |

"Scanned" is 100 env steps inside one `jax.lax.scan`, which is the fastest
way to drive gymnax. The JAX policies run in fp32; MetalGym runs bf16 matmuls
with fp32 accumulation. PufferLib has no Apple GPU backend, so its row is the
configuration a PufferLib user gets on a Mac. PufferLib's native C
environments step at about 94 M steps per second on this machine, but there is
no CartPole among them.

### VoxelNav

`examples/VoxelNav` is a heavier environment after minetrix's bridge task:
a 32 x 32 heightmap with a chasm, six actions including cobble placement, a
depth camera of 32 rays traced through the heightmap, and potential-based
reward shaping. State is about 1 KB per environment. PPO reaches 93 percent
success in 200 iterations, about 100 seconds.

| | env step only | collect | full loop |
|---|---|---|---|
| MetalGym, 1 block | 7.9 M | 3.3 M | 413 K |
| MetalGym, 2 blocks | 7.9 M | 2.4 M | 265 K |
| JAX 0.4.34 on CPU, same env in jnp, 2-block policy | 0.44 M (0.93 M scanned) | 0.27 M | 47 K |
| JAX 0.4.34 + jax-metal, same env | about 300 steps/s at 256 envs | not run | not run |

The JAX twin is `bench/jax_voxelnav_bench.py`, the same generator, camera,
actions, and reward written with vmap. The jax-metal gather kernel seems to be broken, taking 23ms per 256 environments, so not sure what's going on there. Fallback to CPU runs the same code at full speed. 

```sh
clang++ -std=c++20 -O2 -I. -I$METAL_CPP_DIR bench/MetalGym_bench.cpp -o bench_MetalGym \
    -framework Metal -framework Foundation -framework QuartzCore
./bench_MetalGym 4096 2          # envs, blocks; needs default.metallib next to it
python bench/jax_bench.py 4096  # pip install "jax==0.4.34" "jaxlib==0.4.34" jax-metal gymnax optax flax
python bench/puffer_bench.py    # pip install "pufferlib==2.0.6" gymnasium torch
# VoxelNav: bench/MetalGym_bench_voxelnav.cpp the same way; JAX_PLATFORMS=cpu python bench/jax_voxelnav_bench.py 4096
```

## Build

Requirements: Xcode command line tools, CMake 3.20 or newer, and the
[metal-cpp](https://developer.apple.com/metal/cpp/) headers. CMake looks for
metal-cpp in `~/.local/include/metal-cpp`. Set `METAL_CPP_DIR` to use a
different path.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cartpole      # or ./build/voxelnav, ./build/bench_cartpole, ./build/bench_voxelnav
```

Every `.metal` file in the root, `model/`, and `examples/*/` compiles into one
`default.metallib` next to the executables. The shader list is a glob, so run
the first `cmake` command again after you add a `.metal` file. To add an
example, add one `metalrl_example(name path/to/main.cpp)` line to
`CMakeLists.txt`.

The network width is a compile-time constant in `model/config.h`
(`N_EMBED_CFG`, default 128), because the Metal matmul descriptors need it.
The `PPOModel` constructor checks its `embedDim` against it.

## Files

| File | Contents |
|---|---|
| `rollout.h` | The rollout kernel template and the per-env random number helpers |
| `gym.h` | The env runtime: device, buffers, `run`, and `encodeStep` |
| `model/ppo.h` | `PPOModel<Env>`: the policy, `collect`, `train`, and the accessors above |
| `model/config.h` | `N_EMBED_CFG`, shared by the shaders and the host |
| `model/*.metal` | The network kernels: input projection, layernorm, mlp, head, PPO loss, sampler, GAE, Adam |
| `examples/CartPole/` | The CartPole environment and the minimum training loop |
| `examples/VoxelNav/` | Heightmap navigation with bridging, a 32-ray depth camera, and a success-rate print |
| `bench/` | Throughput benchmarks: MetalGym, JAX + gymnax, PufferLib |
