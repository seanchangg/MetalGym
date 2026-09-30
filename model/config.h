//Shared by the Metal shaders and the host. The MPP matmul descriptors in
//mlp.metal and linear.metal need the embed width at compile time, so it lives
//here, in one place, and PPOModel checks its embedDim against it at runtime.
//Change this value and rebuild the shaders to change the network width.
#pragma once

#define N_EMBED_CFG 128
