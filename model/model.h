//Actor-critic policy for one Env, shared by every environment row. forward()
//encodes obs projection -> every block -> ln_final -> head into ONE command
//buffer, backward() encodes the mirror image (PPO loss -> head -> ... ->
//obs projection grads) into ONE command buffer, and step() is the fused Adam
//update. Residual adds run on the GPU via the residualAdd kernel in
//elementwise.metal, so nothing breaks the pass into pieces. Activations live
//in per-block buffers between forward and backward. Weights and Adam state
//are GPU-resident: uploaded once at construction, gradients gathered and
//weights updated entirely on the GPU by step() (gatherGrad*/adamStep/
//scatterWeight), so the host only uploads the batch each iteration.
//
//Network: obs (M x OBS_DIM float) -> inputForward -> M x embedDim stream ->
//N x (layernorm -> mlp -> residual) -> ln_final -> linear head (M x HEAD_DIM).
//Each head row is [logits(ACT_DIM) | value | zero padding], so one 64-wide
//linear tile serves both the actor and the critic.
//
//The row count M is the environment count, the same as Gym<Env>. The obs,
//action, value, and logProb buffers mirror gym.h: the host copies gym obs in,
//calls forward() and act(), and copies actions() back to the gym.
//
//Constraints from the kernels: M and embedDim are multiples of 64,
//embedDim <= 1024, and embedDim == N_EMBED_CFG from model/config.h, the one
//compile-time constant the MPP matmul descriptors in mlp.metal and
//linear.metal need. The shaders include the same header.
//
//Depends only on metal-cpp. Include this AFTER gym.h in the same translation
//unit, because gym.h defines NS_PRIVATE_IMPLEMENTATION and friends.
#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "config.h"

//host mirrors of the kernel parameter structs in model/*.metal
struct InputParams {
    uint32_t rows;
    uint32_t inDim;
    uint32_t outDim;
};
struct LayernormParams {
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
    float eps;
    LayernormParams(uint32_t m, uint32_t e)
        : rows(m), cols(e), group_size(std::min<uint32_t>(e, 1024)), eps(1e-5f) {}
};
struct mlpParams {
    uint32_t M;
    uint32_t N;
    uint32_t S;
};
struct LinearParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
};
struct PolicyParams {
    uint32_t rows;
    uint32_t actDim;
    uint32_t headDim;
    float clip;
    float valueCoef;
    float entropyCoef;
};
struct AdamParams {
    uint32_t count;
    float lr;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float beta1Pow;
    float beta2Pow;
    float gradScale = 1.0f;
};

//batch means returned by backward()
struct LossStats {
    float policy;   //clipped surrogate
    float value;    //valueCoef * 0.5 * (v - R)^2
    float entropy;
    float approxKl; //mean(oldLogProb - newLogProb)
    float clipFrac; //fraction of rows with |ratio - 1| > clip
};

template <typename Env>
struct Model {
    static_assert(sizeof(typename Env::Obs) == Env::OBS_DIM * sizeof(float),
                  "Env::Obs must be exactly OBS_DIM floats");
    static_assert(std::is_integral_v<typename Env::Action> && sizeof(typename Env::Action) == 4,
                  "the softmax head needs a discrete 4-byte Action");
    static constexpr uint32_t HEAD_DIM = 64; //one linear tile: logits, value, zero padding
    static_assert(Env::ACT_DIM + 1 <= HEAD_DIM, "ACT_DIM + value must fit one 64-wide tile");

    struct LayernormBuffers {
        MTL::Buffer* gammaBuffer;
        MTL::Buffer* betaBuffer;
        MTL::Buffer* xnormBuffer;
        MTL::Buffer* stdevBuffer; //float
        MTL::Buffer* outBuffer;
        MTL::Buffer* dGammaBuffer; //float atomic accumulators
        MTL::Buffer* dBetaBuffer;
    };
    struct BlockBuffers {
        LayernormBuffers ln;
        //mlp weights + forward caches
        MTL::Buffer* upprojBuffer;
        MTL::Buffer* downprojBuffer;
        MTL::Buffer* mlpVBuffer;
        MTL::Buffer* mlpMaskBuffer;
        MTL::Buffer* mlpOutBuffer;
        MTL::Buffer* residualBuffer;
        //gradient outputs gathered by step()
        MTL::Buffer* dUpBuffer;
        MTL::Buffer* dDpBuffer;
    };

    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;

    //one pipeline per kernel, shared by every block
    MTL::ComputePipelineState* inputFwdPipeline;
    MTL::ComputePipelineState* inputBwdPipeline;
    MTL::ComputePipelineState* layernormFwdPipeline;
    MTL::ComputePipelineState* layernormBwdPipeline;
    MTL::ComputePipelineState* mlpFwdOnePipeline;
    MTL::ComputePipelineState* mlpFwdTwoPipeline;
    MTL::ComputePipelineState* mlpBwdOnePipeline;
    MTL::ComputePipelineState* mlpBwdTwoPipeline;
    MTL::ComputePipelineState* linearFwdPipeline;
    MTL::ComputePipelineState* linearBwdPipeline;
    MTL::ComputePipelineState* policyLossPipeline;
    MTL::ComputePipelineState* residualAddPipeline;
    MTL::ComputePipelineState* stepPipeline;
    MTL::ComputePipelineState* gatherFloatPipeline;
    MTL::ComputePipelineState* gatherBfloatPipeline;
    MTL::ComputePipelineState* scatterPipeline;

    //dimensions
    uint32_t parallels; //rows per pass (M), one per environment
    uint32_t embedDim;
    uint32_t mlpScale;
    uint32_t nLayers;
    float lr;
    uint32_t hiddenDim; //embedDim * mlpScale
    uint32_t tileRows;  //parallels / 64
    uint32_t tileCols;  //embedDim / 64
    uint32_t activationCount; //parallels * embedDim
    static constexpr uint32_t tileM = 64;
    static constexpr uint32_t tileN = 64;
    static constexpr uint32_t simdGroups = 4;

    //PPO hyperparameters, read on every backward()/step(); change them freely
    float clip = 0.2f;
    float valueCoef = 0.5f;
    float entropyCoef = 0.01f;
    float maxGradNorm = 0.5f; //0 disables clipping

    //kernel params, one of each shared by every block
    InputParams inputParams;
    LayernormParams layernormParams;
    mlpParams mlpParameters;
    LinearParams linearParams;

    //common byte sizes
    size_t activationBytes; //parallels * embedDim * bf16
    size_t hiddenBytes;     //parallels * hiddenDim * bf16
    size_t gammaFloatBytes; //embedDim * float

    //gym-facing buffers, one entry per environment
    MTL::Buffer* obsBuffer;     //parallels x Env::Obs, float; the host fills it before forward()
    MTL::Buffer* actionBuffer;  //parallels x Env::Action, written by act()
    MTL::Buffer* valueBuffer;   //parallels floats, written by act()
    MTL::Buffer* logProbBuffer; //parallels floats, written by act()

    //obs projection: the first layer, and the start of the residual stream
    MTL::Buffer* inWBuffer;   //OBS_DIM x embedDim bf16
    MTL::Buffer* inBBuffer;   //embedDim bf16
    MTL::Buffer* inputBuffer; //parallels x embedDim bf16
    MTL::Buffer* dInWBuffer;  //float
    MTL::Buffer* dInBBuffer;  //float

    //blocks
    std::vector<BlockBuffers> blocks;

    //final layernorm / linear head
    LayernormBuffers lnFinal;
    MTL::Buffer* linearWBuffer;  //embedDim x HEAD_DIM bf16
    MTL::Buffer* logitsBuffer;   //parallels x HEAD_DIM bf16: logits, value, padding
    MTL::Buffer* dLogitsBuffer;
    MTL::Buffer* dLinearWBuffer;

    //training batch, uploaded by backward()
    MTL::Buffer* batchActionBuffer; //uint32
    MTL::Buffer* advantageBuffer;
    MTL::Buffer* returnBuffer;
    MTL::Buffer* oldLogProbBuffer;
    MTL::Buffer* statsBuffer;       //parallels x 5 floats

    //backward flow, shared by every block since the encoders run in order
    MTL::Buffer* dStreamBuffer; //gradient flowing down the residual stream
    MTL::Buffer* dBranchBuffer; //branch gradient inside a block
    MTL::Buffer* dXnormBuffer;  //layernorm backward scratch
    MTL::Buffer* dMlpVBuffer;

    //optimizer: weights and Adam state are GPU-resident. The host vectors in
    //optWeights hold the initial values, read once at construction (initial
    //upload + master build) and never touched again; per-parameter
    //bookkeeping below drives the gather (grads -> flat) and scatter
    //(master -> weight buffers) passes that bracket the Adam kernel inside
    //step().
    struct FlatCopyParams { uint32_t count; uint32_t offset; };
    std::vector<std::vector<__bf16>> optWeights;  //host init values, used for sizes + one-time upload
    std::vector<MTL::Buffer*> optWeightBuffers;   //bf16 buffers the forward kernels read
    std::vector<MTL::Buffer*> optGradBuffers;     //buffers backward writes each parameter's grad into
    std::vector<bool> optGradIsFloat;             //float accumulators vs bf16
    std::vector<size_t> optOffsets;
    size_t optCount = 0;
    float stepCounter = 0.0f;
    MTL::Buffer* parameterBuffer;
    MTL::Buffer* gradientBuffer;
    MTL::Buffer* momentumBuffer;
    MTL::Buffer* varianceBuffer;

    Model(int p, int e, int s, int layers, float learningRate, uint32_t seed = 0)
        : parallels(p), embedDim(e), mlpScale(s), nLayers(layers), lr(learningRate),
          hiddenDim(e * s), tileRows(p / tileM), tileCols(e / tileN),
          activationCount((uint32_t)(p * e)),
          inputParams{(uint32_t)p, Env::OBS_DIM, (uint32_t)e},
          layernormParams(p, e),
          mlpParameters{(uint32_t)p, (uint32_t)e, (uint32_t)s},
          linearParams{(uint32_t)p, HEAD_DIM, (uint32_t)e},
          activationBytes((size_t)p * e * sizeof(__bf16)),
          hiddenBytes((size_t)p * e * s * sizeof(__bf16)),
          gammaFloatBytes((size_t)e * sizeof(float)) {
        if (p % tileM != 0 || e % tileN != 0 || e > 1024 || hiddenDim % tileN != 0 || layers < 1) {
            std::cerr << "Model: rows and embedDim must be multiples of 64, embedDim <= 1024, layers >= 1\n";
            std::exit(1);
        }
        if (e != N_EMBED_CFG) {
            std::cerr << "Model: embedDim " << e << " != N_EMBED_CFG " << N_EMBED_CFG
                      << " from model/config.h\n";
            std::exit(1);
        }
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();

        inputFwdPipeline = makePipeline(library, "inputForward");
        inputBwdPipeline = makePipeline(library, "inputBackward");
        layernormFwdPipeline = makePipeline(library, "layernormForward");
        layernormBwdPipeline = makePipeline(library, "layernormBackward");
        mlpFwdOnePipeline = makePipeline(library, "mlpForwardOne");
        mlpFwdTwoPipeline = makePipeline(library, "mlpForwardTwo");
        mlpBwdOnePipeline = makePipeline(library, "mlpBackwardOne");
        mlpBwdTwoPipeline = makePipeline(library, "mlpBackwardTwo");
        linearFwdPipeline = makePipeline(library, "linearForward");
        linearBwdPipeline = makePipeline(library, "linearBackward");
        policyLossPipeline = makePipeline(library, "policyLossBackward");
        residualAddPipeline = makePipeline(library, "residualAdd");
        stepPipeline = makePipeline(library, "adamStep");
        gatherFloatPipeline = makePipeline(library, "gatherGradFloat");
        gatherBfloatPipeline = makePipeline(library, "gatherGradBfloat");
        scatterPipeline = makePipeline(library, "scatterWeight");
        library->release();

        //gym-facing buffers
        obsBuffer = newSharedBuffer((size_t)parallels * sizeof(typename Env::Obs));
        actionBuffer = newSharedBuffer((size_t)parallels * sizeof(typename Env::Action));
        valueBuffer = newSharedBuffer((size_t)parallels * sizeof(float));
        logProbBuffer = newSharedBuffer((size_t)parallels * sizeof(float));
        std::memset(obsBuffer->contents(), 0, (size_t)parallels * sizeof(typename Env::Obs));
        std::memset(actionBuffer->contents(), 0, (size_t)parallels * sizeof(typename Env::Action));

        //obs projection
        inWBuffer = newSharedBuffer((size_t)Env::OBS_DIM * embedDim * sizeof(__bf16));
        inBBuffer = newSharedBuffer((size_t)embedDim * sizeof(__bf16));
        inputBuffer = newSharedBuffer(activationBytes);
        dInWBuffer = newSharedBuffer((size_t)Env::OBS_DIM * embedDim * sizeof(float));
        dInBBuffer = newSharedBuffer((size_t)embedDim * sizeof(float));

        //blocks
        blocks.resize(nLayers);
        for (BlockBuffers& block : blocks) {
            allocLayernorm(block.ln);
            block.upprojBuffer = newSharedBuffer((size_t)embedDim * hiddenDim * sizeof(__bf16));
            block.downprojBuffer = newSharedBuffer((size_t)hiddenDim * embedDim * sizeof(__bf16));
            block.mlpVBuffer = newSharedBuffer(hiddenBytes);
            block.mlpMaskBuffer = newSharedBuffer((size_t)parallels * hiddenDim * sizeof(uint8_t));
            block.mlpOutBuffer = newSharedBuffer(activationBytes);
            block.residualBuffer = newSharedBuffer(activationBytes);
            block.dUpBuffer = newSharedBuffer((size_t)embedDim * hiddenDim * sizeof(__bf16));
            block.dDpBuffer = newSharedBuffer((size_t)hiddenDim * embedDim * sizeof(__bf16));
        }

        //final layernorm / linear head
        allocLayernorm(lnFinal);
        linearWBuffer = newSharedBuffer((size_t)embedDim * HEAD_DIM * sizeof(__bf16));
        logitsBuffer = newSharedBuffer((size_t)parallels * HEAD_DIM * sizeof(__bf16));
        dLogitsBuffer = newSharedBuffer((size_t)parallels * HEAD_DIM * sizeof(__bf16));
        dLinearWBuffer = newSharedBuffer((size_t)embedDim * HEAD_DIM * sizeof(__bf16));

        //training batch
        batchActionBuffer = newSharedBuffer((size_t)parallels * sizeof(uint32_t));
        advantageBuffer = newSharedBuffer((size_t)parallels * sizeof(float));
        returnBuffer = newSharedBuffer((size_t)parallels * sizeof(float));
        oldLogProbBuffer = newSharedBuffer((size_t)parallels * sizeof(float));
        statsBuffer = newSharedBuffer((size_t)parallels * 5 * sizeof(float));

        //backward flow
        dStreamBuffer = newSharedBuffer(activationBytes);
        dBranchBuffer = newSharedBuffer(activationBytes);
        dXnormBuffer = newSharedBuffer(activationBytes);
        dMlpVBuffer = newSharedBuffer(hiddenBytes);

        //optimizer: register every parameter with its GPU weight/grad buffer
        //and its initial values, then build the flat master copy and Adam
        //state once. Registration order defines the flat layout, nothing else
        //depends on it.
        std::mt19937 rng(seed);
        auto normal = [&](std::vector<__bf16>& weight, float stdev) {
            std::normal_distribution<float> dist(0.0f, stdev);
            for (__bf16& x : weight) x = (__bf16)dist(rng);
        };
        auto constant = [](std::vector<__bf16>& weight, float value) {
            std::fill(weight.begin(), weight.end(), (__bf16)value);
        };
        normal(addParam((size_t)Env::OBS_DIM * embedDim, inWBuffer, dInWBuffer, true),
               1.0f / std::sqrt((float)Env::OBS_DIM));
        constant(addParam((size_t)embedDim, inBBuffer, dInBBuffer, true), 0.0f);
        for (uint32_t n = 0; n < nLayers; ++n) {
            BlockBuffers& block = blocks[n];
            constant(addParam((size_t)embedDim, block.ln.gammaBuffer, block.ln.dGammaBuffer, true), 1.0f);
            constant(addParam((size_t)embedDim, block.ln.betaBuffer, block.ln.dBetaBuffer, true), 0.0f);
            normal(addParam((size_t)embedDim * hiddenDim, block.upprojBuffer, block.dUpBuffer, false),
                   std::sqrt(2.0f / (float)embedDim));
            //scale the branch output down so the residual stream stays O(1) across layers
            normal(addParam((size_t)hiddenDim * embedDim, block.downprojBuffer, block.dDpBuffer, false),
                   1.0f / std::sqrt((float)hiddenDim) / std::sqrt((float)nLayers));
        }
        constant(addParam((size_t)embedDim, lnFinal.gammaBuffer, lnFinal.dGammaBuffer, true), 1.0f);
        constant(addParam((size_t)embedDim, lnFinal.betaBuffer, lnFinal.dBetaBuffer, true), 0.0f);
        {
            //head layout is [K = embedDim rows] x [N = HEAD_DIM cols], row-major.
            //logits start small so the initial policy is near uniform; the
            //value column gets a unit-scale init; padding columns stay zero.
            std::vector<__bf16>& head = addParam((size_t)embedDim * HEAD_DIM, linearWBuffer, dLinearWBuffer, false);
            std::normal_distribution<float> policyDist(0.0f, 0.01f);
            std::normal_distribution<float> valueDist(0.0f, 1.0f / std::sqrt((float)embedDim));
            for (uint32_t k = 0; k < embedDim; ++k) {
                for (uint32_t n = 0; n < HEAD_DIM; ++n) {
                    float x = 0.0f;
                    if (n < Env::ACT_DIM) x = policyDist(rng);
                    else if (n == Env::ACT_DIM) x = valueDist(rng);
                    head[(size_t)k * HEAD_DIM + n] = (__bf16)x;
                }
            }
        }

        size_t total = 0;
        for (std::vector<__bf16>& weight : optWeights) {
            optOffsets.push_back(total);
            total += weight.size();
        }
        optCount = total;
        parameterBuffer = newSharedBuffer(optCount * sizeof(float));
        gradientBuffer = newSharedBuffer(optCount * sizeof(float));
        momentumBuffer = newSharedBuffer(optCount * sizeof(float));
        varianceBuffer = newSharedBuffer(optCount * sizeof(float));

        //one-time upload: master floats + Adam state + the bf16 weight buffers
        auto* masterOut = static_cast<float*>(parameterBuffer->contents());
        for (size_t i = 0; i < optWeights.size(); ++i) {
            std::transform(optWeights[i].begin(), optWeights[i].end(), masterOut + optOffsets[i],
                           [](__bf16 x) { return (float)x; });
            std::memcpy(optWeightBuffers[i]->contents(), optWeights[i].data(), optWeights[i].size() * sizeof(__bf16));
        }
        std::memset(gradientBuffer->contents(), 0, optCount * sizeof(float));
        std::memset(momentumBuffer->contents(), 0, optCount * sizeof(float));
        std::memset(varianceBuffer->contents(), 0, optCount * sizeof(float));
    }

    //------------------------------------------------------------- accessors
    //the same shape as gym.h: one entry per environment
    typename Env::Obs* obs() { return static_cast<typename Env::Obs*>(obsBuffer->contents()); }
    //the same buffer as OBS_DIM floats per env
    float* obsFlat() { return static_cast<float*>(obsBuffer->contents()); }
    typename Env::Action* actions() { return static_cast<typename Env::Action*>(actionBuffer->contents()); }
    float* values() { return static_cast<float*>(valueBuffer->contents()); }
    float* logProbs() { return static_cast<float*>(logProbBuffer->contents()); }
    //the head output of the last forward(): logits[row * HEAD_DIM + a], value at a == ACT_DIM
    float headValue(uint32_t row, uint32_t col) const {
        return (float)static_cast<const __bf16*>(logitsBuffer->contents())[(size_t)row * HEAD_DIM + col];
    }

    //---------------------------------------------------------------- helpers
    MTL::ComputePipelineState* makePipeline(MTL::Library* library, const char* name) {
        MTL::Function* function = library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
        if (!function) {
            std::cerr << name << ": kernel not in default.metallib\n";
            std::exit(1);
        }
        NS::Error* error = nullptr;
        MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
        if (!pipeline) {
            std::cerr << name << " pipeline: " << error->localizedDescription()->utf8String() << "\n";
            std::exit(1);
        }
        function->release();
        return pipeline;
    }
    MTL::Buffer* newSharedBuffer(size_t bytes) {
        return device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    }
    void allocLayernorm(LayernormBuffers& ln) {
        ln.gammaBuffer = newSharedBuffer((size_t)embedDim * sizeof(__bf16));
        ln.betaBuffer = newSharedBuffer((size_t)embedDim * sizeof(__bf16));
        ln.xnormBuffer = newSharedBuffer(activationBytes);
        ln.stdevBuffer = newSharedBuffer((size_t)parallels * sizeof(float));
        ln.outBuffer = newSharedBuffer(activationBytes);
        ln.dGammaBuffer = newSharedBuffer(gammaFloatBytes);
        ln.dBetaBuffer = newSharedBuffer(gammaFloatBytes);
    }
    MTL::ComputeCommandEncoder* makeEncoder(MTL::CommandBuffer* commandBuffer,
                                            MTL::ComputePipelineState* pipeline) {
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
        encoder->setComputePipelineState(pipeline);
        return encoder;
    }
    static void commitWait(MTL::CommandBuffer* commandBuffer, const char* stage) {
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            auto* error = commandBuffer->error();
            std::cerr << stage << " cmdbuf: "
                      << (error ? error->localizedDescription()->utf8String() : "unknown") << "\n";
        }
    }
    //out = a + b, elementwise over one activation-sized tensor
    void encodeResidualAdd(MTL::CommandBuffer* commandBuffer, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, residualAddPipeline);
        encoder->setBuffer(a, 0, 0);
        encoder->setBuffer(b, 0, 1);
        encoder->setBuffer(out, 0, 2);
        encoder->setBytes(&activationCount, sizeof(activationCount), 3);
        encoder->dispatchThreads(MTL::Size(activationCount, 1, 1), MTL::Size(1024, 1, 1));
        encoder->endEncoding();
    }
    void encodeLayernormForward(MTL::CommandBuffer* commandBuffer, LayernormBuffers& ln, MTL::Buffer* xBuffer) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, layernormFwdPipeline);
        encoder->setBuffer(xBuffer, 0, 0);
        encoder->setBuffer(ln.xnormBuffer, 0, 1);
        encoder->setBuffer(ln.outBuffer, 0, 2);
        encoder->setBuffer(ln.gammaBuffer, 0, 3);
        encoder->setBuffer(ln.betaBuffer, 0, 4);
        encoder->setBuffer(ln.stdevBuffer, 0, 5);
        encoder->setBytes(&layernormParams, sizeof(layernormParams), 6);
        encoder->dispatchThreadgroups(MTL::Size(layernormParams.rows, 1, 1), MTL::Size(layernormParams.group_size, 1, 1));
        encoder->endEncoding();
    }
    //dZBuffer is rewritten in place with dX
    void encodeLayernormBackward(MTL::CommandBuffer* commandBuffer, LayernormBuffers& ln, MTL::Buffer* dZBuffer) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, layernormBwdPipeline);
        encoder->setBuffer(dZBuffer, 0, 0);
        encoder->setBuffer(ln.xnormBuffer, 0, 1);
        encoder->setBuffer(ln.gammaBuffer, 0, 2);
        encoder->setBuffer(ln.stdevBuffer, 0, 3);
        encoder->setBuffer(dXnormBuffer, 0, 5);
        encoder->setBuffer(ln.dGammaBuffer, 0, 6);
        encoder->setBuffer(ln.dBetaBuffer, 0, 7);
        encoder->setBytes(&layernormParams, sizeof(layernormParams), 8);
        encoder->dispatchThreadgroups(MTL::Size(layernormParams.rows, 1, 1), MTL::Size(layernormParams.group_size, 1, 1));
        encoder->endEncoding();
    }

    //---------------------------------------------------------------- forward
    //weights are GPU-resident (uploaded once at construction, updated in
    //place by step()). The host fills obs() before the call; the head output
    //lands in logitsBuffer.
    void forward() {
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

        //obs projection
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, inputFwdPipeline);
            encoder->setBuffer(obsBuffer, 0, 0);
            encoder->setBuffer(inWBuffer, 0, 1);
            encoder->setBuffer(inBBuffer, 0, 2);
            encoder->setBuffer(inputBuffer, 0, 3);
            encoder->setBytes(&inputParams, sizeof(inputParams), 4);
            encoder->dispatchThreads(MTL::Size(embedDim, parallels, 1),
                                     MTL::Size(std::min<uint32_t>(embedDim, 256), 1, 1));
            encoder->endEncoding();
        }

        MTL::Buffer* hiddenBuffer = inputBuffer;
        for (uint32_t n = 0; n < nLayers; ++n) {
            BlockBuffers& block = blocks[n];

            //ln -> mlp (stage 1: V = relu(X @ Up), stage 2: out = V @ Dp) -> residual
            encodeLayernormForward(commandBuffer, block.ln, hiddenBuffer);
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpFwdOnePipeline);
                encoder->setBuffer(block.ln.outBuffer, 0, 0);
                encoder->setBuffer(block.upprojBuffer, 0, 1);
                encoder->setBuffer(block.mlpVBuffer, 0, 2);
                encoder->setBuffer(block.mlpMaskBuffer, 0, 3);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 4);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, hiddenDim / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpFwdTwoPipeline);
                encoder->setBuffer(block.downprojBuffer, 0, 0);
                encoder->setBuffer(block.mlpVBuffer, 0, 1);
                encoder->setBuffer(block.mlpOutBuffer, 0, 2);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 3);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, tileCols, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeResidualAdd(commandBuffer, hiddenBuffer, block.mlpOutBuffer, block.residualBuffer);
            hiddenBuffer = block.residualBuffer;
        }

        //final layernorm -> linear head
        encodeLayernormForward(commandBuffer, lnFinal, hiddenBuffer);
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, linearFwdPipeline);
            encoder->setBuffer(lnFinal.outBuffer, 0, 0);
            encoder->setBuffer(linearWBuffer, 0, 1);
            encoder->setBuffer(logitsBuffer, 0, 2);
            encoder->setBytes(&linearParams, sizeof(linearParams), 3);
            encoder->dispatchThreadgroups(MTL::Size(tileRows, HEAD_DIM / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "forward");
    }

    //-------------------------------------------------------------------- act
    //sample one action per environment from the head of the last forward().
    //Fills actions(), logProbs(), and values(). The log prob is computed in
    //float from the same bf16 head the loss kernel reads, so the PPO ratio
    //starts at 1 for on-policy data.
    template <typename Rng>
    void act(Rng& rng) {
        const __bf16* head = static_cast<const __bf16*>(logitsBuffer->contents());
        typename Env::Action* action = actions();
        float* value = values();
        float* logProb = logProbs();
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        float logits[Env::ACT_DIM];
        for (uint32_t row = 0; row < parallels; ++row) {
            const size_t base = (size_t)row * HEAD_DIM;
            float rowMax = -INFINITY;
            for (uint32_t a = 0; a < Env::ACT_DIM; ++a) {
                logits[a] = (float)head[base + a];
                rowMax = std::max(rowMax, logits[a]);
            }
            float sum = 0.0f;
            for (uint32_t a = 0; a < Env::ACT_DIM; ++a) sum += std::exp(logits[a] - rowMax);
            const float lse = rowMax + std::log(sum);
            //inverse CDF over the unnormalized weights exp(l - max)
            const float u = uniform(rng) * sum;
            uint32_t chosen = Env::ACT_DIM - 1;
            float acc = 0.0f;
            for (uint32_t a = 0; a < Env::ACT_DIM; ++a) {
                acc += std::exp(logits[a] - rowMax);
                if (u < acc) { chosen = a; break; }
            }
            action[row] = (typename Env::Action)chosen;
            logProb[row] = logits[chosen] - lse;
            value[row] = (float)head[base + Env::ACT_DIM];
        }
    }
    //argmax action per environment, for evaluation
    void actGreedy() {
        typename Env::Action* action = actions();
        for (uint32_t row = 0; row < parallels; ++row) {
            uint32_t best = 0;
            for (uint32_t a = 1; a < Env::ACT_DIM; ++a) {
                if (headValue(row, a) > headValue(row, best)) best = a;
            }
            action[row] = (typename Env::Action)best;
        }
    }

    //--------------------------------------------------------------- backward
    //PPO update for the batch whose obs were run through the last forward().
    //The four arrays hold one entry per row: the action taken, the GAE
    //advantage, the return (value target), and the log prob recorded by act()
    //at collection time. Returns the batch-mean loss terms.
    LossStats backward(const typename Env::Action* batchActions, const float* advantages,
                       const float* returns, const float* oldLogProbs) {
        std::memcpy(batchActionBuffer->contents(), batchActions, (size_t)parallels * sizeof(uint32_t));
        std::memcpy(advantageBuffer->contents(), advantages, (size_t)parallels * sizeof(float));
        std::memcpy(returnBuffer->contents(), returns, (size_t)parallels * sizeof(float));
        std::memcpy(oldLogProbBuffer->contents(), oldLogProbs, (size_t)parallels * sizeof(float));

        //zero every accumulated gradient before the pass. dInW/dInB are
        //written whole by inputBackward and need no zeroing.
        std::memset(dLinearWBuffer->contents(), 0, (size_t)embedDim * HEAD_DIM * sizeof(__bf16));
        std::memset(lnFinal.dGammaBuffer->contents(), 0, gammaFloatBytes);
        std::memset(lnFinal.dBetaBuffer->contents(), 0, gammaFloatBytes);
        for (BlockBuffers& block : blocks) {
            std::memset(block.ln.dGammaBuffer->contents(), 0, gammaFloatBytes);
            std::memset(block.ln.dBetaBuffer->contents(), 0, gammaFloatBytes);
            std::memset(block.dUpBuffer->contents(), 0, (size_t)embedDim * hiddenDim * sizeof(__bf16));
            std::memset(block.dDpBuffer->contents(), 0, (size_t)hiddenDim * embedDim * sizeof(__bf16));
        }

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

        //loss -> dLogits, per-row stats
        {
            PolicyParams policyParams{parallels, Env::ACT_DIM, HEAD_DIM, clip, valueCoef, entropyCoef};
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, policyLossPipeline);
            encoder->setBuffer(logitsBuffer, 0, 0);
            encoder->setBuffer(batchActionBuffer, 0, 1);
            encoder->setBuffer(advantageBuffer, 0, 2);
            encoder->setBuffer(returnBuffer, 0, 3);
            encoder->setBuffer(oldLogProbBuffer, 0, 4);
            encoder->setBuffer(dLogitsBuffer, 0, 5);
            encoder->setBuffer(statsBuffer, 0, 6);
            encoder->setBytes(&policyParams, sizeof(policyParams), 7);
            encoder->dispatchThreads(MTL::Size(parallels, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }

        //linear head -> final layernorm
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, linearBwdPipeline);
            encoder->setBuffer(lnFinal.outBuffer, 0, 0);
            encoder->setBuffer(linearWBuffer, 0, 1);
            encoder->setBuffer(dLogitsBuffer, 0, 2);
            encoder->setBuffer(dStreamBuffer, 0, 3);
            encoder->setBuffer(dLinearWBuffer, 0, 4);
            encoder->setBytes(&linearParams, sizeof(linearParams), 5);
            encoder->dispatchThreadgroups(MTL::Size(tileRows, 1, 1), MTL::Size(32 * simdGroups, 1, 1));
            encoder->endEncoding();
        }
        encodeLayernormBackward(commandBuffer, lnFinal, dStreamBuffer);

        for (int n = (int)nLayers - 1; n >= 0; --n) {
            BlockBuffers& block = blocks[n];

            //mlp backward: dStream is dZ; stage 1 makes the masked dV, stage 2
            //produces dUp/dDp and lands the branch gradient dX in dBranch
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpBwdOnePipeline);
                encoder->setBuffer(block.downprojBuffer, 0, 1);
                encoder->setBuffer(block.mlpMaskBuffer, 0, 2);
                encoder->setBuffer(dStreamBuffer, 0, 3);
                encoder->setBuffer(dMlpVBuffer, 0, 5);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 6);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, hiddenDim / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpBwdTwoPipeline);
                encoder->setBuffer(block.ln.outBuffer, 0, 0);
                encoder->setBuffer(dBranchBuffer, 0, 1);
                encoder->setBuffer(dMlpVBuffer, 0, 2);
                encoder->setBuffer(block.upprojBuffer, 0, 3);
                encoder->setBuffer(block.dUpBuffer, 0, 4);
                encoder->setBuffer(block.mlpVBuffer, 0, 5);
                encoder->setBuffer(dStreamBuffer, 0, 6);
                encoder->setBuffer(block.dDpBuffer, 0, 7);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 8);
                //covers both tilings inside the kernel: hidden strips for dUp/dDp, M x N tiles for dX
                encoder->dispatchThreadgroups(MTL::Size(std::max(hiddenDim / tileM, tileRows * tileCols), 1, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeLayernormBackward(commandBuffer, block.ln, dBranchBuffer);
            encodeResidualAdd(commandBuffer, dBranchBuffer, dStreamBuffer, dStreamBuffer); //skip connection
        }

        //obs projection: dStream is the gradient at the stream start
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, inputBwdPipeline);
            encoder->setBuffer(obsBuffer, 0, 0);
            encoder->setBuffer(dStreamBuffer, 0, 1);
            encoder->setBuffer(dInWBuffer, 0, 2);
            encoder->setBuffer(dInBBuffer, 0, 3);
            encoder->setBytes(&inputParams, sizeof(inputParams), 4);
            encoder->dispatchThreads(MTL::Size((Env::OBS_DIM + 1) * embedDim, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "backward");
        //gradients stay on the GPU; step() gathers them straight from the
        //per-parameter buffers

        LossStats stats{};
        const float* rows = static_cast<const float*>(statsBuffer->contents());
        for (uint32_t row = 0; row < parallels; ++row) {
            stats.policy += rows[row * 5 + 0];
            stats.value += rows[row * 5 + 1];
            stats.entropy += rows[row * 5 + 2];
            stats.approxKl += rows[row * 5 + 3];
            stats.clipFrac += rows[row * 5 + 4];
        }
        const float inv = 1.0f / (float)parallels;
        stats.policy *= inv;
        stats.value *= inv;
        stats.entropy *= inv;
        stats.approxKl *= inv;
        stats.clipFrac *= inv;
        return stats;
    }

    //------------------------------------------------------------------- step
    std::vector<__bf16>& addParam(size_t count, MTL::Buffer* weightBuffer, MTL::Buffer* gradBuffer, bool gradIsFloat) {
        optWeights.emplace_back(count, (__bf16)0.0f);
        optWeightBuffers.push_back(weightBuffer);
        optGradBuffers.push_back(gradBuffer);
        optGradIsFloat.push_back(gradIsFloat);
        return optWeights.back();
    }
    //gather each parameter's gradient into the flat buffer, clip by global
    //norm on the host, run Adam on the flat master params, scatter the
    //updated master back to the bf16 weight buffers the next forward reads.
    //Returns the pre-clip gradient norm.
    float step() {
        stepCounter += 1.0f;

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        for (size_t i = 0; i < optWeights.size(); ++i) {
            FlatCopyParams copyParams{(uint32_t)optWeights[i].size(), (uint32_t)optOffsets[i]};
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer,
                optGradIsFloat[i] ? gatherFloatPipeline : gatherBfloatPipeline);
            encoder->setBuffer(optGradBuffers[i], 0, 0);
            encoder->setBuffer(gradientBuffer, 0, 1);
            encoder->setBytes(&copyParams, sizeof(copyParams), 2);
            encoder->dispatchThreads(MTL::Size(copyParams.count, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "gather grads");

        //global norm over the flat gradient; the buffer is shared so the host reads it directly
        const float* grads = static_cast<const float*>(gradientBuffer->contents());
        double sumSq = 0.0;
        for (size_t i = 0; i < optCount; ++i) sumSq += (double)grads[i] * grads[i];
        const float gradNorm = (float)std::sqrt(sumSq);

        AdamParams adamParams;
        adamParams.count = (uint32_t)optCount;
        adamParams.lr = lr;
        adamParams.beta1Pow = std::pow(adamParams.beta1, stepCounter);
        adamParams.beta2Pow = std::pow(adamParams.beta2, stepCounter);
        if (maxGradNorm > 0.0f && gradNorm > maxGradNorm) adamParams.gradScale = maxGradNorm / (gradNorm + 1e-6f);

        commandBuffer = commandQueue->commandBuffer();
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, stepPipeline);
            encoder->setBuffer(parameterBuffer, 0, 0);
            encoder->setBuffer(gradientBuffer, 0, 1);
            encoder->setBuffer(momentumBuffer, 0, 2);
            encoder->setBuffer(varianceBuffer, 0, 3);
            encoder->setBytes(&adamParams, sizeof(adamParams), 4);
            encoder->dispatchThreads(MTL::Size(optCount, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        for (size_t i = 0; i < optWeights.size(); ++i) {
            FlatCopyParams copyParams{(uint32_t)optWeights[i].size(), (uint32_t)optOffsets[i]};
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, scatterPipeline);
            encoder->setBuffer(parameterBuffer, 0, 0);
            encoder->setBuffer(optWeightBuffers[i], 0, 1);
            encoder->setBytes(&copyParams, sizeof(copyParams), 2);
            encoder->dispatchThreads(MTL::Size(copyParams.count, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "adam step");
        return gradNorm;
    }

    ~Model() {
        auto releaseLayernorm = [](LayernormBuffers& ln) {
            ln.gammaBuffer->release();
            ln.betaBuffer->release();
            ln.xnormBuffer->release();
            ln.stdevBuffer->release();
            ln.outBuffer->release();
            ln.dGammaBuffer->release();
            ln.dBetaBuffer->release();
        };
        for (BlockBuffers& block : blocks) {
            releaseLayernorm(block.ln);
            block.upprojBuffer->release();
            block.downprojBuffer->release();
            block.mlpVBuffer->release();
            block.mlpMaskBuffer->release();
            block.mlpOutBuffer->release();
            block.residualBuffer->release();
            block.dUpBuffer->release();
            block.dDpBuffer->release();
        }
        releaseLayernorm(lnFinal);
        obsBuffer->release();
        actionBuffer->release();
        valueBuffer->release();
        logProbBuffer->release();
        inWBuffer->release();
        inBBuffer->release();
        inputBuffer->release();
        dInWBuffer->release();
        dInBBuffer->release();
        linearWBuffer->release();
        logitsBuffer->release();
        dLogitsBuffer->release();
        dLinearWBuffer->release();
        batchActionBuffer->release();
        advantageBuffer->release();
        returnBuffer->release();
        oldLogProbBuffer->release();
        statsBuffer->release();
        dStreamBuffer->release();
        dBranchBuffer->release();
        dXnormBuffer->release();
        dMlpVBuffer->release();
        parameterBuffer->release();
        gradientBuffer->release();
        momentumBuffer->release();
        varianceBuffer->release();
        inputFwdPipeline->release();
        inputBwdPipeline->release();
        layernormFwdPipeline->release();
        layernormBwdPipeline->release();
        mlpFwdOnePipeline->release();
        mlpFwdTwoPipeline->release();
        mlpBwdOnePipeline->release();
        mlpBwdTwoPipeline->release();
        linearFwdPipeline->release();
        linearBwdPipeline->release();
        policyLossPipeline->release();
        residualAddPipeline->release();
        stepPipeline->release();
        gatherFloatPipeline->release();
        gatherBfloatPipeline->release();
        scatterPipeline->release();
        commandQueue->release();
        pool->release();
        device->release();
    }
};
