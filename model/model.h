//Actor-critic policy shared by every env row. Borrows the gym's device, queue,
//rng, and obs buffer. Manual path: forward(), act(rng), gym.run(). Fused path:
//collect() runs T steps + GAE in one command buffer; trainSlot(t) runs one PPO
//minibatch (forward, loss, backward, Adam) in one command buffer.
//Head row = [logits(ACT_DIM) | value | zero pad] in one 64-wide tile.
//embedDim must equal N_EMBED_CFG (model/config.h). Include after gym.h.
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
struct SampleParams {
    uint32_t rows;
    uint32_t actDim;
    uint32_t headDim;
};
struct GaeParams {
    uint32_t rows;
    uint32_t horizon;
    uint32_t headDim;
    uint32_t actDim;
    float gamma;
    float lambda;
};
struct NormalizeParams {
    uint32_t rows;
};
struct AdamParams {
    uint32_t count;
    float lr;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float beta1Pow;
    float beta2Pow;
    float maxGradNorm;
};

struct LossStats {
    float policy;
    float value;
    float entropy;
    float approxKl;
    float clipFrac;
    float gradNorm;
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
        MTL::Buffer* stdevBuffer;
        MTL::Buffer* outBuffer;
        MTL::Buffer* dGammaBuffer;
        MTL::Buffer* dBetaBuffer;
    };
    struct BlockBuffers {
        LayernormBuffers ln;
        MTL::Buffer* upprojBuffer;
        MTL::Buffer* downprojBuffer;
        MTL::Buffer* mlpVBuffer;
        MTL::Buffer* mlpMaskBuffer;
        MTL::Buffer* mlpOutBuffer;
        MTL::Buffer* residualBuffer;
        MTL::Buffer* dUpBuffer;
        MTL::Buffer* dDpBuffer;
    };

    Gym<Env>& gym;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;

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
    MTL::ComputePipelineState* samplePipeline;
    MTL::ComputePipelineState* gaePipeline;
    MTL::ComputePipelineState* normalizePipeline;
    MTL::ComputePipelineState* residualAddPipeline;
    MTL::ComputePipelineState* gradNormPipeline;
    MTL::ComputePipelineState* stepPipeline;
    MTL::ComputePipelineState* gatherFloatPipeline;
    MTL::ComputePipelineState* gatherBfloatPipeline;
    MTL::ComputePipelineState* scatterPipeline;

    uint32_t parallels;
    uint32_t horizon;
    uint32_t embedDim;
    uint32_t mlpScale;
    uint32_t nLayers;
    float lr;
    uint32_t hiddenDim;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t activationCount;
    static constexpr uint32_t tileM = 64;
    static constexpr uint32_t tileN = 64;
    static constexpr uint32_t simdGroups = 4;

    float clip = 0.2f;
    float valueCoef = 0.5f;
    float entropyCoef = 0.01f;
    float maxGradNorm = 0.5f; //0 disables clipping
    float gamma = 0.99f;
    float lambda = 0.95f;

    InputParams inputParams;
    LayernormParams layernormParams;
    mlpParams mlpParameters;
    LinearParams linearParams;
    SampleParams sampleParams;
    NormalizeParams normalizeParams;

    size_t activationBytes;
    size_t hiddenBytes;
    size_t gammaFloatBytes;
    size_t obsSlotBytes;
    size_t rowFloatBytes;

    MTL::Buffer* valueBuffer;
    MTL::Buffer* logProbBuffer;

    MTL::Buffer* obsSlots;
    MTL::Buffer* actionSlots;
    MTL::Buffer* logProbSlots;
    MTL::Buffer* valueSlots;
    MTL::Buffer* rewardSlots;
    MTL::Buffer* doneSlots;
    MTL::Buffer* advantageSlots;
    MTL::Buffer* returnSlots;
    MTL::Buffer* advNormBuffer;
    bool rolloutStarted = false;
    std::vector<uint32_t> slotOrder;
    std::mt19937 hostRng;

    MTL::Buffer* inWBuffer;
    MTL::Buffer* inBBuffer;
    MTL::Buffer* inputBuffer;
    MTL::Buffer* dInWBuffer;
    MTL::Buffer* dInBBuffer;

    std::vector<BlockBuffers> blocks;

    LayernormBuffers lnFinal;
    MTL::Buffer* linearWBuffer;
    MTL::Buffer* logitsBuffer;
    MTL::Buffer* dLogitsBuffer;
    MTL::Buffer* dLinearWBuffer;
    MTL::Buffer* statsBuffer;

    MTL::Buffer* dStreamBuffer;
    MTL::Buffer* dBranchBuffer;
    MTL::Buffer* dXnormBuffer;
    MTL::Buffer* dMlpVBuffer;

    struct FlatCopyParams { uint32_t count; uint32_t offset; };
    std::vector<std::vector<__bf16>> optWeights;
    std::vector<MTL::Buffer*> optWeightBuffers;
    std::vector<MTL::Buffer*> optGradBuffers;
    std::vector<bool> optGradIsFloat;
    std::vector<size_t> optOffsets;
    size_t optCount = 0;
    float stepCounter = 0.0f;
    MTL::Buffer* parameterBuffer;
    MTL::Buffer* gradientBuffer;
    MTL::Buffer* momentumBuffer;
    MTL::Buffer* varianceBuffer;
    MTL::Buffer* normSqBuffer;

    Model(Gym<Env>& g, int e, int s, int layers, float learningRate, int T = 32, uint32_t seed = 0)
        : gym(g), parallels(g.parallels), horizon((uint32_t)T), embedDim(e), mlpScale(s), nLayers(layers), lr(learningRate),
          hiddenDim(e * s), tileRows(g.parallels / tileM), tileCols(e / tileN),
          activationCount((uint32_t)(g.parallels * e)),
          inputParams{(uint32_t)g.parallels, Env::OBS_DIM, (uint32_t)e},
          layernormParams(g.parallels, e),
          mlpParameters{(uint32_t)g.parallels, (uint32_t)e, (uint32_t)s},
          linearParams{(uint32_t)g.parallels, HEAD_DIM, (uint32_t)e},
          sampleParams{(uint32_t)g.parallels, Env::ACT_DIM, HEAD_DIM},
          normalizeParams{(uint32_t)g.parallels},
          activationBytes((size_t)g.parallels * e * sizeof(__bf16)),
          hiddenBytes((size_t)g.parallels * e * s * sizeof(__bf16)),
          gammaFloatBytes((size_t)e * sizeof(float)),
          obsSlotBytes((size_t)g.parallels * sizeof(typename Env::Obs)),
          rowFloatBytes((size_t)g.parallels * sizeof(float)) {
        if (parallels % tileM != 0 || e % tileN != 0 || e > 1024 || hiddenDim % tileN != 0 || layers < 1 || T < 1) {
            std::cerr << "Model: rows and embedDim must be multiples of 64, embedDim <= 1024, layers >= 1, horizon >= 1\n";
            std::exit(1);
        }
        if (e != N_EMBED_CFG) {
            std::cerr << "Model: embedDim " << e << " != N_EMBED_CFG " << N_EMBED_CFG
                      << " from model/config.h\n";
            std::exit(1);
        }
        device = gym.device;
        commandQueue = gym.commandQueue;
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
        samplePipeline = makePipeline(library, "sampleAction");
        gaePipeline = makePipeline(library, "gaeBackward");
        normalizePipeline = makePipeline(library, "normalizeAdvantage");
        residualAddPipeline = makePipeline(library, "residualAdd");
        gradNormPipeline = makePipeline(library, "gradNormSq");
        stepPipeline = makePipeline(library, "adamStep");
        gatherFloatPipeline = makePipeline(library, "gatherGradFloat");
        gatherBfloatPipeline = makePipeline(library, "gatherGradBfloat");
        scatterPipeline = makePipeline(library, "scatterWeight");
        library->release();

        valueBuffer = newSharedBuffer(rowFloatBytes);
        logProbBuffer = newSharedBuffer(rowFloatBytes);

        const size_t slotCount = (size_t)horizon * parallels;
        obsSlots = newSharedBuffer((size_t)(horizon + 1) * obsSlotBytes);
        actionSlots = newSharedBuffer(slotCount * sizeof(uint32_t));
        logProbSlots = newSharedBuffer(slotCount * sizeof(float));
        valueSlots = newSharedBuffer(slotCount * sizeof(float));
        rewardSlots = newSharedBuffer(slotCount * sizeof(float));
        doneSlots = newSharedBuffer(slotCount * sizeof(uint8_t));
        advantageSlots = newSharedBuffer(slotCount * sizeof(float));
        returnSlots = newSharedBuffer(slotCount * sizeof(float));
        advNormBuffer = newSharedBuffer(rowFloatBytes);
        std::memset(actionSlots->contents(), 0, slotCount * sizeof(uint32_t));
        slotOrder.resize(horizon);
        for (uint32_t t = 0; t < horizon; t++) slotOrder[t] = t;
        hostRng.seed(seed);

        inWBuffer = newSharedBuffer((size_t)Env::OBS_DIM * embedDim * sizeof(__bf16));
        inBBuffer = newSharedBuffer((size_t)embedDim * sizeof(__bf16));
        inputBuffer = newSharedBuffer(activationBytes);
        dInWBuffer = newSharedBuffer((size_t)Env::OBS_DIM * embedDim * sizeof(float));
        dInBBuffer = newSharedBuffer((size_t)embedDim * sizeof(float));

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

        allocLayernorm(lnFinal);
        linearWBuffer = newSharedBuffer((size_t)embedDim * HEAD_DIM * sizeof(__bf16));
        logitsBuffer = newSharedBuffer((size_t)parallels * HEAD_DIM * sizeof(__bf16));
        dLogitsBuffer = newSharedBuffer((size_t)parallels * HEAD_DIM * sizeof(__bf16));
        dLinearWBuffer = newSharedBuffer((size_t)embedDim * HEAD_DIM * sizeof(__bf16));
        statsBuffer = newSharedBuffer((size_t)parallels * 5 * sizeof(float));

        dStreamBuffer = newSharedBuffer(activationBytes);
        dBranchBuffer = newSharedBuffer(activationBytes);
        dXnormBuffer = newSharedBuffer(activationBytes);
        dMlpVBuffer = newSharedBuffer(hiddenBytes);

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
            normal(addParam((size_t)hiddenDim * embedDim, block.downprojBuffer, block.dDpBuffer, false),
                   1.0f / std::sqrt((float)hiddenDim) / std::sqrt((float)nLayers));
        }
        constant(addParam((size_t)embedDim, lnFinal.gammaBuffer, lnFinal.dGammaBuffer, true), 1.0f);
        constant(addParam((size_t)embedDim, lnFinal.betaBuffer, lnFinal.dBetaBuffer, true), 0.0f);
        {
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
        normSqBuffer = newSharedBuffer(4 * sizeof(float));

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

    float* values() { return static_cast<float*>(valueBuffer->contents()); }
    float* logProbs() { return static_cast<float*>(logProbBuffer->contents()); }
    float headValue(uint32_t row, uint32_t col) const {
        return (float)static_cast<const __bf16*>(logitsBuffer->contents())[(size_t)row * HEAD_DIM + col];
    }
    typename Env::Obs* slotObs(uint32_t t) { return reinterpret_cast<typename Env::Obs*>(static_cast<char*>(obsSlots->contents()) + t * obsSlotBytes); }
    typename Env::Action* slotActions(uint32_t t) { return static_cast<typename Env::Action*>(actionSlots->contents()) + (size_t)t * parallels; }
    float* slotRewards(uint32_t t) { return static_cast<float*>(rewardSlots->contents()) + (size_t)t * parallels; }
    uint8_t* slotDones(uint32_t t) { return static_cast<uint8_t*>(doneSlots->contents()) + (size_t)t * parallels; }
    float* slotValues(uint32_t t) { return static_cast<float*>(valueSlots->contents()) + (size_t)t * parallels; }
    float* slotAdvantages(uint32_t t) { return static_cast<float*>(advantageSlots->contents()) + (size_t)t * parallels; }
    float* slotReturns(uint32_t t) { return static_cast<float*>(returnSlots->contents()) + (size_t)t * parallels; }

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
        ln.stdevBuffer = newSharedBuffer(rowFloatBytes);
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
    static void encodeZero(MTL::CommandBuffer* commandBuffer, MTL::Buffer* buffer, size_t bytes) {
        MTL::BlitCommandEncoder* blit = commandBuffer->blitCommandEncoder();
        blit->fillBuffer(buffer, NS::Range::Make(0, bytes), 0);
        blit->endEncoding();
    }
    static void encodeCopy(MTL::CommandBuffer* commandBuffer, MTL::Buffer* src, size_t srcOffset,
                           MTL::Buffer* dst, size_t dstOffset, size_t bytes) {
        MTL::BlitCommandEncoder* blit = commandBuffer->blitCommandEncoder();
        blit->copyFromBuffer(src, srcOffset, dst, dstOffset, bytes);
        blit->endEncoding();
    }
    inline size_t obsOffset(uint32_t t) const { return (size_t)t * obsSlotBytes; }
    inline size_t rowOffset(uint32_t t) const { return (size_t)t * rowFloatBytes; }
    inline size_t doneOffset(uint32_t t) const { return (size_t)t * parallels; }

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

    void encodeForward(MTL::CommandBuffer* commandBuffer, MTL::Buffer* obsBuffer, size_t obsByteOffset) {
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, inputFwdPipeline);
            encoder->setBuffer(obsBuffer, obsByteOffset, 0);
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
    }

    void forward() {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        encodeForward(commandBuffer, gym.obsBuffer, 0);
        commitWait(commandBuffer, "forward");
        pool->release();
    }

    template <typename Rng>
    void act(Rng& rng) {
        const __bf16* head = static_cast<const __bf16*>(logitsBuffer->contents());
        typename Env::Action* action = gym.actions();
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
    void actGreedy() {
        typename Env::Action* action = gym.actions();
        for (uint32_t row = 0; row < parallels; ++row) {
            uint32_t best = 0;
            for (uint32_t a = 1; a < Env::ACT_DIM; ++a) {
                if (headValue(row, a) > headValue(row, best)) best = a;
            }
            action[row] = (typename Env::Action)best;
        }
    }
    void encodeSample(MTL::CommandBuffer* commandBuffer, uint32_t t) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, samplePipeline);
        encoder->setBuffer(logitsBuffer, 0, 0);
        encoder->setBuffer(gym.rngBuffer, 0, 1);
        encoder->setBuffer(actionSlots, rowOffset(t), 2);
        encoder->setBuffer(logProbSlots, rowOffset(t), 3);
        encoder->setBuffer(valueSlots, rowOffset(t), 4);
        encoder->setBytes(&sampleParams, sizeof(sampleParams), 5);
        encoder->dispatchThreads(MTL::Size(parallels, 1, 1), MTL::Size(256, 1, 1));
        encoder->endEncoding();
    }

    void collect() {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

        if (!rolloutStarted) {
            gym.encodeStep(commandBuffer, actionSlots, 0, obsSlots, obsOffset(0), rewardSlots, 0);
            rolloutStarted = true;
        } else {
            encodeCopy(commandBuffer, obsSlots, obsOffset(horizon), obsSlots, obsOffset(0), obsSlotBytes);
        }

        for (uint32_t t = 0; t < horizon; ++t) {
            encodeForward(commandBuffer, obsSlots, obsOffset(t));
            encodeSample(commandBuffer, t);
            gym.encodeStep(commandBuffer, actionSlots, rowOffset(t), obsSlots, obsOffset(t + 1), rewardSlots, rowOffset(t));
            //the rollout kernel reads and writes the gym's live done flags; keep a copy per slot
            encodeCopy(commandBuffer, gym.doneBuffer, 0, doneSlots, doneOffset(t), parallels);
        }

        encodeForward(commandBuffer, obsSlots, obsOffset(horizon));
        {
            GaeParams gaeParams{parallels, horizon, HEAD_DIM, Env::ACT_DIM, gamma, lambda};
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, gaePipeline);
            encoder->setBuffer(rewardSlots, 0, 0);
            encoder->setBuffer(doneSlots, 0, 1);
            encoder->setBuffer(valueSlots, 0, 2);
            encoder->setBuffer(logitsBuffer, 0, 3);
            encoder->setBuffer(advantageSlots, 0, 4);
            encoder->setBuffer(returnSlots, 0, 5);
            encoder->setBytes(&gaeParams, sizeof(gaeParams), 6);
            encoder->dispatchThreads(MTL::Size(parallels, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        commitWait(commandBuffer, "collect");
        pool->release();
    }

    void encodeLoss(MTL::CommandBuffer* commandBuffer, uint32_t t) {
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, normalizePipeline);
            encoder->setBuffer(advantageSlots, rowOffset(t), 0);
            encoder->setBuffer(advNormBuffer, 0, 1);
            encoder->setBytes(&normalizeParams, sizeof(normalizeParams), 2);
            encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(std::min<uint32_t>(parallels, 1024), 1, 1));
            encoder->endEncoding();
        }
        {
            PolicyParams policyParams{parallels, Env::ACT_DIM, HEAD_DIM, clip, valueCoef, entropyCoef};
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, policyLossPipeline);
            encoder->setBuffer(logitsBuffer, 0, 0);
            encoder->setBuffer(actionSlots, rowOffset(t), 1);
            encoder->setBuffer(advNormBuffer, 0, 2);
            encoder->setBuffer(returnSlots, rowOffset(t), 3);
            encoder->setBuffer(logProbSlots, rowOffset(t), 4);
            encoder->setBuffer(dLogitsBuffer, 0, 5);
            encoder->setBuffer(statsBuffer, 0, 6);
            encoder->setBytes(&policyParams, sizeof(policyParams), 7);
            encoder->dispatchThreads(MTL::Size(parallels, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
    }

    void encodeBackward(MTL::CommandBuffer* commandBuffer, MTL::Buffer* obsBuffer, size_t obsByteOffset) {
        encodeZero(commandBuffer, dLinearWBuffer, (size_t)embedDim * HEAD_DIM * sizeof(__bf16));
        encodeZero(commandBuffer, lnFinal.dGammaBuffer, gammaFloatBytes);
        encodeZero(commandBuffer, lnFinal.dBetaBuffer, gammaFloatBytes);
        for (BlockBuffers& block : blocks) {
            encodeZero(commandBuffer, block.ln.dGammaBuffer, gammaFloatBytes);
            encodeZero(commandBuffer, block.ln.dBetaBuffer, gammaFloatBytes);
            encodeZero(commandBuffer, block.dUpBuffer, (size_t)embedDim * hiddenDim * sizeof(__bf16));
            encodeZero(commandBuffer, block.dDpBuffer, (size_t)hiddenDim * embedDim * sizeof(__bf16));
        }

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
                encoder->dispatchThreadgroups(MTL::Size(std::max(hiddenDim / tileM, tileRows * tileCols), 1, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeLayernormBackward(commandBuffer, block.ln, dBranchBuffer);
            encodeResidualAdd(commandBuffer, dBranchBuffer, dStreamBuffer, dStreamBuffer);
        }

        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, inputBwdPipeline);
            encoder->setBuffer(obsBuffer, obsByteOffset, 0);
            encoder->setBuffer(dStreamBuffer, 0, 1);
            encoder->setBuffer(dInWBuffer, 0, 2);
            encoder->setBuffer(dInBBuffer, 0, 3);
            encoder->setBytes(&inputParams, sizeof(inputParams), 4);
            encoder->dispatchThreads(MTL::Size((Env::OBS_DIM + 1) * embedDim, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
    }

    std::vector<__bf16>& addParam(size_t count, MTL::Buffer* weightBuffer, MTL::Buffer* gradBuffer, bool gradIsFloat) {
        optWeights.emplace_back(count, (__bf16)0.0f);
        optWeightBuffers.push_back(weightBuffer);
        optGradBuffers.push_back(gradBuffer);
        optGradIsFloat.push_back(gradIsFloat);
        return optWeights.back();
    }
    void encodeStep(MTL::CommandBuffer* commandBuffer) {
        stepCounter += 1.0f;
        AdamParams adamParams;
        adamParams.count = (uint32_t)optCount;
        adamParams.lr = lr;
        adamParams.beta1Pow = std::pow(adamParams.beta1, stepCounter);
        adamParams.beta2Pow = std::pow(adamParams.beta2, stepCounter);
        adamParams.maxGradNorm = maxGradNorm;

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
        encodeZero(commandBuffer, normSqBuffer, 4 * sizeof(float));
        {
            uint32_t count = (uint32_t)optCount;
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, gradNormPipeline);
            encoder->setBuffer(gradientBuffer, 0, 0);
            encoder->setBuffer(normSqBuffer, 0, 1);
            encoder->setBytes(&count, sizeof(count), 2);
            encoder->dispatchThreads(MTL::Size((optCount + 31) / 32 * 32, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, stepPipeline);
            encoder->setBuffer(parameterBuffer, 0, 0);
            encoder->setBuffer(gradientBuffer, 0, 1);
            encoder->setBuffer(momentumBuffer, 0, 2);
            encoder->setBuffer(varianceBuffer, 0, 3);
            encoder->setBytes(&adamParams, sizeof(adamParams), 4);
            encoder->setBuffer(normSqBuffer, 0, 5);
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
    }

    void trainSlot(uint32_t t) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        encodeForward(commandBuffer, obsSlots, obsOffset(t));
        encodeLoss(commandBuffer, t);
        encodeBackward(commandBuffer, obsSlots, obsOffset(t));
        encodeStep(commandBuffer);
        commitWait(commandBuffer, "train");
        pool->release();
    }

    void train(uint32_t epochs = 4) {
        for (uint32_t e = 0; e < epochs; e++) {
            std::shuffle(slotOrder.begin(), slotOrder.end(), hostRng);
            for (uint32_t t : slotOrder) trainSlot(t);
        }
    }

    LossStats lastStats() const {
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
        stats.gradNorm = std::sqrt(static_cast<const float*>(normSqBuffer->contents())[0]);
        return stats;
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
        valueBuffer->release();
        logProbBuffer->release();
        obsSlots->release();
        actionSlots->release();
        logProbSlots->release();
        valueSlots->release();
        rewardSlots->release();
        doneSlots->release();
        advantageSlots->release();
        returnSlots->release();
        advNormBuffer->release();
        inWBuffer->release();
        inBBuffer->release();
        inputBuffer->release();
        dInWBuffer->release();
        dInBBuffer->release();
        linearWBuffer->release();
        logitsBuffer->release();
        dLogitsBuffer->release();
        dLinearWBuffer->release();
        statsBuffer->release();
        dStreamBuffer->release();
        dBranchBuffer->release();
        dXnormBuffer->release();
        dMlpVBuffer->release();
        parameterBuffer->release();
        gradientBuffer->release();
        momentumBuffer->release();
        varianceBuffer->release();
        normSqBuffer->release();
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
        samplePipeline->release();
        gaePipeline->release();
        normalizePipeline->release();
        residualAddPipeline->release();
        gradNormPipeline->release();
        stepPipeline->release();
        gatherFloatPipeline->release();
        gatherBfloatPipeline->release();
        scatterPipeline->release();
    }
};
