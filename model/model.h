//Actor-critic policy shared by every env row. Borrows the gym's device, queue,
//rng, and obs buffer. Manual path: forward(), act(rng), gym.run(). Fused path:
//collect() runs T steps + GAE in one command buffer; trainSlot(t) runs one PPO
//minibatch (forward, loss, backward, Adam) in one command buffer.
//Head row = [logits(ACT_DIM) | value | zero pad] in one 64-wide tile.
//Discrete: Env::Action is a 4-byte integer, the head holds logits, softmax.
//Continuous: Env::Action is a struct of ACT_DIM floats, the head holds the
//means of a Gaussian, and a shared log_std parameter sets the widths.
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
struct InputBackwardParams {
    uint32_t rows;
    uint32_t inDim;
    uint32_t outDim;
    uint32_t chunkRows;
};
struct mlpWeightParams {
    uint32_t M;
    uint32_t N;
    uint32_t S;
    uint32_t chunkRows;
};
struct LinearWeightParams {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t chunkRows;
};
struct ReduceParams {
    uint32_t len;
    uint32_t chunks;
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
    uint32_t floatCount;
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
    static constexpr bool CONTINUOUS = !std::is_integral_v<typename Env::Action>;
    static_assert(CONTINUOUS || sizeof(typename Env::Action) == 4,
                  "the softmax head needs a discrete 4-byte Action");
    static_assert(!CONTINUOUS || sizeof(typename Env::Action) == Env::ACT_DIM * sizeof(float),
                  "the Gaussian head needs an Action of exactly ACT_DIM floats");
    static constexpr float INIT_LOG_STD = 0.0f; //continuous only: std 1 at start, see setLogStd()
    static constexpr uint32_t HEAD_DIM = 64; //one linear tile: logits, value, zero padding
    static_assert(Env::ACT_DIM + 1 <= HEAD_DIM, "ACT_DIM + value must fit one 64-wide tile");

    //a byte range inside one of the arenas
    struct Slice {
        MTL::Buffer* buffer = nullptr;
        size_t offset = 0;
    };
    struct LayernormBuffers {
        Slice gamma;
        Slice beta;
        MTL::Buffer* xnormBuffer;
        MTL::Buffer* stdevBuffer;
        MTL::Buffer* outBuffer;
        Slice dGamma;
        Slice dBeta;
    };
    struct BlockBuffers {
        LayernormBuffers ln;
        Slice upproj;
        Slice downproj;
        MTL::Buffer* mlpVBuffer;
        MTL::Buffer* mlpMaskBuffer;
        MTL::Buffer* mlpOutBuffer;
        MTL::Buffer* residualBuffer;
        Slice dUp;
        Slice dDp;
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
    MTL::ComputePipelineState* mlpBwdWeightsPipeline;
    MTL::ComputePipelineState* linearBwdDwPipeline;
    MTL::ComputePipelineState* reducePipeline;
    MTL::ComputePipelineState* linearFwdPipeline;
    MTL::ComputePipelineState* linearBwdPipeline;
    MTL::ComputePipelineState* policyLossPipeline;
    MTL::ComputePipelineState* samplePipeline;
    MTL::ComputePipelineState* gaePipeline;
    MTL::ComputePipelineState* normalizePipeline;
    MTL::ComputePipelineState* residualAddPipeline;
    MTL::ComputePipelineState* gradNormPipeline;
    MTL::ComputePipelineState* stepPipeline;

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
    uint32_t chunks;    //batch chunks for the weight-gradient reductions
    uint32_t chunkRows; //parallels / chunks, a multiple of 64
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
    InputBackwardParams inputBackwardParams;
    mlpWeightParams mlpWeightParameters;
    LinearWeightParams linearWeightParams;
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
    size_t actionSlotBytes; //parallels x sizeof(Env::Action)

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

    Slice inW;
    Slice inB;
    MTL::Buffer* inputBuffer;
    Slice dInW;
    Slice dInB;

    std::vector<BlockBuffers> blocks;

    LayernormBuffers lnFinal;
    Slice linearW;
    Slice logStd;  //continuous only, ACT_DIM bf16 in the weight arena
    Slice dLogStd; //continuous only, ACT_DIM float in the gradient arena
    MTL::Buffer* logitsBuffer;
    MTL::Buffer* dLogitsBuffer;
    Slice dLinearW;
    MTL::Buffer* statsBuffer;

    MTL::Buffer* dStreamBuffer;
    MTL::Buffer* dBranchBuffer;
    MTL::Buffer* dXnormBuffer;
    MTL::Buffer* dMlpVBuffer;
    MTL::Buffer* dUpPartBuffer; //chunks x embedDim x hiddenDim float
    MTL::Buffer* dDpPartBuffer;
    MTL::Buffer* dWPartBuffer;  //chunks x embedDim x HEAD_DIM float

    //parameters: float-grad params first, then bf16-grad params, each aligned
    //to 128 elements. The flat master, momentum, variance, and weight arena
    //share these element offsets.
    struct ParamSpec {
        size_t count;
        bool gradIsFloat;
        std::vector<__bf16> init;
        Slice* weight;
        Slice* grad;
        size_t offset = 0;
    };
    std::vector<ParamSpec> params;
    size_t optCount = 0;
    size_t optFloatCount = 0;
    float stepCounter = 0.0f;
    MTL::Buffer* weightArena;     //bf16, read by the kernels
    MTL::Buffer* gradFloatArena;  //float accumulators
    MTL::Buffer* gradBfArena;     //bf16 matmul outputs
    MTL::Buffer* parameterBuffer; //float master
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
          rowFloatBytes((size_t)g.parallels * sizeof(float)),
          actionSlotBytes((size_t)g.parallels * sizeof(typename Env::Action)) {
        if (parallels % tileM != 0 || e % tileN != 0 || e > 1024 || hiddenDim % tileN != 0 || layers < 1 || T < 1) {
            std::cerr << "Model: rows and embedDim must be multiples of 64, embedDim <= 1024, layers >= 1, horizon >= 1\n";
            std::exit(1);
        }
        if (e != N_EMBED_CFG) {
            std::cerr << "Model: embedDim " << e << " != N_EMBED_CFG " << N_EMBED_CFG
                      << " from model/config.h\n";
            std::exit(1);
        }
        chunks = 1;
        for (uint32_t c = 16; c >= 1; --c) {
            if (tileRows % c == 0) { chunks = c; break; }
        }
        chunkRows = parallels / chunks;
        inputBackwardParams = InputBackwardParams{parallels, Env::OBS_DIM, embedDim, chunkRows};
        mlpWeightParameters = mlpWeightParams{parallels, embedDim, mlpScale, chunkRows};
        linearWeightParams = LinearWeightParams{parallels, HEAD_DIM, embedDim, chunkRows};
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
        mlpBwdWeightsPipeline = makePipeline(library, "mlpBackwardWeights");
        linearBwdDwPipeline = makePipeline(library, "linearBackwardDw");
        reducePipeline = makePipeline(library, "reducePartials");
        linearFwdPipeline = makePipeline(library, "linearForward");
        linearBwdPipeline = makePipeline(library, "linearBackward");
        policyLossPipeline = makePipeline(library, CONTINUOUS ? "gaussianLossBackward" : "policyLossBackward");
        samplePipeline = makePipeline(library, CONTINUOUS ? "sampleGaussian" : "sampleAction");
        gaePipeline = makePipeline(library, "gaeBackward");
        normalizePipeline = makePipeline(library, "normalizeAdvantage");
        residualAddPipeline = makePipeline(library, "residualAdd");
        gradNormPipeline = makePipeline(library, "gradNormSq");
        stepPipeline = makePipeline(library, "adamStep");
        library->release();

        valueBuffer = newSharedBuffer(rowFloatBytes);
        logProbBuffer = newSharedBuffer(rowFloatBytes);

        const size_t slotCount = (size_t)horizon * parallels;
        obsSlots = newSharedBuffer((size_t)(horizon + 1) * obsSlotBytes);
        actionSlots = newSharedBuffer((size_t)horizon * actionSlotBytes);
        logProbSlots = newSharedBuffer(slotCount * sizeof(float));
        valueSlots = newSharedBuffer(slotCount * sizeof(float));
        rewardSlots = newSharedBuffer(slotCount * sizeof(float));
        doneSlots = newSharedBuffer(slotCount * sizeof(uint8_t));
        advantageSlots = newSharedBuffer(slotCount * sizeof(float));
        returnSlots = newSharedBuffer(slotCount * sizeof(float));
        advNormBuffer = newSharedBuffer(rowFloatBytes);
        std::memset(actionSlots->contents(), 0, (size_t)horizon * actionSlotBytes);
        slotOrder.resize(horizon);
        for (uint32_t t = 0; t < horizon; t++) slotOrder[t] = t;
        hostRng.seed(seed);

        inputBuffer = newSharedBuffer(activationBytes);

        blocks.resize(nLayers);
        for (BlockBuffers& block : blocks) {
            allocLayernorm(block.ln);
            block.mlpVBuffer = newSharedBuffer(hiddenBytes);
            block.mlpMaskBuffer = newSharedBuffer((size_t)parallels * hiddenDim * sizeof(uint8_t));
            block.mlpOutBuffer = newSharedBuffer(activationBytes);
            block.residualBuffer = newSharedBuffer(activationBytes);
        }

        allocLayernorm(lnFinal);
        logitsBuffer = newSharedBuffer((size_t)parallels * HEAD_DIM * sizeof(__bf16));
        dLogitsBuffer = newSharedBuffer((size_t)parallels * HEAD_DIM * sizeof(__bf16));
        statsBuffer = newSharedBuffer((size_t)parallels * 5 * sizeof(float));

        dStreamBuffer = newSharedBuffer(activationBytes);
        dBranchBuffer = newSharedBuffer(activationBytes);
        dXnormBuffer = newSharedBuffer(activationBytes);
        dMlpVBuffer = newSharedBuffer(hiddenBytes);
        dUpPartBuffer = newSharedBuffer((size_t)chunks * embedDim * hiddenDim * sizeof(float));
        dDpPartBuffer = newSharedBuffer((size_t)chunks * embedDim * hiddenDim * sizeof(float));
        dWPartBuffer = newSharedBuffer((size_t)chunks * embedDim * HEAD_DIM * sizeof(float));

        std::mt19937 rng(seed);
        auto normal = [&](std::vector<__bf16>& weight, float stdev) {
            std::normal_distribution<float> dist(0.0f, stdev);
            for (__bf16& x : weight) x = (__bf16)dist(rng);
        };
        auto constant = [](std::vector<__bf16>& weight, float value) {
            std::fill(weight.begin(), weight.end(), (__bf16)value);
        };
        normal(addParam((size_t)Env::OBS_DIM * embedDim, true, inW, dInW), 1.0f / std::sqrt((float)Env::OBS_DIM));
        constant(addParam((size_t)embedDim, true, inB, dInB), 0.0f);
        for (uint32_t n = 0; n < nLayers; ++n) {
            BlockBuffers& block = blocks[n];
            constant(addParam((size_t)embedDim, true, block.ln.gamma, block.ln.dGamma), 1.0f);
            constant(addParam((size_t)embedDim, true, block.ln.beta, block.ln.dBeta), 0.0f);
            normal(addParam((size_t)embedDim * hiddenDim, false, block.upproj, block.dUp), std::sqrt(2.0f / (float)embedDim));
            normal(addParam((size_t)hiddenDim * embedDim, false, block.downproj, block.dDp),
                   1.0f / std::sqrt((float)hiddenDim) / std::sqrt((float)nLayers));
        }
        constant(addParam((size_t)embedDim, true, lnFinal.gamma, lnFinal.dGamma), 1.0f);
        constant(addParam((size_t)embedDim, true, lnFinal.beta, lnFinal.dBeta), 0.0f);
        {
            std::vector<__bf16>& head = addParam((size_t)embedDim * HEAD_DIM, false, linearW, dLinearW);
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
        if constexpr (CONTINUOUS) {
            constant(addParam((size_t)Env::ACT_DIM, true, logStd, dLogStd), INIT_LOG_STD);
        }

        //layout: float-grad params, then bf16-grad params
        const size_t align = 128;
        size_t total = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (ParamSpec& spec : params) {
                if (spec.gradIsFloat != (pass == 0)) continue;
                spec.offset = total;
                total += (spec.count + align - 1) / align * align;
            }
            if (pass == 0) optFloatCount = total;
        }
        optCount = total;
        weightArena = newSharedBuffer(optCount * sizeof(__bf16));
        gradFloatArena = newSharedBuffer(optFloatCount * sizeof(float));
        gradBfArena = newSharedBuffer((optCount - optFloatCount) * sizeof(__bf16));
        parameterBuffer = newSharedBuffer(optCount * sizeof(float));
        momentumBuffer = newSharedBuffer(optCount * sizeof(float));
        varianceBuffer = newSharedBuffer(optCount * sizeof(float));
        normSqBuffer = newSharedBuffer(4 * sizeof(float));

        auto* master = static_cast<float*>(parameterBuffer->contents());
        auto* weights = static_cast<__bf16*>(weightArena->contents());
        std::fill(master, master + optCount, 0.0f);
        std::fill(weights, weights + optCount, (__bf16)0.0f);
        for (ParamSpec& spec : params) {
            std::transform(spec.init.begin(), spec.init.end(), master + spec.offset, [](__bf16 x) { return (float)x; });
            std::copy(spec.init.begin(), spec.init.end(), weights + spec.offset);
            *spec.weight = Slice{weightArena, spec.offset * sizeof(__bf16)};
            *spec.grad = spec.gradIsFloat ? Slice{gradFloatArena, spec.offset * sizeof(float)}
                                          : Slice{gradBfArena, (spec.offset - optFloatCount) * sizeof(__bf16)};
            spec.init.clear();
        }
        std::memset(momentumBuffer->contents(), 0, optCount * sizeof(float));
        std::memset(varianceBuffer->contents(), 0, optCount * sizeof(float));
    }

    float* values() { return static_cast<float*>(valueBuffer->contents()); }
    float* logProbs() { return static_cast<float*>(logProbBuffer->contents()); }
    float headValue(uint32_t row, uint32_t col) const {
        return (float)static_cast<const __bf16*>(logitsBuffer->contents())[(size_t)row * HEAD_DIM + col];
    }
    //continuous only: the shared log standard deviation of action dim k
    float logStdValue(uint32_t k) const {
        static_assert(CONTINUOUS, "logStdValue needs a continuous Env::Action");
        return (float)static_cast<const __bf16*>(weightArena->contents())[logStd.offset / sizeof(__bf16) + k];
    }
    //continuous only: overwrite every log_std, for example to start narrower
    //than INIT_LOG_STD or to act deterministically. Call between train() calls.
    void setLogStd(float value) {
        static_assert(CONTINUOUS, "setLogStd needs a continuous Env::Action");
        const size_t first = logStd.offset / sizeof(__bf16);
        auto* master = static_cast<float*>(parameterBuffer->contents());
        auto* weights = static_cast<__bf16*>(weightArena->contents());
        for (uint32_t k = 0; k < Env::ACT_DIM; ++k) {
            master[first + k] = value;
            weights[first + k] = (__bf16)value;
        }
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
        ln.xnormBuffer = newSharedBuffer(activationBytes);
        ln.stdevBuffer = newSharedBuffer(rowFloatBytes);
        ln.outBuffer = newSharedBuffer(activationBytes);
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
    static void bind(MTL::ComputeCommandEncoder* encoder, const Slice& slice, uint32_t index) {
        encoder->setBuffer(slice.buffer, slice.offset, index);
    }
    //out = bf16(sum of chunk partials)
    void encodeReduce(MTL::CommandBuffer* commandBuffer, MTL::Buffer* part, const Slice& out, uint32_t len) {
        ReduceParams reduceParams{len, chunks};
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, reducePipeline);
        encoder->setBuffer(part, 0, 0);
        bind(encoder, out, 1);
        encoder->setBytes(&reduceParams, sizeof(reduceParams), 2);
        encoder->dispatchThreads(MTL::Size(len, 1, 1), MTL::Size(256, 1, 1));
        encoder->endEncoding();
    }
    static void encodeCopy(MTL::CommandBuffer* commandBuffer, MTL::Buffer* src, size_t srcOffset,
                           MTL::Buffer* dst, size_t dstOffset, size_t bytes) {
        MTL::BlitCommandEncoder* blit = commandBuffer->blitCommandEncoder();
        blit->copyFromBuffer(src, srcOffset, dst, dstOffset, bytes);
        blit->endEncoding();
    }
    inline size_t obsOffset(uint32_t t) const { return (size_t)t * obsSlotBytes; }
    inline size_t rowOffset(uint32_t t) const { return (size_t)t * rowFloatBytes; }
    inline size_t actionOffset(uint32_t t) const { return (size_t)t * actionSlotBytes; }
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
        bind(encoder, ln.gamma, 3);
        bind(encoder, ln.beta, 4);
        encoder->setBuffer(ln.stdevBuffer, 0, 5);
        encoder->setBytes(&layernormParams, sizeof(layernormParams), 6);
        encoder->dispatchThreadgroups(MTL::Size(layernormParams.rows, 1, 1), MTL::Size(layernormParams.group_size, 1, 1));
        encoder->endEncoding();
    }
    void encodeLayernormBackward(MTL::CommandBuffer* commandBuffer, LayernormBuffers& ln, MTL::Buffer* dZBuffer) {
        MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, layernormBwdPipeline);
        encoder->setBuffer(dZBuffer, 0, 0);
        encoder->setBuffer(ln.xnormBuffer, 0, 1);
        bind(encoder, ln.gamma, 2);
        encoder->setBuffer(ln.stdevBuffer, 0, 3);
        encoder->setBuffer(dXnormBuffer, 0, 5);
        bind(encoder, ln.dGamma, 6);
        bind(encoder, ln.dBeta, 7);
        encoder->setBytes(&layernormParams, sizeof(layernormParams), 8);
        encoder->dispatchThreadgroups(MTL::Size(layernormParams.rows, 1, 1), MTL::Size(layernormParams.group_size, 1, 1));
        encoder->endEncoding();
    }

    void encodeForward(MTL::CommandBuffer* commandBuffer, MTL::Buffer* obsBuffer, size_t obsByteOffset) {
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, inputFwdPipeline);
            encoder->setBuffer(obsBuffer, obsByteOffset, 0);
            bind(encoder, inW, 1);
            bind(encoder, inB, 2);
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
                bind(encoder, block.upproj, 1);
                encoder->setBuffer(block.mlpVBuffer, 0, 2);
                encoder->setBuffer(block.mlpMaskBuffer, 0, 3);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 4);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, hiddenDim / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpFwdTwoPipeline);
                bind(encoder, block.downproj, 0);
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
            bind(encoder, linearW, 1);
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
        if constexpr (CONTINUOUS) {
            std::normal_distribution<float> normal(0.0f, 1.0f);
            float* actionF = reinterpret_cast<float*>(action);
            for (uint32_t row = 0; row < parallels; ++row) {
                const size_t base = (size_t)row * HEAD_DIM;
                float lp = 0.0f;
                for (uint32_t k = 0; k < Env::ACT_DIM; ++k) {
                    const float ls = logStdValue(k);
                    const float mu = (float)head[base + k];
                    const float a = mu + std::exp(ls) * normal(rng);
                    const float d = (a - mu) * std::exp(-ls);
                    actionF[(size_t)row * Env::ACT_DIM + k] = a;
                    lp += -0.5f * d * d - ls - 0.9189385332046727f;
                }
                logProb[row] = lp;
                value[row] = (float)head[base + Env::ACT_DIM];
            }
            return;
        }
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
    //discrete: argmax; continuous: the mean
    void actGreedy() {
        typename Env::Action* action = gym.actions();
        if constexpr (CONTINUOUS) {
            float* actionF = reinterpret_cast<float*>(action);
            for (uint32_t row = 0; row < parallels; ++row) {
                for (uint32_t k = 0; k < Env::ACT_DIM; ++k) actionF[(size_t)row * Env::ACT_DIM + k] = headValue(row, k);
            }
            return;
        }
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
        encoder->setBuffer(actionSlots, actionOffset(t), 2);
        encoder->setBuffer(logProbSlots, rowOffset(t), 3);
        encoder->setBuffer(valueSlots, rowOffset(t), 4);
        encoder->setBytes(&sampleParams, sizeof(sampleParams), 5);
        if constexpr (CONTINUOUS) bind(encoder, logStd, 6);
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
            gym.encodeStep(commandBuffer, actionSlots, actionOffset(t), obsSlots, obsOffset(t + 1), rewardSlots, rowOffset(t));
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
            encoder->setBuffer(actionSlots, actionOffset(t), 1);
            encoder->setBuffer(advNormBuffer, 0, 2);
            encoder->setBuffer(returnSlots, rowOffset(t), 3);
            encoder->setBuffer(logProbSlots, rowOffset(t), 4);
            encoder->setBuffer(dLogitsBuffer, 0, 5);
            encoder->setBuffer(statsBuffer, 0, 6);
            encoder->setBytes(&policyParams, sizeof(policyParams), 7);
            if constexpr (CONTINUOUS) {
                //the Gaussian loss accumulates the log_std gradient with atomics
                bind(encoder, logStd, 8);
                bind(encoder, dLogStd, 9);
            }
            encoder->dispatchThreads(MTL::Size(parallels, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
    }

    //zero both gradient arenas and the grad-norm accumulator. Runs before the
    //loss because the Gaussian loss already accumulates into the float arena.
    void encodeZeroGrads(MTL::CommandBuffer* commandBuffer) {
        MTL::BlitCommandEncoder* blit = commandBuffer->blitCommandEncoder();
        blit->fillBuffer(gradFloatArena, NS::Range::Make(0, optFloatCount * sizeof(float)), 0);
        blit->fillBuffer(gradBfArena, NS::Range::Make(0, (optCount - optFloatCount) * sizeof(__bf16)), 0);
        blit->fillBuffer(normSqBuffer, NS::Range::Make(0, 4 * sizeof(float)), 0);
        blit->endEncoding();
    }

    void encodeBackward(MTL::CommandBuffer* commandBuffer, MTL::Buffer* obsBuffer, size_t obsByteOffset) {
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, linearBwdPipeline);
            bind(encoder, linearW, 1);
            encoder->setBuffer(dLogitsBuffer, 0, 2);
            encoder->setBuffer(dStreamBuffer, 0, 3);
            encoder->setBytes(&linearParams, sizeof(linearParams), 5);
            encoder->dispatchThreadgroups(MTL::Size(tileRows, 1, 1), MTL::Size(32 * simdGroups, 1, 1));
            encoder->endEncoding();
        }
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, linearBwdDwPipeline);
            encoder->setBuffer(lnFinal.outBuffer, 0, 0);
            encoder->setBuffer(dLogitsBuffer, 0, 1);
            encoder->setBuffer(dWPartBuffer, 0, 2);
            encoder->setBytes(&linearWeightParams, sizeof(linearWeightParams), 3);
            encoder->dispatchThreadgroups(MTL::Size(HEAD_DIM / tileN, chunks, 1), MTL::Size(32 * simdGroups, 1, 1));
            encoder->endEncoding();
        }
        encodeReduce(commandBuffer, dWPartBuffer, dLinearW, embedDim * HEAD_DIM);
        encodeLayernormBackward(commandBuffer, lnFinal, dStreamBuffer);

        for (int n = (int)nLayers - 1; n >= 0; --n) {
            BlockBuffers& block = blocks[n];

            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpBwdOnePipeline);
                bind(encoder, block.downproj, 1);
                encoder->setBuffer(block.mlpMaskBuffer, 0, 2);
                encoder->setBuffer(dStreamBuffer, 0, 3);
                encoder->setBuffer(dMlpVBuffer, 0, 5);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 6);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, hiddenDim / tileN, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpBwdWeightsPipeline);
                encoder->setBuffer(block.ln.outBuffer, 0, 0);
                encoder->setBuffer(dMlpVBuffer, 0, 1);
                encoder->setBuffer(block.mlpVBuffer, 0, 2);
                encoder->setBuffer(dStreamBuffer, 0, 3);
                encoder->setBuffer(dUpPartBuffer, 0, 4);
                encoder->setBuffer(dDpPartBuffer, 0, 5);
                encoder->setBytes(&mlpWeightParameters, sizeof(mlpWeightParameters), 6);
                encoder->dispatchThreadgroups(MTL::Size(hiddenDim / tileM, chunks, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeReduce(commandBuffer, dUpPartBuffer, block.dUp, embedDim * hiddenDim);
            encodeReduce(commandBuffer, dDpPartBuffer, block.dDp, embedDim * hiddenDim);
            {
                MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, mlpBwdTwoPipeline);
                encoder->setBuffer(dBranchBuffer, 0, 1);
                encoder->setBuffer(dMlpVBuffer, 0, 2);
                bind(encoder, block.upproj, 3);
                encoder->setBytes(&mlpParameters, sizeof(mlpParameters), 8);
                encoder->dispatchThreadgroups(MTL::Size(tileRows, tileCols, 1), MTL::Size(32 * simdGroups, 1, 1));
                encoder->endEncoding();
            }
            encodeLayernormBackward(commandBuffer, block.ln, dBranchBuffer);
            encodeResidualAdd(commandBuffer, dBranchBuffer, dStreamBuffer, dStreamBuffer);
        }

        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, inputBwdPipeline);
            encoder->setBuffer(obsBuffer, obsByteOffset, 0);
            encoder->setBuffer(dStreamBuffer, 0, 1);
            bind(encoder, dInW, 2);
            bind(encoder, dInB, 3);
            encoder->setBytes(&inputBackwardParams, sizeof(inputBackwardParams), 4);
            encoder->dispatchThreads(MTL::Size((Env::OBS_DIM + 1) * embedDim, chunks, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
    }

    std::vector<__bf16>& addParam(size_t count, bool gradIsFloat, Slice& weight, Slice& grad) {
        params.push_back(ParamSpec{count, gradIsFloat, std::vector<__bf16>(count, (__bf16)0.0f), &weight, &grad});
        return params.back().init;
    }
    //grad norm over both arenas, then Adam on the float master, which also
    //writes the bf16 weight arena the next forward reads
    void encodeStep(MTL::CommandBuffer* commandBuffer) {
        stepCounter += 1.0f;
        AdamParams adamParams;
        adamParams.count = (uint32_t)optCount;
        adamParams.floatCount = (uint32_t)optFloatCount;
        adamParams.lr = lr;
        adamParams.beta1Pow = std::pow(adamParams.beta1, stepCounter);
        adamParams.beta2Pow = std::pow(adamParams.beta2, stepCounter);
        adamParams.maxGradNorm = maxGradNorm;
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, gradNormPipeline);
            encoder->setBuffer(gradFloatArena, 0, 0);
            encoder->setBuffer(gradBfArena, 0, 1);
            encoder->setBuffer(normSqBuffer, 0, 2);
            encoder->setBytes(&adamParams, sizeof(adamParams), 3);
            encoder->dispatchThreads(MTL::Size((optCount + 31) / 32 * 32, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
        {
            MTL::ComputeCommandEncoder* encoder = makeEncoder(commandBuffer, stepPipeline);
            encoder->setBuffer(parameterBuffer, 0, 0);
            encoder->setBuffer(gradFloatArena, 0, 1);
            encoder->setBuffer(gradBfArena, 0, 2);
            encoder->setBuffer(momentumBuffer, 0, 3);
            encoder->setBuffer(varianceBuffer, 0, 4);
            encoder->setBuffer(weightArena, 0, 5);
            encoder->setBuffer(normSqBuffer, 0, 6);
            encoder->setBytes(&adamParams, sizeof(adamParams), 7);
            encoder->dispatchThreads(MTL::Size(optCount, 1, 1), MTL::Size(256, 1, 1));
            encoder->endEncoding();
        }
    }

    void trainSlot(uint32_t t) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        encodeForward(commandBuffer, obsSlots, obsOffset(t));
        encodeZeroGrads(commandBuffer);
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
            ln.xnormBuffer->release();
            ln.stdevBuffer->release();
            ln.outBuffer->release();
        };
        for (BlockBuffers& block : blocks) {
            releaseLayernorm(block.ln);
            block.mlpVBuffer->release();
            block.mlpMaskBuffer->release();
            block.mlpOutBuffer->release();
            block.residualBuffer->release();
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
        inputBuffer->release();
        logitsBuffer->release();
        dLogitsBuffer->release();
        statsBuffer->release();
        dStreamBuffer->release();
        dBranchBuffer->release();
        dXnormBuffer->release();
        dMlpVBuffer->release();
        dUpPartBuffer->release();
        dDpPartBuffer->release();
        dWPartBuffer->release();
        weightArena->release();
        gradFloatArena->release();
        gradBfArena->release();
        parameterBuffer->release();
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
        mlpBwdWeightsPipeline->release();
        linearBwdDwPipeline->release();
        reducePipeline->release();
        linearFwdPipeline->release();
        linearBwdPipeline->release();
        policyLossPipeline->release();
        samplePipeline->release();
        gaePipeline->release();
        normalizePipeline->release();
        residualAddPipeline->release();
        gradNormPipeline->release();
        stepPipeline->release();
    }
};
