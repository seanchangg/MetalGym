#pragma once
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iostream>


template <typename Env>
struct Gym {
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* stateBuffer;
    MTL::Buffer* actionBuffer;
    MTL::Buffer* obsBuffer;
    MTL::Buffer* rewardBuffer;
    MTL::Buffer* doneBuffer;
    MTL::Buffer* rngBuffer;
    size_t stateBytes;
    size_t actionBytes;
    size_t obsBytes;
    size_t rewardBytes;
    size_t doneBytes;
    size_t rngBytes;
    uint32_t parallels;

    Gym(int p) : parallels(p), stateBytes(p*sizeof(typename Env::State)), actionBytes(p*sizeof(typename Env::Action)), 
   				 obsBytes(p*sizeof(typename Env::Obs)), rewardBytes(p*sizeof(float)), doneBytes(p*sizeof(uint8_t)), 
   				 rngBytes(p*sizeof(uint32_t))  
   	{
        pool = NS::AutoreleasePool::alloc()->init();
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        MTL::Library* library = device->newDefaultLibrary();
        MTL::Function* function = library->newFunction(NS::String::string(Env::KERNEL, NS::UTF8StringEncoding)); 
        
        stateBuffer = device->newBuffer(stateBytes, MTL::ResourceStorageModeShared);
        actionBuffer = device->newBuffer(actionBytes, MTL::ResourceStorageModeShared);
        obsBuffer = device->newBuffer(obsBytes, MTL::ResourceStorageModeShared);
        rewardBuffer = device->newBuffer(rewardBytes, MTL::ResourceStorageModeShared);
        doneBuffer = device->newBuffer(doneBytes, MTL::ResourceStorageModeShared);
        rngBuffer = device->newBuffer(rngBytes, MTL::ResourceStorageModeShared);
        
        
        //done = 1 so the first dispatch resets every env
        std::memset(doneBuffer->contents(), 1, doneBytes);
        std::memset(actionBuffer->contents(), 0, actionBytes);
        //xorshift stays at zero forever, so every rng slot needs a nonzero seed
        auto* rngInit = static_cast<uint32_t*>(rngBuffer->contents());
        for (uint32_t i = 0; i < parallels; i++) {
            rngInit[i] = i * 2654435761u + 1u;
        }

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(function, &error);
        if (!pipelineState) {
            std::cerr << "pipeline for " << Env::KERNEL << " failed: "
                      << (error ? error->localizedDescription()->utf8String() : "unknown") << "\n";
            std::exit(1);
        }
		function->release();
		library->release();
    }
    ~Gym() {
            rngBuffer->release();
            doneBuffer->release();
            rewardBuffer->release();
            obsBuffer->release();
            actionBuffer->release();
            stateBuffer->release();
            
            pipelineState->release();
            commandQueue->release();
            pool->release();
            device->release();
        }
    typename Env::Action* actions() { return static_cast<typename Env::Action*>(actionBuffer->contents()); }
    uint8_t* done() { return static_cast<uint8_t*>(doneBuffer->contents()); }
    typename Env::Obs* obs() { return static_cast<typename Env::Obs*>(obsBuffer->contents()); }

    void run(std::vector<float>& reward) {
        //command buffers are autoreleased; drain them per step or the pool grows
        NS::AutoreleasePool* stepPool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipelineState);
        encoder->setBuffer(stateBuffer, 0, 0);
        encoder->setBuffer(actionBuffer, 0, 1);
        encoder->setBuffer(obsBuffer, 0, 2);
        encoder->setBuffer(rewardBuffer, 0, 3);
        encoder->setBuffer(doneBuffer, 0, 4);
        encoder->setBuffer(rngBuffer, 0, 5);
        encoder->dispatchThreads(MTL::Size(parallels, 1, 1), MTL::Size(pipelineState->maxTotalThreadsPerThreadgroup(), 1, 1));
        
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        auto* rewardOut = static_cast<float*>(rewardBuffer->contents());
        std::copy(rewardOut, rewardOut+reward.size(), reward.begin());
        stepPool->release();
    }
};