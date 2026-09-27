Environment Template:   

**Metal Side: **  

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

**Host Side: **

struct Environment {  
	struct State { };  
	static constexpr uint32_t OBS_DIM;  
    static constexpr uint32_t ACT_DIM;  
    static constexpr const char* KERNEL; //make sure this matches the kernel name  
};