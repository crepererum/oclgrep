#pragma once

#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include "common.hpp"

class oclrunner;

class oclengine {
    friend oclrunner;

    public:
        // config
        static constexpr std::uint32_t cache_mask      = calc_alignement_mask(7); // sets cache alignement of local text cache base 32
        static constexpr std::uint32_t flag_iter_max   = 1;                       // index of "we've reached too many iteratios"-flag
        static constexpr std::uint32_t flag_stack_full = 0;                       // index of "thread-local stack was too small"-flag
        static constexpr std::uint32_t flags_n         = 2;                       // number of flags
        static constexpr std::uint32_t group_size      = 64;                      // OpenCL group size
        static constexpr std::uint32_t max_iter_count  = 2048;                    // limits number of iterations to prevent timeouts
        static constexpr std::uint32_t max_stack_size  = 128;                     // limits thread-local stack
        static constexpr std::uint32_t multi_input_n   = 64;                      // load-balancing by using multiple start postions per thread
        static constexpr std::uint32_t oversize_cache  = 4;                       // cache_size=group_size*oversize_cache
        static constexpr std::uint32_t result_fail     = 0xffffffff;              // placeholder for "FAIL" results of automaton
        static constexpr std::uint32_t sync_count      = 128;                     // controls after how many iterations group threads sync
        static constexpr std::uint32_t use_cache       = 0;                       // controls if kernels use local memory cache

        oclengine();

    private:
        cl::Platform platform;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue queue;

        cl::Program programAutomaton;
        cl::Program programCollector;

        cl::Kernel kernelAutomaton;
        cl::Kernel kernelTransform;
        cl::Kernel kernelScan;
        cl::Kernel kernelMove;
};

class oclrunner {
    public:
        oclrunner(const std::shared_ptr<oclengine>& eng, std::uint32_t max_chunk_size, const serial::graph& graph, bool printProfile);

        std::vector<std::uint32_t> run(const std::u32string& chunk);

    private:
        std::shared_ptr<oclengine> eng;
        std::uint32_t max_chunk_size;
        serial::graph graph;
        bool printProfile;

        cl::Buffer dAutomatonData;
        cl::Buffer dText;
        cl::Buffer dOutput;
        cl::Buffer dFlags;
        cl::Buffer dScanbuffer0;
        cl::Buffer dScanbuffer1;
};
