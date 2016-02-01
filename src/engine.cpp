#include <cstdint>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <utility>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include "common.hpp"

// resource data
// http://www.burtonini.com/blog/computers/ld-blobs-2007-07-13-15-50
extern char _binary_automaton_cl_start[];
extern char _binary_automaton_cl_end[];
extern char _binary_collector_cl_start[];
extern char _binary_collector_cl_end[];


static_assert(sizeof(std::uint32_t) == sizeof(cl_uint), "OpenCL uint has to be 32bit");
static_assert(sizeof(char32_t) == sizeof(std::uint32_t), "uint32 and char32_t have to have the same size");
static_assert(sizeof(char) == sizeof(cl_char), "OpenCL char has not the same size as host char");


cl::Program buildProgramFromPtr(const char* begin, const char* end, const cl::Context& context, const std::vector<cl::Device>& devices, const std::map<std::string, std::string>& defines) {
    // dump data to string
    std::string sourceCode(begin, end);

    // create build options string
    std::stringstream buildOptionsSS;
    buildOptionsSS << "-Werror ";
    for (const auto& kv : defines) {
        buildOptionsSS << "-D" << std::get<0>(kv) << "=" << std::get<1>(kv) << " ";
    }
    auto buildOptions = buildOptionsSS.str();

    // create and build program
    cl::Program program(context, sourceCode);
    try {
        program.build(devices, buildOptions.c_str());
    } catch (const cl::Error& /*e*/) {
        // assume that this as an build error, dump build log to an exception
        std::stringstream ss;
        ss << "OpenCl build errors:" << std::endl;
        for (const auto& dev : devices) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            if (!buildLog.empty()) {
                ss << buildLog << std::endl;
            }
        }
        throw internal_exception(ss.str());
    }

    // everything went fine, return final program
    return program;
}

float getEventTimeMS(const cl::Event& evt) {
    evt.wait();
    cl_ulong t_start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong t_end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return static_cast<float>(t_end - t_start) / (1000.f * 1000.f);
}

constexpr std::uint32_t calc_alignement_mask(std::size_t n_bytes) {
    constexpr std::uint32_t full = ~static_cast<std::uint32_t>(0);
    std::uint32_t base = static_cast<std::uint32_t>(1) << n_bytes;
    return full - base + static_cast<std::uint32_t>(1);
}

constexpr std::size_t adjust_globalsize(std::size_t globalsize, std::size_t localsize) {
    if (globalsize % localsize != 0) {
        globalsize += localsize - globalsize % localsize;
    }
    return globalsize;
}

std::vector<std::uint32_t> runEngine(const serial::graph& graph, const std::u32string& fcontent, bool printProfile) {
    // config
    constexpr std::uint32_t cache_mask      = calc_alignement_mask(7); // sets cache alignement of local text cache base 32
    constexpr std::uint32_t flag_iter_max   = 1;                       // index of "we've reached too many iteratios"-flag
    constexpr std::uint32_t flag_stack_full = 0;                       // index of "thread-local stack was too small"-flag
    constexpr std::uint32_t flags_n         = 2;                       // number of flags
    constexpr std::uint32_t group_size      = 64;                      // OpenCL group size
    constexpr std::uint32_t max_iter_count  = 2048;                    // limits number of iterations to prevent timeouts
    constexpr std::uint32_t max_stack_size  = 128;                     // limits thread-local stack
    constexpr std::uint32_t multi_input_n   = 64;                      // load-balancing by using multiple start postions per thread
    constexpr std::uint32_t oversize_cache  = 4;                       // cache_size=group_size*oversize_cache
    constexpr std::uint32_t result_fail     = 0xffffffff;              // placeholder for "FAIL" results of automaton
    constexpr std::uint32_t sync_count      = 128;                     // controls after how many iterations group threads sync
    constexpr std::uint32_t use_cache       = 0;                       // controls if kernels use local memory cache

    // set up OpenCL
    std::vector<cl::Platform> pool_platforms;
    cl::Platform::get(&pool_platforms);
    if (pool_platforms.empty()) {
        throw user_error("no OpenCL platforms found!");
    }
    cl::Platform platform = pool_platforms[0]; // XXX: make this selectable!

    std::vector<cl::Device> pool_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &pool_devices);
    if (pool_devices.empty()) {
        throw user_error("no OpenCL devices found!");
    }
    std::vector<cl::Device> devices{pool_devices[0]}; // XXX: make this a user choice!
    for (const auto& dev : devices) {
        if (!dev.getInfo<CL_DEVICE_ENDIAN_LITTLE>()) {
            throw user_error("not all selected devices are little endian!");
        }
        if (dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() < graph.size()) {
            throw user_error("compiled automaton is too large for the OpenCL device!");
        }
    }

    cl::Context context(devices);

    cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling);

    // build kernel
    std::map<std::string, std::string> buildDefines{
        {"CACHE_MASK",      std::to_string(cache_mask)},
        {"FLAG_ITER_MAX",   std::to_string(flag_iter_max)},
        {"FLAG_STACK_FULL", std::to_string(flag_stack_full)},
        {"GROUP_SIZE",      std::to_string(group_size)},
        {"ID_BEGIN",        std::to_string(serial::id_begin)},
        {"ID_FAIL",         std::to_string(serial::id_fail)},
        {"ID_OK",           std::to_string(serial::id_ok)},
        {"MAX_ITER_COUNT",  std::to_string(max_iter_count)},
        {"MAX_STACK_SIZE",  std::to_string(max_stack_size)},
        {"OVERSIZE_CACHE",  std::to_string(oversize_cache)},
        {"RESULT_FAIL",     std::to_string(result_fail)},
        {"SYNC_COUNT",      std::to_string(sync_count)},
        {"USE_CACHE",       std::to_string(use_cache)},
    };

    cl::Program programAutomaton = buildProgramFromPtr(_binary_automaton_cl_start, _binary_automaton_cl_end, context, devices, buildDefines);
    cl::Program programCollector = buildProgramFromPtr(_binary_collector_cl_start, _binary_collector_cl_end, context, devices, buildDefines);
    cl::Kernel kernelAutomaton(programAutomaton, "automaton");
    cl::Kernel kernelTransform(programCollector, "transform");
    cl::Kernel kernelScan(programCollector, "scan");
    cl::Kernel kernelMove(programCollector, "move");

    // OpenCL events
    cl::Event evtUploadAutomaton;
    cl::Event evtUploadText;
    cl::Event evtUploadFlags;
    cl::Event evtKernelAutomaton;
    cl::Event evtKernelTransform;
    std::vector<cl::Event> evtsKernelScan;
    cl::Event evtKernelMove;
    cl::Event evtDownloadOutputSize;
    cl::Event evtDownloadOutput;
    cl::Event evtDownloadFlags;

    // create buffer
    std::vector<char> flags(flags_n, 0);

    cl::Buffer dAutomatonData(
        context,
        CL_MEM_READ_ONLY,
        graph.size() * sizeof(std::uint8_t),
        nullptr
    );

    cl::Buffer dText(
        context,
        CL_MEM_READ_ONLY,
        fcontent.size() * sizeof(char32_t),
        nullptr
    );

    cl::Buffer dOutput(
        context,
        CL_MEM_READ_WRITE,
        fcontent.size() * sizeof(cl_uint),
        nullptr
    );

    cl::Buffer dFlags(
        context,
        CL_MEM_READ_WRITE,
        flags.size() * sizeof(char),
        nullptr
    );

    cl::Buffer dScanbuffer0(
        context,
        CL_MEM_READ_WRITE,
        fcontent.size() * sizeof(cl_uint),
        nullptr
    );

    cl::Buffer dScanbuffer1(
        context,
        CL_MEM_READ_WRITE,
        fcontent.size() * sizeof(cl_uint),
        nullptr
    );

    // upload data
    queue.enqueueWriteBuffer(dAutomatonData, false, 0, graph.size()  * sizeof(std::uint8_t), graph.data.data(), nullptr, &evtUploadAutomaton);
    queue.enqueueWriteBuffer(dText, false, 0, fcontent.size()  * sizeof(char32_t), fcontent.data(), nullptr, &evtUploadText);
    queue.enqueueWriteBuffer(dFlags, false, 0, flags.size() * sizeof(char), flags.data(), nullptr, &evtUploadFlags);

    // run automaton kernel
    kernelAutomaton.setArg(0, static_cast<cl_uint>(graph.n));
    kernelAutomaton.setArg(1, static_cast<cl_uint>(graph.o));
    kernelAutomaton.setArg(2, static_cast<cl_uint>(fcontent.size()));
    kernelAutomaton.setArg(3, static_cast<cl_uint>(multi_input_n));
    kernelAutomaton.setArg(4, dAutomatonData);
    kernelAutomaton.setArg(5, dText);
    kernelAutomaton.setArg(6, dOutput);
    kernelAutomaton.setArg(7, dFlags);
    kernelAutomaton.setArg(8, oversize_cache * group_size * sizeof(char32_t), nullptr);

    std::size_t totalSize = fcontent.size() / multi_input_n;
    if (fcontent.size() % multi_input_n != 0) {
        totalSize += 1;
    }
    totalSize = adjust_globalsize(totalSize, group_size);
    queue.enqueueNDRangeKernel(kernelAutomaton, cl::NullRange, cl::NDRange(totalSize), cl::NDRange(group_size), nullptr, &evtKernelAutomaton);

    // run transform kernel
    std::size_t globalsize = adjust_globalsize(fcontent.size(), group_size);
    kernelTransform.setArg(0, dOutput);
    kernelTransform.setArg(1, dScanbuffer0);
    kernelTransform.setArg(2, static_cast<cl_uint>(fcontent.size()));
    queue.enqueueNDRangeKernel(kernelTransform, cl::NullRange, cl::NDRange(globalsize), cl::NDRange(group_size), nullptr, &evtKernelTransform);

    // run scan kernel
    std::size_t offset = 1;
    while (offset < fcontent.size()) {
        evtsKernelScan.emplace_back();
        kernelScan.setArg(0, dScanbuffer0);
        kernelScan.setArg(1, dScanbuffer1);
        kernelScan.setArg(2, static_cast<cl_uint>(fcontent.size()));
        kernelScan.setArg(3, static_cast<cl_uint>(offset));
        queue.enqueueNDRangeKernel(kernelScan, cl::NullRange, cl::NDRange(globalsize), cl::NDRange(group_size), nullptr, &evtsKernelScan[evtsKernelScan.size() - 1]);
        std::swap(dScanbuffer0, dScanbuffer1);
        offset = offset << 1;
    }

    // run move kernel
    kernelMove.setArg(0, dScanbuffer0);
    kernelMove.setArg(1, dOutput);
    kernelMove.setArg(2, dScanbuffer1);
    kernelMove.setArg(3, static_cast<cl_uint>(fcontent.size()));
    queue.enqueueNDRangeKernel(kernelMove, cl::NullRange, cl::NDRange(globalsize), cl::NDRange(group_size), nullptr, &evtKernelMove);
    std::swap(dOutput, dScanbuffer1);

    // get output
    std::size_t outputSize;
    queue.enqueueReadBuffer(dScanbuffer0, true, (fcontent.size() - 1) * sizeof(cl_uint), 1 * sizeof(cl_uint), &outputSize, nullptr, &evtDownloadOutputSize);
    std::vector<uint32_t> output(outputSize, 0);
    queue.enqueueReadBuffer(dOutput, false, 0, outputSize * sizeof(cl_uint), output.data(), nullptr, &evtDownloadOutput);

    queue.enqueueReadBuffer(dFlags, false, 0, flags.size() * sizeof(char), flags.data(), nullptr, &evtDownloadFlags);

    queue.finish();

    if (printProfile) {
        std::cout << "Profiling data:" << std::endl
            << "  uploadAutomaton    = " << getEventTimeMS(evtUploadAutomaton) << "ms" << std::endl
            << "  uploadText         = " << getEventTimeMS(evtUploadText) << "ms" << std::endl
            << "  uploadFlags        = " << getEventTimeMS(evtUploadFlags) << "ms" << std::endl
            << "  kernelAutomaton    = " << getEventTimeMS(evtKernelAutomaton) << "ms" << std::endl
            << "  kernelTransform    = " << getEventTimeMS(evtKernelTransform) << "ms" << std::endl
            << "  kernelScan         = " << std::endl;
        float sumScan = 0.f;
        for (std::size_t i = 0; i < evtsKernelScan.size(); ++i) {
            float t = getEventTimeMS(evtsKernelScan[i]);
            sumScan += t;
            std::cout << "    " << t << "ms" << std::endl;
        }
        std::cout << "    ====" << std::endl
            << "    " << sumScan << "ms" << std::endl
            << "  kernelMove         = " << getEventTimeMS(evtKernelMove) << "ms" << std::endl
            << "  downloadOutputSize = " << getEventTimeMS(evtDownloadOutputSize) << "ms" << std::endl
            << "  downloadOutput     = " << getEventTimeMS(evtDownloadOutput) << "ms" << std::endl
            << "  downloadFlags      = " << getEventTimeMS(evtDownloadFlags) << "ms" << std::endl;
    }

    if (flags[flag_stack_full]) {
        throw user_error("Automaton engine error: task stack was full!");
    }
    if (flags[flag_iter_max]) {
        throw user_error("Automaton engine error: reached maximum iteration count!");
    }

    return output;
}
