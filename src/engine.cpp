#include <cstdint>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <utility>

#include "common.hpp"
#include "engine.hpp"

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

constexpr std::size_t adjust_globalsize(std::size_t globalsize, std::size_t localsize) {
    if (globalsize % localsize != 0) {
        globalsize += localsize - globalsize % localsize;
    }
    return globalsize;
}

oclengine::oclengine() {
    // set up OpenCL
    std::vector<cl::Platform> pool_platforms;
    cl::Platform::get(&pool_platforms);
    if (pool_platforms.empty()) {
        throw user_error("no OpenCL platforms found!");
    }
    platform = pool_platforms[0]; // XXX: make this selectable!

    std::vector<cl::Device> pool_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &pool_devices);
    if (pool_devices.empty()) {
        throw user_error("no OpenCL devices found!");
    }
    devices = {pool_devices[0]}; // XXX: make this a user choice!
    for (const auto& dev : devices) {
        if (!dev.getInfo<CL_DEVICE_ENDIAN_LITTLE>()) {
            throw user_error("not all selected devices are little endian!");
        }
    }

    context = cl::Context(devices);
    queue = cl::CommandQueue(context, devices[0], cl::QueueProperties::Profiling);

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

    programAutomaton = buildProgramFromPtr(_binary_automaton_cl_start, _binary_automaton_cl_end, context, devices, buildDefines);
    programCollector = buildProgramFromPtr(_binary_collector_cl_start, _binary_collector_cl_end, context, devices, buildDefines);
    kernelAutomaton = cl::Kernel(programAutomaton, "automaton");
    kernelTransform = cl::Kernel(programCollector, "transform");
    kernelScan = cl::Kernel(programCollector, "scan");
    kernelMove = cl::Kernel(programCollector, "move");
}

oclrunner::oclrunner(const std::shared_ptr<oclengine>& eng, std::uint32_t max_chunk_size, const serial::graph& graph, bool printProfile) : eng(eng), max_chunk_size(max_chunk_size), graph(graph), printProfile(printProfile) {
    // basic checks
    for (const auto& dev : eng->devices) {
        if (dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() < graph.size()) {
            throw user_error("compiled automaton is too large for the OpenCL device!");
        }
    }

    // OpenCL events
    cl::Event evtUploadAutomaton;

    // create buffer
    static_assert(sizeof(serial::word) == sizeof(std::uint32_t), "ah, that won't work with OpenCL");
    dAutomatonData = cl::Buffer(
        eng->context,
        CL_MEM_READ_ONLY,
        graph.size() * sizeof(std::uint32_t),
        nullptr
    );

    dText = cl::Buffer(
        eng->context,
        CL_MEM_READ_ONLY,
        max_chunk_size * sizeof(char32_t),
        nullptr
    );

    dOutput = cl::Buffer(
        eng->context,
        CL_MEM_READ_WRITE,
        max_chunk_size * sizeof(cl_uint),
        nullptr
    );

    dFlags = cl::Buffer(
        eng->context,
        CL_MEM_READ_WRITE,
        eng->flags_n * sizeof(char),
        nullptr
    );

    dScanbuffer0 = cl::Buffer(
        eng->context,
        CL_MEM_READ_WRITE,
        max_chunk_size * sizeof(cl_uint),
        nullptr
    );

    dScanbuffer1 = cl::Buffer(
        eng->context,
        CL_MEM_READ_WRITE,
        max_chunk_size * sizeof(cl_uint),
        nullptr
    );

    // upload some data
    eng->queue.enqueueWriteBuffer(dAutomatonData, false, 0, graph.size()  * sizeof(std::uint32_t), graph.data.data(), nullptr, &evtUploadAutomaton);

    eng->queue.finish();

    if (printProfile) {
        std::cout << "Profiling data:" << std::endl
            << "  uploadAutomaton    = " << getEventTimeMS(evtUploadAutomaton) << "ms" << std::endl;
    }
}

std::vector<std::uint32_t> oclrunner::run(const std::u32string& chunk) {
    sanity_assert(chunk.size() > 0, "chunk must contain content");
    sanity_assert(chunk.size() <= max_chunk_size, "chunk is too big for this config");

    // OpenCL events
    cl::Event evtUploadText;
    cl::Event evtUploadFlags;
    cl::Event evtKernelAutomaton;
    cl::Event evtKernelTransform;
    std::vector<cl::Event> evtsKernelScan;
    cl::Event evtKernelMove;
    cl::Event evtDownloadOutputSize;
    cl::Event evtDownloadOutput;
    cl::Event evtDownloadFlags;

    // upload data
    std::vector<char> flags(eng->flags_n, 0);

    eng->queue.enqueueWriteBuffer(dText, false, 0, chunk.size()  * sizeof(char32_t), chunk.data(), nullptr, &evtUploadText);
    eng->queue.enqueueWriteBuffer(dFlags, false, 0, flags.size() * sizeof(char), flags.data(), nullptr, &evtUploadFlags);

    // run automaton kernel
    eng->kernelAutomaton.setArg(0, static_cast<cl_uint>(graph.n));
    eng->kernelAutomaton.setArg(1, static_cast<cl_uint>(graph.o));
    eng->kernelAutomaton.setArg(2, static_cast<cl_uint>(chunk.size()));
    eng->kernelAutomaton.setArg(3, static_cast<cl_uint>(eng->multi_input_n));
    eng->kernelAutomaton.setArg(4, dAutomatonData);
    eng->kernelAutomaton.setArg(5, dText);
    eng->kernelAutomaton.setArg(6, dOutput);
    eng->kernelAutomaton.setArg(7, dFlags);
    eng->kernelAutomaton.setArg(8, eng->oversize_cache * eng->group_size * sizeof(char32_t), nullptr);

    std::size_t totalSize = chunk.size() / eng->multi_input_n;
    if (chunk.size() % eng->multi_input_n != 0) {
        totalSize += 1;
    }
    totalSize = adjust_globalsize(totalSize, eng->group_size);
    eng->queue.enqueueNDRangeKernel(eng->kernelAutomaton, cl::NullRange, cl::NDRange(totalSize), cl::NDRange(eng->group_size), nullptr, &evtKernelAutomaton);

    // run transform kernel
    std::size_t globalsize = adjust_globalsize(chunk.size(), eng->group_size);
    eng->kernelTransform.setArg(0, dOutput);
    eng->kernelTransform.setArg(1, dScanbuffer0);
    eng->kernelTransform.setArg(2, static_cast<cl_uint>(chunk.size()));
    eng->queue.enqueueNDRangeKernel(eng->kernelTransform, cl::NullRange, cl::NDRange(globalsize), cl::NDRange(eng->group_size), nullptr, &evtKernelTransform);

    // run scan kernel
    std::size_t offset = 1;
    while (offset < chunk.size()) {
        evtsKernelScan.emplace_back();
        eng->kernelScan.setArg(0, dScanbuffer0);
        eng->kernelScan.setArg(1, dScanbuffer1);
        eng->kernelScan.setArg(2, static_cast<cl_uint>(chunk.size()));
        eng->kernelScan.setArg(3, static_cast<cl_uint>(offset));
        eng->queue.enqueueNDRangeKernel(eng->kernelScan, cl::NullRange, cl::NDRange(globalsize), cl::NDRange(eng->group_size), nullptr, &evtsKernelScan[evtsKernelScan.size() - 1]);
        std::swap(dScanbuffer0, dScanbuffer1);
        offset = offset << 1;
    }

    // run move kernel
    eng->kernelMove.setArg(0, dScanbuffer0);
    eng->kernelMove.setArg(1, dOutput);
    eng->kernelMove.setArg(2, dScanbuffer1);
    eng->kernelMove.setArg(3, static_cast<cl_uint>(chunk.size()));
    eng->queue.enqueueNDRangeKernel(eng->kernelMove, cl::NullRange, cl::NDRange(globalsize), cl::NDRange(eng->group_size), nullptr, &evtKernelMove);
    std::swap(dOutput, dScanbuffer1);

    // get output
    std::uint32_t outputSize;
    eng->queue.enqueueReadBuffer(dScanbuffer0, true, (chunk.size() - 1) * sizeof(cl_uint), 1 * sizeof(cl_uint), &outputSize, nullptr, &evtDownloadOutputSize);
    sanity_assert(outputSize <= chunk.size(), "outputSize must be at max the chunk size");

    std::vector<uint32_t> output(outputSize, 0);
    if (outputSize > 0) {
        eng->queue.enqueueReadBuffer(dOutput, false, 0, outputSize * sizeof(cl_uint), output.data(), nullptr, &evtDownloadOutput);
    }

    eng->queue.enqueueReadBuffer(dFlags, false, 0, flags.size() * sizeof(char), flags.data(), nullptr, &evtDownloadFlags);

    eng->queue.finish();

    if (printProfile) {
        std::cout << "Profiling data:" << std::endl
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
            << "  downloadOutputSize = " << getEventTimeMS(evtDownloadOutputSize) << "ms" << std::endl;
        if (outputSize > 0) {
            // only that that case the event got fired
            std::cout << "  downloadOutput     = " << getEventTimeMS(evtDownloadOutput) << "ms" << std::endl;
        }
        std::cout
            << "  downloadFlags      = " << getEventTimeMS(evtDownloadFlags) << "ms" << std::endl;
    }

    if (flags[eng->flag_stack_full]) {
        throw user_error("Automaton engine error: task stack was full!");
    }
    if (flags[eng->flag_iter_max]) {
        throw user_error("Automaton engine error: reached maximum iteration count!");
    }

    return output;
}
