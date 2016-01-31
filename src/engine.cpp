#include <cstdint>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include "common.hpp"


static_assert(sizeof(std::uint32_t) == sizeof(cl_uint), "OpenCL uint has to be 32bit");
static_assert(sizeof(char32_t) == sizeof(std::uint32_t), "uint32 and char32_t have to have the same size");
static_assert(sizeof(char) == sizeof(cl_char), "OpenCL char has not the same size as host char");


cl::Program buildProgramFromFile(const std::string& fname, const cl::Context& context, const std::vector<cl::Device>& devices, const std::map<std::string, std::string>& defines) {
    // open file
    std::ifstream file(fname.c_str());
    if (file.fail()) {
        throw user_error("Cannot open file " + fname);
    }

    // dump file content to string
    std::string sourceCode;
    file.seekg(0, std::ios::end);
    sourceCode.reserve(static_cast<std::size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    sourceCode.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // create build options string
    std::stringstream buildOptionsSS;
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

std::vector<std::uint32_t> runEngine(const serial::graph& graph, const std::u32string& fcontent, bool printProfile) {
    constexpr std::uint32_t cache_mask      = calc_alignement_mask(5); // sets cache alignement of local text cache base 32
    constexpr std::uint32_t flag_iter_max   = 1;                       // index of "we've reached too many iteratios"-flag
    constexpr std::uint32_t flag_stack_full = 0;                       // index of "thread-local stack was too small"-flag
    constexpr std::uint32_t flags_n         = 2;                       // number of flags
    constexpr std::uint32_t group_size      = 128;                     // OpenCL group size
    constexpr std::uint32_t max_iter_count  = 512;                     // limits number of iterations to prevent timeouts
    constexpr std::uint32_t max_stack_size  = 16;                      // limits thread-local stack
    constexpr std::uint32_t multi_input_n   = 2;                       // load-balancing by using multiple start postions per thread
    constexpr std::uint32_t oversize_cache  = 4;                       // cache_size=group_size*oversize_cache
    constexpr std::uint32_t result_fail     = 0xffffffff;              // placeholder for "FAIL" results of automaton
    constexpr std::uint32_t sync_count      = 32;                      // controls after how many iterations group threads sync

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

    std::map<std::string, std::string> buildDefines{
        {"CACHE_MASK",      std::to_string(cache_mask)},
        {"FLAG_ITER_MAX",   std::to_string(flag_iter_max)},
        {"FLAG_STACK_FULL", std::to_string(flag_stack_full)},
        {"ID_BEGIN",        std::to_string(serial::id_begin)},
        {"ID_FAIL",         std::to_string(serial::id_fail)},
        {"ID_OK",           std::to_string(serial::id_ok)},
        {"MAX_ITER_COUNT",  std::to_string(max_iter_count)},
        {"MAX_STACK_SIZE",  std::to_string(max_stack_size)},
        {"OVERSIZE_CACHE",  std::to_string(oversize_cache)},
        {"RESULT_FAIL",     std::to_string(result_fail)},
        {"SYNC_COUNT",      std::to_string(sync_count)},
    };

    cl::Program programAutomaton = buildProgramFromFile("automaton.cl", context, devices, buildDefines);
    cl::Kernel kernelAutomaton(programAutomaton, "automaton");

    std::vector<char> flags(flags_n, 0);

    cl::Buffer dAutomatonData(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        graph.size() * sizeof(std::uint8_t),
        const_cast<void*>(static_cast<const void*>(graph.data.data())) // that's ok, trust me ;)
    );

    cl::Buffer dText(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        fcontent.size() * sizeof(char32_t),
        const_cast<void*>(static_cast<const void*>(fcontent.data())) // that's ok, trust me ;)
    );

    cl::Buffer dOutput(
        context,
        CL_MEM_READ_WRITE,
        fcontent.size() * sizeof(cl_uint),
        nullptr
    );

    cl::Buffer dFlags(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        flags.size() * sizeof(char),
        flags.data()
    );

    kernelAutomaton.setArg(0, static_cast<cl_uint>(graph.n));
    kernelAutomaton.setArg(1, static_cast<cl_uint>(graph.m));
    kernelAutomaton.setArg(2, static_cast<cl_uint>(graph.o));
    kernelAutomaton.setArg(3, static_cast<cl_uint>(fcontent.size()));
    kernelAutomaton.setArg(4, static_cast<cl_uint>(multi_input_n));
    kernelAutomaton.setArg(5, dAutomatonData);
    kernelAutomaton.setArg(6, dText);
    kernelAutomaton.setArg(7, dOutput);
    kernelAutomaton.setArg(8, dFlags);
    kernelAutomaton.setArg(9, oversize_cache * group_size, nullptr);

    cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling);
    cl::Event evtKernelAutomaton;

    std::size_t totalSize = fcontent.size() / multi_input_n;
    if (fcontent.size() % multi_input_n != 0) {
        totalSize += 1;
    }
    if (totalSize % group_size != 0) {
        totalSize += group_size - totalSize % group_size;
    }
    queue.enqueueNDRangeKernel(kernelAutomaton, cl::NullRange, cl::NDRange(totalSize), cl::NDRange(group_size), nullptr, &evtKernelAutomaton);

    std::vector<uint32_t> output(fcontent.size(), 0);
    queue.enqueueReadBuffer(dOutput, false, 0, fcontent.size() * sizeof(cl_uint), output.data());

    queue.enqueueReadBuffer(dFlags, false, 0, flags.size() * sizeof(char), flags.data());

    queue.finish();

    if (printProfile) {
        std::cout << "Profiling data:" << std::endl
            << "  kernelAutomaton = " << getEventTimeMS(evtKernelAutomaton) << "ms" << std::endl;
    }

    if (flags[flag_stack_full]) {
        throw user_error("Automaton engine error: task stack was full!");
    }
    if (flags[flag_iter_max]) {
        throw user_error("Automaton engine error: reached maximum iteration count!");
    }

    return output;
}
