#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <string>

#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "common.hpp"

namespace po = boost::program_options;

std::string readfile(const std::string& fname) {
    std::ifstream input(fname, std::ios::binary);
    if (!input.good()) {
        throw user_error("file does not exist!");
    }
    return std::string(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>())
    );
}

void print_graph(const serial::graph& g) {
    std::cout << "Graph (n=" << g.n << ", o=" << g.o << ", size=" << g.size() << "byte):" << std::endl;

    for (std::size_t i_node = 0; i_node < g.n; ++i_node) {
        std::size_t base_node = *reinterpret_cast<const serial::id*>(&g.data[i_node * sizeof(serial::id)]);
        std::size_t m = *reinterpret_cast<const serial::id*>(&g.data[base_node]);
        std::cout << "  node" << i_node << " (m=" << m << "):" << std::endl;

        std::size_t base_node_body = base_node + sizeof(serial::id);
        for (std::size_t i_value_slot = 0; i_value_slot < m; ++i_value_slot) {
            std::size_t base_value_slot = base_node_body + i_value_slot * (sizeof(serial::character) + g.o * sizeof(serial::id));
            serial::character c = *reinterpret_cast<const serial::character*>(&g.data[base_value_slot]);
            std::cout << "    " << c << " => [";

            std::size_t base_slot = base_value_slot + sizeof(serial::character);
            for (std::size_t i_entry = 0; i_entry < g.o; ++i_entry) {
                std::size_t base_entry = base_slot + i_entry * sizeof(serial::id);
                serial::id id = *reinterpret_cast<const serial::id*>(&g.data[base_entry]);

                if (i_entry > 0) {
                    std::cout << ",";
                }
                std::cout << id;
            }

            std::cout << "]" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    try {
        // before we start, check if we're working on an UTF8 system
        boost::locale::generator gen;
        std::locale loc = gen("");
        if (!std::use_facet<boost::locale::info>(loc).utf8()) {
            throw user_error("sorry, this program only works on UTF8 systems");
        }

        // parse command line argument
        std::string regex_utf8;
        std::string file;

        po::options_description desc("Allowed options");
        desc.add_options()
            ("regex", po::value(&regex_utf8)->required(), "regex that should be matched")
            ("file", po::value(&file)->required(), "file where we look for the regex")
            ("normalize-regex", "apply NFKC normalization to regex")
            ("normalize-file", "apply NFKC normalization to data from input file")
            ("print-graph", "print graph data to stdout")
            ("print-profile", "print OpenCL profiling data to stdout")
            ("no-output", "do not print actual output (for debug reasons)")
            ("help", "produce help message")
        ;

        po::positional_options_description p;
        p.add("regex", 1);
        p.add("file", 1);

        po::variables_map vm;
        po::store(
            po::command_line_parser(argc, argv).options(desc).positional(p).run(),
            vm
        );

        if (vm.count("help")) {
            std::cout << "oclgrep REGEX FILE" << std::endl
                << desc << std::endl;
            return 1;
        }

        try {
            po::notify(vm);
        } catch(std::exception& e) {
            throw user_error(e.what());
        }

        // load file
        auto fcontent_utf8 = readfile(file);
        if (fcontent_utf8.empty()) {
            throw user_error("Empty files cannot be processed!");
        }

        // convert external UTF8 data to UTF32
        auto regex_utf32 = boost::locale::conv::utf_to_utf<char32_t>(regex_utf8);
        auto fcontent_utf32 = boost::locale::conv::utf_to_utf<char32_t>(fcontent_utf8);

        // normalize regex and input
        if (vm.count("normalize-regex")) {
            regex_utf32 = boost::locale::conv::utf_to_utf<char32_t>(
                boost::locale::normalize(
                    boost::locale::conv::utf_to_utf<wchar_t>(regex_utf32),
                    boost::locale::norm_nfkc
                )
            );
        }
        if (vm.count("normalize-file")) {
            // XXX: we'll have a problem with indices afterwards :(
            fcontent_utf32 = boost::locale::conv::utf_to_utf<char32_t>(
                boost::locale::normalize(
                    boost::locale::conv::utf_to_utf<wchar_t>(fcontent_utf32),
                    boost::locale::norm_nfkc
                )
            );
        }

        // parse regex to graph
        auto graph = string_to_graph(regex_utf32);
        if (vm.count("print-graph")) {
            print_graph(graph);
        }

        // tada...
        auto result = runEngine(graph, fcontent_utf32, vm.count("print-profile"));

        if (!vm.count("no-output")) {
            for (const auto& idx : result) {
                std::cout << idx << std::endl;
            }
        }
    } catch (user_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (std::exception& e) {
        std::cerr
            << "=========================================================================" << std::endl
            << "there was an internal error, please report this as a bug"                  << std::endl
            << "================================= ERROR =================================" << std::endl
            << e.what()                                                                    << std::endl
            << "=========================================================================" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
