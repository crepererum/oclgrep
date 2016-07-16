#include <string>
#include <fstream>
#include <iostream>
#include <streambuf>

#include <boost/locale.hpp>

#include "regex_parser.hpp"


int main(int argc, char** argv) {
    if (argc == 2) {
        try {
            // pre-check
            boost::locale::generator gen;
            std::locale loc = gen("");
            if (!std::use_facet<boost::locale::info>(loc).utf8()) {
                throw user_error("sorry, this program only works on UTF8 systems");
            }

            // get AFL data
            std::ifstream t(argv[1]);
            std::string regex_utf8(
                (std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>()
            );

            // prepare data
            auto regex_utf32 = boost::locale::conv::utf_to_utf<char32_t>(regex_utf8);

            // TEST
            string_to_graph(regex_utf32);

            // done
            return 0;
        } catch (user_error& e) {
            std::cerr << e.what() << std::endl;
            return 0; // expected, because fuzzing
        }
    } else {
        std::cerr << "fun with `string_to_graph FILE_CONTAINING_REGEX`" << std::endl;
        return 0;
    }
}
