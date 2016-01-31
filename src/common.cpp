#include "common.hpp"


internal_exception::internal_exception(const std::string& msg) : msg(msg) {}

const char* internal_exception::what() const noexcept {
    return msg.c_str();
}


user_error::user_error(const std::string& msg) : msg(msg) {}

const char* user_error::what() const noexcept {
    return msg.c_str();
}


sanity_error::sanity_error(const std::string& msg, std::string file, std::string func, std::size_t line) : msg(msg), file(file), func(func), line(line) {
    std::stringstream ss;
    ss << "Sanity check failed: \"" << msg << "\" @ " << file << ":" << func << ":" << line;
    what_cache = ss.str();
}

const char* sanity_error::what() const noexcept {
    return what_cache.c_str();
}
