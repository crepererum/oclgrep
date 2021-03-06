#pragma once

#include <cstdint>

#include <exception>
#include <sstream>
#include <string>
#include <vector>


static_assert(sizeof(wchar_t) == 4, "wchar_t does not seem to encode UTF32");


class internal_exception : public std::exception {
    public:
        internal_exception(const std::string& msg);
        const char* what() const noexcept;
    private:
        std::string msg;
};


class user_error : public std::exception {
    public:
        user_error(const std::string& msg);
        const char* what() const noexcept;
    private:
        std::string msg;
};


class sanity_error : public std::exception {
    public:
        sanity_error(const std::string& msg, std::string file, std::string func, std::size_t line);
        const char* what() const noexcept;

    private:
        std::string msg;
        std::string file;
        std::string func;
        std::size_t line;
        std::string what_cache;
};


#define sanity_assert(ok, msg) if (!(ok)) throw sanity_error((msg), __FILE__, __func__, __LINE__);


namespace serial {
    using character = char32_t;
    using id = std::uint32_t;
    using word = std::uint32_t;
    using buffer = std::vector<word>;

    static_assert(sizeof(character) == sizeof(word), "that won't serialize very well!");
    static_assert(sizeof(id) == sizeof(word), "that won't serialize very well!");

    struct graph {
        std::size_t n;                  // number of nodes
        std::size_t o;                  // maximum cardinality of multi-edges
        buffer data; // size = n * m * (sizeof(character) + o * sizeof(id))

        graph(std::size_t n, std::size_t o) : n(n), o(o), data(n, 0) {} // 0 is also the id of fail, so good for unused space

        std::size_t size() const {
            return data.size();
        }

        void grow(std::size_t i) {
            data.resize(data.size() + i, 0);
        }
    };

    constexpr id id_fail = 0;
    constexpr id id_ok = 1;
    constexpr id id_begin = 2;
}


constexpr std::uint32_t calc_alignement_mask(std::size_t n_bytes) {
    constexpr std::uint32_t full = ~static_cast<std::uint32_t>(0);
    std::uint32_t base = static_cast<std::uint32_t>(1) << n_bytes;
    return full - base + static_cast<std::uint32_t>(1);
}
