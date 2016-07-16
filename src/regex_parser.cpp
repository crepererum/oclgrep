#include <cstdint>

#include <algorithm>
#include <exception>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/io.hpp>
#include <boost/locale.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/optional.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/variant.hpp>
#include <boost/variant/get.hpp>

#include "common.hpp"
#include "regex_parser.hpp"
#include "config.hpp"


namespace fusion = boost::fusion;
namespace x3 = boost::spirit::x3;


static_assert(sizeof(char32_t) == sizeof(wchar_t), "wchar is not 32bit, but it's required for spirit x3");


namespace ast {
    using optional_n = boost::optional<unsigned int>;

    using multiplier_amount = unsigned int;

    struct multiplier_range {
        optional_n min;
        optional_n max;

        multiplier_range() = default;
        multiplier_range(const optional_n& min, const optional_n& max) : min(min), max(max) {}
    };

    struct multiplier_plus {};
    struct multiplier_question {};
    struct multiplier_star {};

    using multiplier = boost::variant<multiplier_range, multiplier_amount, multiplier_plus, multiplier_question, multiplier_star>;

    using character = char32_t;

    struct character_range {
        character begin;
        character end;

        character_range() = default;
        character_range(character begin, character end) : begin(begin), end(end) {}
    };

    using characterclass_element = boost::variant<character_range, character>;

    using characterclass = std::vector<characterclass_element>;

    using word = std::vector<character>;

    using chunkcontent = boost::variant<characterclass, word>;

    struct chunk {
        chunkcontent content;
        boost::optional<multiplier> amount;
    };

    using regex = std::vector<chunk>;

    using fusion::operator<<;
}

BOOST_FUSION_ADAPT_STRUCT(ast::multiplier_range, min, max)
BOOST_FUSION_ADAPT_STRUCT(ast::character_range, begin, end)
BOOST_FUSION_ADAPT_STRUCT(ast::chunk, content, amount)


namespace parser {
    static x3::rule<class multiplier_amount, ast::multiplier_amount> multiplier_amount = "multiplier_amount";
    static x3::rule<class multiplier_range, ast::multiplier_range> multiplier_range = "multiplier_range";
    static x3::rule<class multiplier_plus, ast::multiplier_plus> multiplier_plus = "multiplier_plus";
    static x3::rule<class multiplier_question, ast::multiplier_question> multiplier_question = "multiplier_question";
    static x3::rule<class multiplier_star, ast::multiplier_star> multiplier_star = "multiplier_star";
    static x3::rule<class multiplier, ast::multiplier> multiplier = "multiplier";
    static x3::rule<class character, ast::character> character = "character";
    static x3::rule<class character_range, ast::character_range> character_range = "character_range";
    static x3::rule<class characterclass_element, ast::characterclass_element> characterclass_element = "characterclass_element";
    static x3::rule<class characterclass, ast::characterclass> characterclass = "characterclass";
    static x3::rule<class word, ast::word> word = "word";
    static x3::rule<class chunkcontent, ast::chunkcontent> chunkcontent = "chunkcontent";
    static x3::rule<class chunk, ast::chunk> chunk = "chunk";
    static x3::rule<class regex, ast::regex> regex = "regex";

    auto const multiplier_amount_def = '{' >> x3::uint_ >> '}';
    auto const multiplier_range_def = '{' >> -x3::uint_ >> ',' >> -x3::uint_ >> '}';
    auto const multiplier_plus_def = x3::omit['+'];
    auto const multiplier_question_def = x3::omit['?'];
    auto const multiplier_star_def = x3::omit['*'];
    auto const multiplier_def = multiplier_range | multiplier_amount | multiplier_plus | multiplier_question | multiplier_star;
    auto const character_def = x3::char_ - '[' - ']' - '{' - '}' - '+' - '*' - '?' - '-' - static_cast<wchar_t>(0x00000000) - static_cast<wchar_t>(0xffffffff);
    auto const character_range_def = character >> '-' >> character;
    auto const characterclass_element_def = character_range | character;
    auto const characterclass_def = '[' >> +(characterclass_element) >> ']';
    auto const wordelement_def = characterclass | character;
    auto const word_def = +character;
    auto const chunkcontent_def = characterclass | word;
    auto const chunk_def = chunkcontent >> (-multiplier);
    auto const regex_def = +chunk;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
    BOOST_SPIRIT_DEFINE(
        multiplier_amount,
        multiplier_range,
        multiplier_plus,
        multiplier_question,
        multiplier_star,
        multiplier,
        character,
        character_range,
        characterclass_element,
        characterclass,
        word,
        chunkcontent,
        chunk,
        regex
    )
#pragma clang diagnostic pop
}


ast::regex parse_ast(const std::u32string& input) {
    ast::regex result;
    auto it = input.begin();
    auto end = input.end();
    bool r = phrase_parse(it, end, parser::regex, x3::standard_wide::space, result);
    if (r && it == end) {
        return result;
    } else {
        std::string msg("malformed regex: ");
        auto pos = static_cast<std::size_t>(it - input.begin());
        std::stringstream errmsg;
        errmsg << msg << boost::locale::conv::utf_to_utf<char>(input) << std::endl
            << std::string(msg.size() + pos, ' ') << "^";
        throw user_error(errmsg.str());
    }
}


namespace graph {
    using slot_inner_t = std::vector<std::uint32_t>;
    using slot_t = std::shared_ptr<slot_inner_t>;

    slot_t make_slot(std::initializer_list<uint32_t> data) {
        return std::make_shared<slot_inner_t>(data);
    }

    struct node {
        std::vector<std::pair<char32_t, slot_t>> next;
        std::uint32_t id;

        node(std::uint32_t& id) : id(id++) {}
    };
    using node_t = std::shared_ptr<node>;
    using graph_t = std::vector<node_t>;
}


namespace transformers {
    using collection_slots_t = std::vector<graph::slot_t>;
    using transformer_result_t = std::pair<graph::graph_t, collection_slots_t>;

    class characterclass_element_visitor : public boost::static_visitor<ast::character_range> {
        public:
            ast::character_range operator()(const ast::character& character) const {
                return ast::character_range(character, character);
            }

            ast::character_range operator()(const ast::character_range& character_range) const {
                return character_range;
            }
    };

    class generic_visitor : public boost::static_visitor<transformer_result_t> {
        public:
            generic_visitor(std::uint32_t& id, collection_slots_t slots) : id(id), slots(slots) {}

        protected:
            std::uint32_t& id;
            collection_slots_t slots;
    };

    class character_transformer : public generic_visitor {
        public:
            using generic_visitor::generic_visitor;

            transformer_result_t operator()(const ast::character& character) const {
                // 1. create new node
                auto result = std::make_shared<graph::node>(id);

                // 2. connect last ones to this one
                for (auto& last : slots) {
                    last->push_back(result->id);
                }

                // 3. fill this one
                result->next.push_back(std::make_pair(0, graph::make_slot({serial::id_fail})));
                result->next.push_back(std::make_pair(character, graph::make_slot({})));
                result->next.push_back(std::make_pair(character + 1, graph::make_slot({serial::id_fail})));

                // 4. get slots
                collection_slots_t slots_new{std::get<1>(result->next[1])};

                // 5. done
                return {graph::graph_t{std::move(result)}, slots_new};
            }
    };

    class chunkcontent_transformer : public generic_visitor {
        public:
            using generic_visitor::generic_visitor;

            transformer_result_t operator()(const ast::word& word) const {
                graph::graph_t result_nodes;
                collection_slots_t slots_new = slots;
                for (const auto& character : word) {
                    auto sub_result = character_transformer(id, slots_new)(character);
                    result_nodes.insert(result_nodes.end(), std::get<0>(sub_result).begin(), std::get<0>(sub_result).end());
                    slots_new = std::get<1>(sub_result);
                }
                return std::make_pair(result_nodes, slots_new);
            }

            transformer_result_t operator()(const ast::characterclass& characterclass) const {
                // 1. create new node
                auto result = std::make_shared<graph::node>(id);

                // 2. connect last ones to this one
                for (auto& last : slots) {
                    last->push_back(result->id);
                }

                // 3. prepare and merge ranges
                std::vector<ast::character_range> ranges;
                for (const auto& x : characterclass) {
                    ranges.push_back(boost::apply_visitor(characterclass_element_visitor(), x));
                }
                std::sort(ranges.begin(), ranges.end(), [](const ast::character_range& a, const ast::character_range& b) {
                    return a.begin < b.begin;
                });
                std::vector<ast::character_range> ranges_dedup;
                for (const auto& x : ranges) {
                    if (ranges_dedup.empty()) {
                        ranges_dedup.push_back(x);
                    } else {
                        auto& last = ranges_dedup[ranges_dedup.size() - 1];
                        if (x.begin <= last.end + 1) {
                            last.end = x.end;
                        } else {
                            ranges_dedup.push_back(x);
                        }
                    }
                }
                if (ranges_dedup.size() > cfg::max_ranges) {
                    throw user_error("Too many ranges in character class!");
                }

                // 4. fill this one
                result->next.push_back(std::make_pair(0, graph::make_slot({serial::id_fail})));
                collection_slots_t slots_new{};
                char32_t last_char = 0;
                for (const auto& r : ranges_dedup) {
                    if (last_char > 0) { // add range up to this element, skip first
                        result->next.push_back(std::make_pair(last_char + 1, graph::make_slot({serial::id_fail})));
                    }
                    result->next.push_back(std::make_pair(r.begin, graph::make_slot({})));
                    slots_new.push_back(std::get<1>(result->next[result->next.size() - 1]));
                    last_char = r.end;
                }
                result->next.push_back(std::make_pair(last_char + 1, graph::make_slot({serial::id_fail})));

                // 4. done
                return {graph::graph_t{std::move(result)}, slots_new};
            }
    };

    class multiplier_transformator : public generic_visitor {
        public:
            multiplier_transformator(std::uint32_t& id, collection_slots_t slots, const ast::chunkcontent& content) : generic_visitor(id, slots), content(content) {}

            transformer_result_t operator()(const ast::multiplier_amount& amount) const {
                return doit(amount, ast::optional_n(amount));
            }

            transformer_result_t operator()(const ast::multiplier_range& range) const {
                std::size_t min = 0;
                if (range.min) {
                    min = *(range.min);
                }
                if (range.max && *(range.max) < min) {
                    throw user_error("Illegal regex multiplier!");
                }
                return doit(min, range.max);
            }

            transformer_result_t operator()(const ast::multiplier_plus& /*plus*/) const {
                return doit(1, boost::none);
            }

            transformer_result_t operator()(const ast::multiplier_question& /*question*/) const {
                return doit(0, ast::optional_n(1));
            }

            transformer_result_t operator()(const ast::multiplier_star& /*star*/) const {
                return doit(0, boost::none);
            }

        protected:
            const ast::chunkcontent& content;

            transformer_result_t doit(std::size_t min, ast::optional_n max) const {
                if (min > cfg::max_multiplier) {
                    throw user_error("multiplier minimum is too large!");
                }
                if (max && *max > cfg::max_multiplier) {
                    throw user_error("multiplier maximum is too large!");
                }
                graph::graph_t nodes_result;

                // 1. start with the words we need at least
                collection_slots_t slots_current = slots;
                graph::graph_t nodes_current;
                std::size_t i = 0;
                for (; i < min; ++i) {
                    auto sub_result = boost::apply_visitor(chunkcontent_transformer(id, slots_current), content);
                    nodes_result.insert(nodes_result.end(), nodes_current.begin(), nodes_current.end());
                    std::tie(nodes_current, slots_current) = sub_result;
                }

                // 2. add optional words
                collection_slots_t slots_result;
                if (max) {
                    // some maximum => emit nodes, add all of them to result
                    // also emit 1 additional word, so we can link it to FAIL

                    // a) create nodes
                    for (; i <= *max; ++i) {
                        auto sub_result = boost::apply_visitor(chunkcontent_transformer(id, slots_current), content);
                        nodes_result.insert(nodes_result.end(), nodes_current.begin(), nodes_current.end());
                        slots_result.insert(slots_result.end(), slots_current.begin(), slots_current.end());
                        std::tie(nodes_current, slots_current) = sub_result;
                    }

                    // b) now link the last node to FAIL, do NOT add it to slots_result
                    nodes_result.insert(nodes_result.end(), nodes_current.begin(), nodes_current.end());
                    for (auto& slot : slots_current) {
                        slot->push_back(serial::id_fail);
                    }
                } else {
                    // no max => create new word and link slots to nodes current (loop)

                    // a) create node
                    auto sub_result = boost::apply_visitor(chunkcontent_transformer(id, slots_current), content);
                    nodes_result.insert(nodes_result.end(), nodes_current.begin(), nodes_current.end());
                    slots_result.insert(slots_result.end(), slots_current.begin(), slots_current.end());
                    std::tie(nodes_current, slots_current) = sub_result;

                    // b) link it
                    nodes_result.insert(nodes_result.end(), nodes_current.begin(), nodes_current.end());
                    for (auto& slot : slots_current) {
                        slot->push_back(nodes_current[0]->id);
                    }
                    slots_result.insert(slots_result.end(), slots_current.begin(), slots_current.end());
                }

                // done
                return {nodes_result, slots_result};
            }
    };

    class chunk_transfomer : public generic_visitor {
        public:
            using generic_visitor::generic_visitor;

            transformer_result_t operator()(const ast::chunk& chunk) const {
                if (chunk.amount) {
                    return boost::apply_visitor(multiplier_transformator(id, slots, chunk.content), *(chunk.amount));
                } else {
                    return boost::apply_visitor(chunkcontent_transformer(id, slots), chunk.content);
                }
            }
    };
}


graph::graph_t ast_to_graph(const ast::regex& r) {
    // start graph
    std::uint32_t id = 0;
    graph::graph_t nodes;
    nodes.push_back(std::make_shared<graph::node>(id)); // FAIL node
    nodes.push_back(std::make_shared<graph::node>(id)); // OK node
    transformers::collection_slots_t slots; // no slots yet

    // iterate over entire regex
    for (const auto& chunk : r) {
        auto sub_result = transformers::chunk_transfomer(id, slots)(chunk);
        nodes.insert(nodes.end(), std::get<0>(sub_result).begin(), std::get<0>(sub_result).end());
        slots = std::get<1>(sub_result);
    }

    // fill remaining slots with good outcome
    for (auto& last : slots) {
        last->push_back(serial::id_ok);
    }

    sanity_assert(id == nodes.size(), "Some nodes are lost :(");;
    return nodes;
}


template <typename T>
void write_to_buffer(serial::buffer& b, std::size_t base, T element) {
    static_assert(sizeof(serial::word) == 4, "ups, need to rewrite the serializer!");
    static_assert(sizeof(T) % sizeof(serial::word) == 0, "that doesn't match the word boundaries!");

    // XXX: check big vs little endian!
    for (std::size_t i = 0; i < sizeof(T) / sizeof(serial::word); ++i) {
        b[base + i] = (element >> (8 * 4 * i)) & 0xffffffff;
    }
}


serial::graph serialize(const graph::graph_t& g) {
    // 1. calculate maximum numbers for allocations
    std::size_t n = g.size();
    std::size_t m = 0;
    std::size_t o = 0;
    for (const auto& node : g) {
        m = std::max(m, node->next.size());
        for (const auto& value_slot : node->next) {
            o = std::max(o, std::get<1>(value_slot)->size());
        }
    }

    // 2. create buffer
    serial::graph result(n, o);
    // at this point, the dispatch table exists

    // 3. write data
    for (std::size_t i_node = 0; i_node < n; ++i_node) {
        const auto& node = g[i_node];
        std::size_t base_node = result.size();

        // write current size to dispatch table
        write_to_buffer(result.data, i_node, static_cast<serial::id>(base_node));

        // start node by writing its size
        result.grow(1);
        write_to_buffer(result.data, base_node, static_cast<serial::id>(node->next.size()));

        // write node body
        std::size_t base_node_body = base_node + 1;
        for (std::size_t i_value_slot = 0; i_value_slot < node->next.size(); ++i_value_slot) {
            const auto& value_slot = node->next[i_value_slot];

            std::size_t base_value_slot = base_node_body + i_value_slot * (1 + o);
            serial::character c = std::get<0>(value_slot);

            // write character that belongs to slot
            result.grow(1);
            write_to_buffer(result.data, base_value_slot, c);

            // write fixed size, sorted, dedup data to slot
            std::size_t base_value_slot_payload = base_value_slot + 1;
            std::vector<std::uint32_t> entries_sorted(std::get<1>(value_slot)->begin(), std::get<1>(value_slot)->end());
            std::sort(entries_sorted.begin(), entries_sorted.end());
            entries_sorted.erase(std::unique(entries_sorted.begin(), entries_sorted.end()), entries_sorted.end());
            for (std::size_t i_slot_entry = 0; i_slot_entry < o; ++i_slot_entry) {
                result.grow(1);
                if (i_slot_entry < entries_sorted.size()) {
                    std::size_t base_slot_entry = base_value_slot_payload + i_slot_entry;
                    serial::id id = entries_sorted[i_slot_entry];
                    write_to_buffer(result.data, base_slot_entry, id);
                } // else => data is 0
            }
        }
    }

    // 4. done
    return result;
}


serial::graph string_to_graph(const std::u32string& input) {
    auto r = parse_ast(input);
    if (r.empty()) {
        throw user_error("Empty regex is not allowed!");
    }

    auto g = ast_to_graph(r);

    return serialize(g);
}
