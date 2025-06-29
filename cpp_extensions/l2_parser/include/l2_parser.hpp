#ifndef L2_PARSER_HPP
#define L2_PARSER_HPP

#include <vector>
#include <string>
#include <utility>

namespace l2_parser {

/**
 * Represents a price-quantity pair in the order book
 */
using PriceLevel = std::pair<double, double>;

/**
 * Represents a list of price levels (bid or ask)
 */
using PriceLevels = std::vector<PriceLevel>;

/**
 * Result structure for parsing operations
 */
struct ParseResult {
    PriceLevels levels;
    bool success;
    std::string error_message;
    
    ParseResult() : success(false) {}
    ParseResult(const PriceLevels& l) : levels(l), success(true) {}
    ParseResult(const std::string& error) : success(false), error_message(error) {}
};

/**
 * High-performance L2 order book data parser
 * 
 * Parses string representations of nested arrays like:
 * "[[684.5, 0.123], [684.6, 0.456], [684.7, 0.789]]"
 * 
 * Optimized for speed with minimal memory allocations.
 */
class L2Parser {
private:
    // Internal parsing state
    const char* current_pos;
    const char* end_pos;
    
    // Helper methods
    bool skip_whitespace();
    bool parse_number(double& result);
    bool parse_price_level(PriceLevel& level);
    bool expect_char(char expected);
    
public:
    /**
     * Parse a string containing L2 order book data
     * 
     * @param input String representation of price levels
     * @return ParseResult containing parsed data or error information
     */
    ParseResult parse(const std::string& input);
    
    /**
     * Parse multiple L2 strings in batch for better performance
     * 
     * @param inputs Vector of input strings
     * @return Vector of ParseResults
     */
    std::vector<ParseResult> parse_batch(const std::vector<std::string>& inputs);
    
    /**
     * Validate input string format without full parsing
     * 
     * @param input String to validate
     * @return true if format appears valid
     */
    bool validate_format(const std::string& input);
};

/**
 * Utility functions for common operations
 */
namespace utils {
    /**
     * Convert ParseResult to flat arrays for easier Python integration
     */
    void to_flat_arrays(const ParseResult& result, 
                       std::vector<double>& prices, 
                       std::vector<double>& quantities);
    
    /**
     * Estimate memory requirements for parsing
     */
    size_t estimate_memory_usage(const std::string& input);
    
    /**
     * Get parser statistics
     */
    struct ParserStats {
        size_t total_parsed;
        size_t successful_parses;
        size_t failed_parses;
        double average_levels_per_parse;
    };
    
    ParserStats get_stats();
    void reset_stats();
}

} // namespace l2_parser

#endif // L2_PARSER_HPP
