#include "../include/l2_parser.hpp"
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>

namespace l2_parser {

// Global statistics
static utils::ParserStats global_stats = {0, 0, 0, 0.0};

bool L2Parser::skip_whitespace() {
    while (current_pos < end_pos && std::isspace(*current_pos)) {
        current_pos++;
    }
    return current_pos < end_pos;
}

bool L2Parser::expect_char(char expected) {
    if (!skip_whitespace()) return false;
    if (*current_pos == expected) {
        current_pos++;
        return true;
    }
    return false;
}

bool L2Parser::parse_number(double& result) {
    if (!skip_whitespace()) return false;
    
    char* end_ptr;
    const char* start = current_pos;
    
    // Use strtod for fast, robust number parsing
    result = std::strtod(current_pos, &end_ptr);
    
    // Check if any characters were consumed
    if (end_ptr == current_pos) {
        return false;
    }
    
    // Update position
    current_pos = end_ptr;
    
    // Validate the result (check for overflow/underflow)
    if (result == HUGE_VAL || result == -HUGE_VAL) {
        return false;
    }
    
    return true;
}

bool L2Parser::parse_price_level(PriceLevel& level) {
    // Expect opening bracket for price level: '['
    if (!expect_char('[')) {
        return false;
    }
    
    // Parse price (first number)
    if (!parse_number(level.first)) {
        return false;
    }
    
    // Expect comma separator
    if (!expect_char(',')) {
        return false;
    }
    
    // Parse quantity (second number)
    if (!parse_number(level.second)) {
        return false;
    }
    
    // Expect closing bracket for price level: ']'
    if (!expect_char(']')) {
        return false;
    }
    
    return true;
}

ParseResult L2Parser::parse(const std::string& input) {
    // Update global statistics
    global_stats.total_parsed++;
    
    // Handle empty or null input
    if (input.empty()) {
        global_stats.failed_parses++;
        return ParseResult("Empty input string");
    }
    
    // Initialize parsing state
    current_pos = input.c_str();
    end_pos = current_pos + input.length();
    
    PriceLevels levels;
    levels.reserve(32); // Reserve space for typical order book depth
    
    // Expect opening bracket for the array: '['
    if (!expect_char('[')) {
        global_stats.failed_parses++;
        return ParseResult("Expected opening bracket '['");
    }
    
    // Handle empty array case
    if (!skip_whitespace()) {
        global_stats.failed_parses++;
        return ParseResult("Unexpected end of input");
    }
    
    if (*current_pos == ']') {
        current_pos++;
        global_stats.successful_parses++;
        return ParseResult(levels); // Empty array is valid
    }
    
    // Parse price levels
    while (current_pos < end_pos) {
        PriceLevel level;
        
        if (!parse_price_level(level)) {
            global_stats.failed_parses++;
            size_t pos = current_pos - input.c_str();
            return ParseResult("Failed to parse price level at position " + std::to_string(pos));
        }
        
        levels.push_back(level);
        
        // Skip whitespace and check what's next
        if (!skip_whitespace()) {
            global_stats.failed_parses++;
            return ParseResult("Unexpected end of input");
        }
        
        if (*current_pos == ']') {
            // End of array
            current_pos++;
            break;
        } else if (*current_pos == ',') {
            // More elements to come
            current_pos++;
            continue;
        } else {
            // Unexpected character
            global_stats.failed_parses++;
            return ParseResult("Expected ',' or ']' after price level");
        }
    }
    
    // Verify we reached the end properly
    skip_whitespace();
    if (current_pos < end_pos) {
        global_stats.failed_parses++;
        return ParseResult("Unexpected characters after closing bracket");
    }
    
    // Update statistics
    global_stats.successful_parses++;
    global_stats.average_levels_per_parse = 
        (global_stats.average_levels_per_parse * (global_stats.successful_parses - 1) + levels.size()) 
        / global_stats.successful_parses;
    
    return ParseResult(levels);
}

std::vector<ParseResult> L2Parser::parse_batch(const std::vector<std::string>& inputs) {
    std::vector<ParseResult> results;
    results.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        results.push_back(parse(input));
    }
    
    return results;
}

bool L2Parser::validate_format(const std::string& input) {
    if (input.empty()) return false;
    
    // Quick validation - check if it looks like a nested array
    auto trimmed_start = input.find_first_not_of(" \t\n\r");
    auto trimmed_end = input.find_last_not_of(" \t\n\r");
    
    if (trimmed_start == std::string::npos || trimmed_end == std::string::npos) {
        return false;
    }
    
    // Should start with '[' and end with ']'
    if (input[trimmed_start] != '[' || input[trimmed_end] != ']') {
        return false;
    }
    
    // Quick check for balanced brackets
    int bracket_count = 0;
    for (size_t i = trimmed_start; i <= trimmed_end; i++) {
        if (input[i] == '[') bracket_count++;
        else if (input[i] == ']') bracket_count--;
        
        if (bracket_count < 0) return false;
    }
    
    return bracket_count == 0;
}

namespace utils {

void to_flat_arrays(const ParseResult& result, 
                   std::vector<double>& prices, 
                   std::vector<double>& quantities) {
    if (!result.success) {
        prices.clear();
        quantities.clear();
        return;
    }
    
    size_t size = result.levels.size();
    prices.reserve(size);
    quantities.reserve(size);
    
    for (const auto& level : result.levels) {
        prices.push_back(level.first);
        quantities.push_back(level.second);
    }
}

size_t estimate_memory_usage(const std::string& input) {
    // Rough estimate: each price level needs ~32 bytes (2 doubles + overhead)
    // Count potential levels by counting '[' characters (excluding the outer one)
    size_t bracket_count = std::count(input.begin(), input.end(), '[');
    if (bracket_count > 0) bracket_count--; // Subtract outer bracket
    
    return bracket_count * 32 + input.size(); // Add input string size
}

ParserStats get_stats() {
    return global_stats;
}

void reset_stats() {
    global_stats = {0, 0, 0, 0.0};
}

} // namespace utils

} // namespace l2_parser
