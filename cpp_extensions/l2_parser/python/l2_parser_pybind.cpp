#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/l2_parser.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_l2_parser, m) {
    m.doc() = "High-performance L2 order book data parser";
    
    // Define the PriceLevel type (simplified approach)
    py::class_<l2_parser::PriceLevel>(m, "PriceLevel")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_property_readonly("price", [](const l2_parser::PriceLevel& level) { return level.first; })
        .def_property_readonly("quantity", [](const l2_parser::PriceLevel& level) { return level.second; })
        .def("__repr__", [](const l2_parser::PriceLevel& level) {
            return "PriceLevel(price=" + std::to_string(level.first) + 
                   ", quantity=" + std::to_string(level.second) + ")";
        });
    
    // Define the ParseResult type
    py::class_<l2_parser::ParseResult>(m, "ParseResult")
        .def(py::init<>())
        .def_readonly("levels", &l2_parser::ParseResult::levels)
        .def_readonly("success", &l2_parser::ParseResult::success)
        .def_readonly("error_message", &l2_parser::ParseResult::error_message)
        .def("__bool__", [](const l2_parser::ParseResult& result) {
            return result.success;
        })
        .def("__len__", [](const l2_parser::ParseResult& result) {
            return result.levels.size();
        })
        .def("to_lists", [](const l2_parser::ParseResult& result) {
            if (!result.success) {
                return py::make_tuple(py::list(), py::list());
            }
            py::list prices, quantities;
            for (const auto& level : result.levels) {
                prices.append(level.first);
                quantities.append(level.second);
            }
            return py::make_tuple(prices, quantities);
        })
        .def("to_numpy", [](const l2_parser::ParseResult& result) {
            if (!result.success) {
                return py::make_tuple(
                    py::array_t<double>(0),
                    py::array_t<double>(0)
                );
            }
            
            size_t size = result.levels.size();
            auto prices = py::array_t<double>(size);
            auto quantities = py::array_t<double>(size);
            
            auto prices_ptr = static_cast<double*>(prices.mutable_unchecked<1>().mutable_data(0));
            auto quantities_ptr = static_cast<double*>(quantities.mutable_unchecked<1>().mutable_data(0));
            
            for (size_t i = 0; i < size; ++i) {
                prices_ptr[i] = result.levels[i].first;
                quantities_ptr[i] = result.levels[i].second;
            }
            
            return py::make_tuple(prices, quantities);
        })
        .def("__repr__", [](const l2_parser::ParseResult& result) {
            if (result.success) {
                return "ParseResult(success=True, levels=" + 
                       std::to_string(result.levels.size()) + ")";
            } else {
                return "ParseResult(success=False, error='" + 
                       result.error_message + "')";
            }
        });
    
    // Define the main L2Parser class
    py::class_<l2_parser::L2Parser>(m, "L2Parser")
        .def(py::init<>())
        .def("parse", &l2_parser::L2Parser::parse,
             "Parse a single L2 string",
             py::arg("input"))
        .def("parse_batch", &l2_parser::L2Parser::parse_batch,
             "Parse multiple L2 strings in batch",
             py::arg("inputs"))
        .def("validate_format", &l2_parser::L2Parser::validate_format,
             "Validate L2 string format without full parsing",
             py::arg("input"));
    
    // Define ParserStats
    py::class_<l2_parser::utils::ParserStats>(m, "ParserStats")
        .def_readonly("total_parsed", &l2_parser::utils::ParserStats::total_parsed)
        .def_readonly("successful_parses", &l2_parser::utils::ParserStats::successful_parses)
        .def_readonly("failed_parses", &l2_parser::utils::ParserStats::failed_parses)
        .def_readonly("average_levels_per_parse", &l2_parser::utils::ParserStats::average_levels_per_parse)
        .def("__repr__", [](const l2_parser::utils::ParserStats& stats) {
            return "ParserStats(total=" + std::to_string(stats.total_parsed) +
                   ", success=" + std::to_string(stats.successful_parses) +
                   ", failed=" + std::to_string(stats.failed_parses) +
                   ", avg_levels=" + std::to_string(stats.average_levels_per_parse) + ")";
        });
    
    // Utility functions
    m.def("get_stats", &l2_parser::utils::get_stats, "Get parser statistics");
    m.def("reset_stats", &l2_parser::utils::reset_stats, "Reset parser statistics");
    m.def("estimate_memory_usage", &l2_parser::utils::estimate_memory_usage,
          "Estimate memory usage for parsing a string", py::arg("input"));
    
    // Convenience functions for easier Python integration
    m.def("parse_l2_string", [](const std::string& input) {
        l2_parser::L2Parser parser;
        return parser.parse(input);
    }, "Parse a single L2 string (convenience function)", py::arg("input"));
    
    m.def("parse_l2_to_lists", [](const std::string& input) {
        l2_parser::L2Parser parser;
        auto result = parser.parse(input);
        
        if (!result.success) {
            throw std::runtime_error("Parsing failed: " + result.error_message);
        }
        
        py::list prices, quantities;
        for (const auto& level : result.levels) {
            prices.append(level.first);
            quantities.append(level.second);
        }
        return py::make_tuple(prices, quantities);
    }, "Parse L2 string and return as Python lists", py::arg("input"));
    
    m.def("parse_l2_to_numpy", [](const std::string& input) {
        l2_parser::L2Parser parser;
        auto result = parser.parse(input);
        
        if (!result.success) {
            throw std::runtime_error("Parsing failed: " + result.error_message);
        }
        
        size_t size = result.levels.size();
        auto prices = py::array_t<double>(size);
        auto quantities = py::array_t<double>(size);
        
        auto prices_ptr = static_cast<double*>(prices.mutable_unchecked<1>().mutable_data(0));
        auto quantities_ptr = static_cast<double*>(quantities.mutable_unchecked<1>().mutable_data(0));
        
        for (size_t i = 0; i < size; ++i) {
            prices_ptr[i] = result.levels[i].first;
            quantities_ptr[i] = result.levels[i].second;
        }
        
        return py::make_tuple(prices, quantities);
    }, "Parse L2 string and return as NumPy arrays", py::arg("input"));
    
    m.def("parse_l2_batch", [](const std::vector<std::string>& inputs) {
        l2_parser::L2Parser parser;
        auto results = parser.parse_batch(inputs);
        
        py::list output_list;
        for (const auto& result : results) {
            if (result.success) {
                py::list prices, quantities;
                for (const auto& level : result.levels) {
                    prices.append(level.first);
                    quantities.append(level.second);
                }
                output_list.append(py::make_tuple(prices, quantities));
            } else {
                output_list.append(py::none());
            }
        }
        return output_list;
    }, "Parse multiple L2 strings and return as list of tuples", py::arg("inputs"));
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Crypto Alpha Mining Framework";
}
