//
// Created by patrick on 9/15/22.
//

#ifndef HGS_LOG_HPP
#define HGS_LOG_HPP

#include <fstream>
#include <string_view>

class Params;

namespace logging {
    class NeuripsLogger {
      public:
        static NeuripsLogger& Instance();

      private:
        Params* _params;
        std::ofstream _output;

        NeuripsLogger(std::string_view output_filename, Params* params);

      public:
        void log(std::size_t iteration, std::size_t iterations_since_last_improvement,
                 std::string_view component, std::string_view message);

        static void initialize(std::string_view filename, Params* params);
    };

    void log(std::size_t iteration, std::size_t iterations_since_last_improvement,
             const std::string& component, const std::string& message);

    void initialize(std::string_view filename, Params* params);
}  // namespace logging

#endif  // HGS_LOG_HPP
