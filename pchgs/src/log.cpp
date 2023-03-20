#include "log.hpp"

#include <memory>

#include "Params.h"

static std::unique_ptr<logging::NeuripsLogger> _instance;

namespace logging {
    NeuripsLogger &NeuripsLogger::Instance() {
        // Fine if it does not exist (for now)
        return *_instance;
    }

    NeuripsLogger::NeuripsLogger(std::string_view output_filename, Params *params)
            : _params(params), _output(output_filename.data()) {}

    void NeuripsLogger::log(std::size_t iteration, std::size_t iterations_since_last_improvement,
                            std::string_view component, std::string_view message) {
        _output << _params->getTimeElapsedSeconds() << ';' << iteration << ';'
                << iterations_since_last_improvement << ';' << component << ';' << message << '\n';
    }

    void NeuripsLogger::initialize(std::string_view filename, Params *params) {
        if (_instance) {
            throw std::logic_error("Already initialized logger!");
        }
        _instance = std::unique_ptr<NeuripsLogger>(new NeuripsLogger(filename, params));
    }


#ifndef SUBMISSION_MODE
    void log(std::size_t iteration, std::size_t iterations_since_last_improvement,
             const std::string &component, const std::string &message) {
        NeuripsLogger::Instance().log(iteration, iterations_since_last_improvement, component,
                                      message);
    }
#else

    void log(std::size_t, std::size_t,
             const std::string &, const std::string &) {}

#endif

#ifndef SUBMISSION_MODE
    void initialize(std::string_view filename, Params *params) {
        NeuripsLogger::initialize(filename, params);
    }
#else

    void initialize(std::string_view, Params *) {}

#endif
}  // namespace logging
