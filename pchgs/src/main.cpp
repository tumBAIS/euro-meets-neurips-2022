#include <time.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "Genetic.h"
#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "Population.h"
#include "commandline.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "log.hpp"

Population build_population_with_removed_customers(Params &params, LocalSearch &localSearch,
                                                   const std::vector<int> &customersToRemove) {
    Population population(&params, &localSearch, false);

    Population tmpPop(&params, &localSearch);
    for (auto &pop: tmpPop.feasibleSubpopulation) {
        for (auto c: customersToRemove) {
            pop->removeCustomer(c, true);
        }
        population.addIndividual(pop, false);
    }
    for (auto &pop: tmpPop.infeasibleSubpopulation) {
        for (auto c: customersToRemove) {
            pop->removeCustomer(c, true);
        }
        population.addIndividual(pop, false);
    }

    return tmpPop;
}

nlohmann::json outputIndividualAsJSON(const Individual &individual, const Params &param) {
    nlohmann::json routes = std::vector<std::vector<int>>(0);

    for (const auto &route: individual.chromR) {
        // Skip empty routes
        if (route.empty()) continue;
        // Translate route
        std::vector<int> req_id_route;
        for (int customer_index: route) {
            req_id_route.push_back(param.cli[customer_index].requestIdx);
        }

        routes.push_back(nlohmann::json{{"cost",     0},
                                        {"prize",    0},
                                        {"requests", req_id_route}});
    }

    return {{"cost",   individual.myCostSol.penalizedCost},
            {"prize",  0},
            {"routes", routes}};
}

Individual parseIndividualFromSolution(Params &param, const nlohmann::json &data) {
    auto indiv = Individual(&param, false);

    auto route_idx = 0;
    for (const auto &route_data: data["routes"]) {
        if (route_idx >= indiv.chromR.size()) {
            throw std::runtime_error("Solution has more routes than individual can carry!");
        }
        for (int req_idx: route_data["requests"]) {
            int client_idx = param.getClientIDXByRequestIDX(req_idx);
            if (client_idx == 0) {
                throw std::runtime_error("Solution contains depot in routes. Aborting.");
            }
            if (param.isCertainlyUnprofitable(client_idx)) continue;

            indiv.chromR[route_idx].push_back(client_idx);
        }
        if (!indiv.chromR[route_idx].empty()) ++route_idx;
    }

    // Take care of profitable/certainly unprofitable customers

    // Mark any customers not in the solution as removed
    indiv.fillIncludedCustomers();

    // Insert any certainly profitable customers
    auto missing_customers = indiv.getUnservedCustomers();
    for (int missing_cust_id: missing_customers) {
        if (!param.isCertainlyProfitable(missing_cust_id)) continue;
        indiv.addCustomer(missing_cust_id, true);
        // Add the trivial depot tour for each missing customer
        indiv.chromR[route_idx++].push_back(missing_cust_id);
    }

    indiv.evaluateCompleteCost();

    return indiv;
}

Individual readIndividualFromFile(Params &param, std::string_view filename) {
    auto file_handle = std::ifstream(filename.data());
    if (!file_handle) {
        throw std::runtime_error(fmt::format("Could not open solution {}!", filename));
    }

    return parseIndividualFromSolution(param, nlohmann::json::parse(file_handle));
}

std::vector<Individual> readIndividuals(Params &param, LocalSearch *ls,
                                        const std::vector<std::string> &solution_files,
                                        bool reoptimize = true) {
    // Read individuals
    std::vector<Individual> individuals(0);
    individuals.reserve(solution_files.size());
    std::transform(
            solution_files.begin(), solution_files.end(), std::back_inserter(individuals),
            [&param](const std::string &filename) { return readIndividualFromFile(param, filename); });
    if (!param.config.quiet) {
        fmt::print("Read {} individuals with costs:", individuals.size());
        for (const auto &indiv: individuals) {
            fmt::print(" {}", indiv.myCostSol.penalizedCost);
        }
        fmt::print("\n");
    }

    if (reoptimize) {
        // Apply local search to each individual
        for (auto &indiv: individuals) {
            ls->run(&indiv, param.penaltyCapacity, param.penaltyTimeWarp);
            // Optimize customers
            optimizeCustomers(param, *ls, &indiv);
        }
    }

    return individuals;
}

Population createWarmStartedPopulation(Params &param, LocalSearch &ls,
                                       const std::vector<std::string> &solution_files,
                                       [[maybe_unused]] size_t target_size) {
    // Read and optimize solutions
    auto seeds = readIndividuals(param, &ls, solution_files, true);
    // Create empty population
    Population pop(&param, &ls, false);

    // Add seeds first
    for (const auto &seed: seeds) {
        pop.addIndividual(&seed, true);
    }
    // Then generate rest of population
    pop.generatePopulation();

    return pop;
}

void writeBKSAsJSON(const Individual &bks, const Params &param, std::ostream &outputStream) {
    auto formattedIndividual = outputIndividualAsJSON(bks, param);
    outputStream << formattedIndividual << std::flush;
}

// Main class of the algorithm. Used to read from the parameters from the command line,
// create the structures and initial population, and run the hybrid genetic search
int main(int argc, char *argv[]) {
    try {
        // Reading the arguments of the program
        CommandLine commandline(argc, argv);

        // Reading the data file and initializing some data structures
        Params params(commandline);

        logging::initialize(commandline.config.logFilePath, &params);

        // Creating the Split and Local Search structures
        LocalSearch localSearch(&params);

        Population population = createWarmStartedPopulation(
                params, localSearch, params.config.seedSolutions,
                2 * (params.config.minimumPopulationSize + params.config.generationSize));

        // Initial population
        if (!params.config.quiet) {
            std::cout << "----- INSTANCE LOADED WITH " << params.nbClients << " CLIENTS AND "
                      << params.nbVehicles << " VEHICLES" << std::endl;
            std::cout << "----- " << params.getNumberOfCertainlyUnprofitableCustomers()
                      << " ARE CERTAINLY UNPROFITABLE, "
                      << params.getNumberOfCertainlyProfitableCustomers()
                      << " ARE CERTAINLY PROFITABLE -----" << std::endl;
            std::cout << "----- BUILDING INITIAL POPULATION" << std::endl;
        }

        // Genetic algorithm
        if (!params.config.quiet) {
            std::cout << "----- STARTING GENETIC ALGORITHM" << std::endl;
        }
        Genetic solver(&params, &population, &localSearch);
        solver.run(commandline.config.nbIter, commandline.config.timeLimit);
        if (!params.config.quiet) {
            std::cout << "----- GENETIC ALGORITHM FINISHED, TIME SPENT: "
                      << params.getTimeElapsedSeconds() << std::endl;
        }

        // Export the best solution, if it exist
        if (population.getBestFound() != nullptr) {
            if (!commandline.config.outputJSONPath.empty()) {
                std::ofstream outputStream(commandline.config.outputJSONPath);
                writeBKSAsJSON(*population.getBestFound(), params, outputStream);
            } else {
                writeBKSAsJSON(*population.getBestFound(), params, std::cout);
            }
        }
    }

        // Catch exceptions
    catch (const std::string &e) {
        std::cout << "EXCEPTION | " << e << std::endl;
    } catch (const std::exception &e) {
        std::cout << "EXCEPTION | " << e.what() << std::endl;
    }

    // Return 0 if the program execution was successfull
    return 0;
}
