#include "Genetic.h"

#include <time.h>

#include <algorithm>
#include <iterator>
#include <unordered_set>

#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "Population.h"
#include "fmt/format.h"
#include "log.hpp"

namespace {
    template<class Iterator>
    void insert_unplanned_tasks(Individual *offspring, Iterator begin,
                                Iterator end, Params *params) {
        // Initialize some variables
        int newDistanceToInsert = INT_MAX;  // TODO:
        int newTimeToInsert = INT_MAX;
        int newDistanceFromInsert = INT_MAX;  // TODO:
        int newTimeFromInsert = INT_MAX;
        int distanceDelta = INT_MAX;  // TODO:

        // Loop over all unplannedTasks
        for (; begin != end; ++begin) {
            int c = *begin;
            // TODO Move to method caller
            if (!offspring->isRequired(c)) {
                continue;
            }
            assert(offspring->isRequired(c));

            // Get the earliest and laster possible arrival at the client
            int earliestArrival = params->cli[c].earliestArrival;
            int latestArrival = params->cli[c].latestArrival;

            int bestDistance = INT_MAX;
            std::pair<int, int> bestLocation;

            // Loop over all routes
            for (int r = 0; r < params->nbVehicles; r++) {
                // Go to the next route if this route is empty
                if (offspring->chromR[r].empty()) {
                    continue;
                }

                newDistanceFromInsert = params->getCost(c, offspring->chromR[r][0]);
                newTimeFromInsert = params->getTravelTime(c, offspring->chromR[r][0]);
                if (earliestArrival + newTimeFromInsert
                    < params->cli[offspring->chromR[r][0]].latestArrival) {
                    distanceDelta = params->getCost(0, c) + newDistanceToInsert
                                    - params->getCost(0, offspring->chromR[r][0]);
                    if (distanceDelta < bestDistance) {
                        bestDistance = distanceDelta;
                        bestLocation = {r, 0};
                    }
                }

                for (int i = 1; i < static_cast<int>(offspring->chromR[r].size()); i++) {
                    newDistanceToInsert = params->getCost(offspring->chromR[r][i - 1], c);
                    newDistanceFromInsert = params->getCost(c, offspring->chromR[r][i]);
                    newTimeFromInsert = params->getTravelTime(c, offspring->chromR[r][i]);
                    if (params->cli[offspring->chromR[r][i - 1]].earliestArrival
                        + newDistanceToInsert
                        < latestArrival
                        && earliestArrival + newTimeFromInsert
                           < params->cli[offspring->chromR[r][i]].latestArrival) {
                        distanceDelta = newDistanceToInsert + newDistanceFromInsert
                                        - params->getCost(offspring->chromR[r][i - 1],
                                                          offspring->chromR[r][i]);
                        if (distanceDelta < bestDistance) {
                            bestDistance = distanceDelta;
                            bestLocation = {r, i};
                        }
                    }
                }

                newDistanceToInsert = params->getCost(offspring->chromR[r].back(), c);
                newTimeToInsert = params->getTravelTime(offspring->chromR[r].back(), c);
                if (params->cli[offspring->chromR[r].back()].earliestArrival + newTimeToInsert
                    < latestArrival) {
                    distanceDelta = newDistanceToInsert + params->getCost(c, 0)
                                    - params->getCost(offspring->chromR[r].back(), 0);
                    if (distanceDelta < bestDistance) {
                        bestDistance = distanceDelta;
                        bestLocation = {r, static_cast<int>(offspring->chromR[r].size())};
                    }
                }
            }

            offspring->chromR[bestLocation.first].insert(
                    offspring->chromR[bestLocation.first].begin() + bestLocation.second, c);
        }
    }

    void remove_random_customers(Individual *indiv, unsigned int customer_count) {
        assert(indiv->isValid());
        assert(customer_count <= indiv->getNumberOfServedCustomers());

        std::vector<int> includedCustomers = compute_included_customers(indiv);

        std::shuffle(includedCustomers.begin(), includedCustomers.end(), indiv->params->rng);

        int removed_customers = 0;
        for (auto next_customer = includedCustomers.begin();
             next_customer != includedCustomers.end() && removed_customers < customer_count;
             ++next_customer) {
            if (indiv->params->isCertainlyProfitable(*next_customer)) continue;
            indiv->removeCustomer(*next_customer, true);
            ++removed_customers;
        }
    }

    void add_random_customers(Individual *indiv, unsigned int customer_count) {
        assert(indiv->getNumberOfUnservedCustomers() >= customer_count);
        auto unservedCustomers = indiv->getUnservedCustomers();

        unservedCustomers.erase(
                std::remove_if(unservedCustomers.begin(), unservedCustomers.end(),
                               [indiv](int customer_id) {
                                   return indiv->params->isCertainlyUnprofitable(customer_id);
                               }),
                unservedCustomers.end());

        customer_count
                = std::min(customer_count, static_cast<unsigned int>(unservedCustomers.size()));

        std::shuffle(unservedCustomers.begin(), unservedCustomers.end(), indiv->params->rng);

        for (unsigned int i = 0; i < customer_count; ++i) {
            indiv->addCustomer(unservedCustomers[i], true);
        }

        insert_unplanned_tasks(indiv, unservedCustomers.begin(),
                               std::next(unservedCustomers.begin(), customer_count), indiv->params);
    }
}  // namespace

void Genetic::run(int maxIterNonProd, int timeLimit) {
    // Do iterations of the Genetic Algorithm, until more then maxIterNonProd consecutive iterations
    // without improvement or a time limit (in seconds) is reached
    int nbIterNonProd = 1;
    for (int nbIter = 0; nbIterNonProd <= maxIterNonProd && !params->isTimeLimitExceeded();
         nbIter++) {
        /* SELECTION AND CROSSOVER */
        // First select parents using getNonIdenticalParentsBinaryTournament
        // Then use the selected parents to create new individuals using OX and SREX
        // Finally select the best new individual based on bestOfSREXAndOXCrossovers
        // Individual* offspring =
        // bestOfSREXAndOXCrossovers(population->getNonIdenticalParentsBinaryTournament());
        // Individual* offspring =
        // crossoverOX(population->getNonIdenticalParentsBinaryTournament());
        Individual *offspring = crossoverSREX(population->getNonIdenticalParentsBinaryTournament());

        // Restart if we fail to generate offspring, i.e., because the 0 solution is best
        if (!offspring) {
            population->restart(true);
            nbIterNonProd = 1;
            // Account for potentially missed penalty updates
            nbIter -= 1;
            continue;
        }

        // Mutate with a probability of 20%
        if (params->rng() % 100 < 20) {
            mutate(offspring);
        }

        /* LOCAL SEARCH */
        // Run the Local Search on the new individual
        localSearch->run(offspring, params->penaltyCapacity, params->penaltyTimeWarp);
        if (params->rng() % 100 == 0) {
            optimizeCustomers(*params, *localSearch, offspring);
        }
        // Check if the new individual is the best feasible individual of the population, based on
        // penalizedCost
        bool isNewBest = population->addIndividual(offspring, true);
        // In case of infeasibility, repair the individual with a certain probability
        if (!offspring->isFeasible
            && params->rng() % 100 < (unsigned int) params->config.repairProbability) {
            // Run the Local Search again, but with penalties for infeasibilities multiplied by 10
            localSearch->run(offspring, params->penaltyCapacity * 10.,
                             params->penaltyTimeWarp * 10.);
            // If the individual is feasible now, check if it is the best feasible individual of the
            // population, based on penalizedCost and add it to the population If the individual is
            // not feasible now, it is not added to the population
            if (offspring->isFeasible) {
                isNewBest = (population->addIndividual(offspring, false) || isNewBest);
            }
        }
        // Check if optimizing included customers boost the solution value
        if (isNewBest) {
            logging::log(nbIter, nbIterNonProd, "[GENETIC]",
                         fmt::format("found improving solution;{};{}",
                                     population->getBestFeasible()->myCostSol.penalizedCost,
                                     population->getBestFound()->myCostSol.penalizedCost));
            auto prev_sol_value = offspring->myCostSol.penalizedCost;
            optimizeCustomers(*params, *localSearch, offspring);
            if (offspring->myCostSol.penalizedCost < prev_sol_value) {
                bool t = population->addIndividual(offspring, false);
                assert(t || !offspring->isFeasible);
            }
        }

        /* TRACKING THE NUMBER OF ITERATIONS SINCE LAST SOLUTION IMPROVEMENT */
        if (isNewBest) {
            nbIterNonProd = 1;
        } else
            nbIterNonProd++;

        /* DIVERSIFICATION, PENALTY MANAGEMENT AND TRACES */
        // Update the penaltyTimeWarp and penaltyCapacity every 100 iterations
        if (nbIter % 100 == 0) {
            population->managePenalties();
            logging::log(nbIter, nbIterNonProd, "[GENETIC]",
                         fmt::format("update;penaltyCapacity;{}", params->penaltyCapacity));
            logging::log(nbIter, nbIterNonProd, "[GENETIC]",
                         fmt::format("update;penaltyTimeWarp;{}", params->penaltyTimeWarp));
        }
        // Print the state of the population every 500 iterations
        if (nbIter % 500 == 0 && !params->config.quiet) {
            population->printState(nbIter, nbIterNonProd);
        }

        /* OTHER PARAMETER CHANGES*/
        // Increase the nbGranular by growNbGranularSize (and set the correlated vertices again)
        // every certain number of iterations, if growNbGranularSize is greater than 0
        if (nbIter > 0 && params->config.growNbGranularSize != 0
            && ((params->config.growNbGranularAfterIterations > 0
                 && nbIter % params->config.growNbGranularAfterIterations == 0)
                || (params->config.growNbGranularAfterNonImprovementIterations > 0
                    && nbIterNonProd % params->config.growNbGranularAfterNonImprovementIterations
                       == 0))) {
            // Note: changing nbGranular also changes how often the order is reshuffled
            params->config.nbGranular += params->config.growNbGranularSize;
            logging::log(nbIter, nbIterNonProd, "[GENETIC]",
                         fmt::format("update;nbGranular;{}", params->config.nbGranular));
        }

        // Increase the minimumPopulationSize by growPopulationSize every certain number of
        // iterations, if growPopulationSize is greater than 0
        if (nbIter > 0 && params->config.growPopulationSize != 0
            && ((params->config.growPopulationAfterIterations > 0
                 && nbIter % params->config.growPopulationAfterIterations == 0)
                || (params->config.growPopulationAfterNonImprovementIterations > 0
                    && nbIterNonProd % params->config.growPopulationAfterNonImprovementIterations
                       == 0))) {
            // This will automatically adjust after some iterations
            params->config.minimumPopulationSize += params->config.growPopulationSize;
            logging::log(nbIter, nbIterNonProd, "[GENETIC]",
                         fmt::format("update;minimumPopulationSize;{}",
                                     params->config.minimumPopulationSize));
        }

        /* FOR TESTS INVOLVING SUCCESSIVE RUNS UNTIL A TIME LIMIT: WE RESET THE ALGORITHM/POPULATION
         * EACH TIME maxIterNonProd IS ATTAINED*/
        if (timeLimit != INT_MAX && nbIterNonProd == maxIterNonProd
            && params->config.doRepeatUntilTimeLimit) {
            population->restart(true);
            logging::log(nbIter, nbIterNonProd, "[GENETIC]", "restart");
            nbIterNonProd = 1;
        }
    }
}

bool Genetic::mutate(Individual *offspring) {
    // Account for the depot
    if (params->rng() % 2 == 0 && offspring->getNumberOfServedCustomers() > 0) {
        remove_random_customers(offspring,
                                std::max(1u, static_cast<unsigned int>(
                                        0.15 * offspring->getNumberOfServedCustomers())));
        return true;
    } else if (offspring->getNumberOfUnservedCustomers() > 0) {
        add_random_customers(offspring,
                             std::max(1u, static_cast<unsigned int>(
                                     0.15 * offspring->getNumberOfUnservedCustomers())));
        return true;
    }
    return false;
}

std::pair<int, int> optimizeCustomers(Params &params, LocalSearch &ls, Individual *offspring) {
    std::vector<unsigned int> subsequence_length;

    auto longest_route
            = std::max_element(offspring->chromR.begin(), offspring->chromR.end(),
                               [](const auto &lhs, const auto &rhs) { return lhs.size() < rhs.size(); })
                    ->size();
    for (unsigned int i = 1; i <= longest_route; ++i) {
        subsequence_length.push_back(i);
    }

    auto [unprofitableCustomers, profitableCustomers] = ls.optimizeIncludedCustomers(
            *offspring, params.config.minOptimalCustomerPricingPerturbationFactor,
            params.config.maxOptimalCustomerPricingPerturbationFactor, subsequence_length);

    if (!unprofitableCustomers.empty()) {
        for (auto cust_id: unprofitableCustomers) {
            offspring->removeCustomer(cust_id, true);
        }
        offspring->evaluateCompleteCost();
        assert(offspring->isValid());
    }

    if (!profitableCustomers.empty()) {
        for (auto cust_id: profitableCustomers) {
            offspring->addCustomer(cust_id, true);
        }
        insert_unplanned_tasks(offspring, profitableCustomers.begin(), profitableCustomers.end(),
                               &params);
        offspring->evaluateCompleteCost();
        assert(offspring->isValid());
    }

    // Re-run the local search if the solution changed.
    if (!profitableCustomers.empty() || !unprofitableCustomers.empty()) {
        ls.run(offspring, params.penaltyCapacity, params.penaltyTimeWarp);
    }

    return {unprofitableCustomers.size(), profitableCustomers.size()};
}

Individual *Genetic::crossoverSREX(std::pair<const Individual *, const Individual *> parents) {
    // Get the number of routes of both parents
    int nOfRoutesA = parents.first->myCostSol.nbRoutes;
    int nOfRoutesB = parents.second->myCostSol.nbRoutes;

    if (nOfRoutesA == 0 && nOfRoutesB == 0) {
        return nullptr;
    } else if (nOfRoutesA == 0 || nOfRoutesB == 0) {
        const Individual *betterParent
                = (parents.first->myCostSol.penalizedCost < parents.second->myCostSol.penalizedCost)
                  ? parents.first
                  : parents.second;
        *candidateOffsprings[0] = *betterParent;
        return candidateOffsprings[0];
    }

    for (int i = 0; i < 2; ++i) {
        *candidateOffsprings[i] = *parents.first;
        candidateOffsprings[i]->clearRoutes();
    }

    // Picking the start index of routes to replace of parent A
    // We like to replace routes with a large overlap of tasks, so we choose adjacent routes (they
    // are sorted on polar angle)
    int startA = params->rng() % nOfRoutesA;
    int nOfMovedRoutes = std::max(
            1u, params->rng() % (std::min(nOfRoutesA, nOfRoutesB)));  // Prevent not moving any routes
    assert(nOfMovedRoutes >= 1);
    // TODO Why startA? Routes of B will be different and thus may not correlate polar angle wise
    int startB = startA < nOfRoutesB ? startA : 0;

    std::unordered_set<int> clientsInSelectedA;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        // Insert the first
        clientsInSelectedA.insert(parents.first->chromR[(startA + r) % nOfRoutesA].begin(),
                                  parents.first->chromR[(startA + r) % nOfRoutesA].end());
    }

    std::unordered_set<int> clientsInSelectedB;
    for (int r = 0; r < nOfMovedRoutes; r++) {
        clientsInSelectedB.insert(parents.second->chromR[(startB + r) % nOfRoutesB].begin(),
                                  parents.second->chromR[(startB + r) % nOfRoutesB].end());
    }

    bool improved = true;
    while (improved) {
        assert(startA + nOfMovedRoutes - 1 < parents.first->chromR.size());
        // Difference for moving 'left' in parent A
        const int differenceALeft
                = static_cast<int>(std::count_if(
                        parents.first->chromR[(startA - 1 + nOfRoutesA) % nOfRoutesA].begin(),
                        parents.first->chromR[(startA - 1 + nOfRoutesA) % nOfRoutesA].end(),
                        [&clientsInSelectedB](int c) {
                            return clientsInSelectedB.find(c) == clientsInSelectedB.end();
                        }))
                  - static_cast<int>(std::count_if(
                        parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA].begin(),
                        parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA].end(),
                        [&clientsInSelectedB](int c) {
                            return clientsInSelectedB.find(c) == clientsInSelectedB.end();
                        }));

        // Difference for moving 'right' in parent A
        const int differenceARight
                = static_cast<int>(std::count_if(
                        parents.first->chromR[(startA + nOfMovedRoutes) % nOfRoutesA].begin(),
                        parents.first->chromR[(startA + nOfMovedRoutes) % nOfRoutesA].end(),
                        [&clientsInSelectedB](int c) {
                            return clientsInSelectedB.find(c) == clientsInSelectedB.end();
                        }))
                  - static_cast<int>(
                          std::count_if(parents.first->chromR[startA].begin(),
                                        parents.first->chromR[startA].end(), [&clientsInSelectedB](int c) {
                                      return clientsInSelectedB.find(c) == clientsInSelectedB.end();
                                  }));

        // Difference for moving 'left' in parent B
        const int differenceBLeft
                = static_cast<int>(std::count_if(
                        parents.second->chromR[(startB - 1 + nOfMovedRoutes) % nOfRoutesB].begin(),
                        parents.second->chromR[(startB - 1 + nOfMovedRoutes) % nOfRoutesB].end(),
                        [&clientsInSelectedA](int c) {
                            return clientsInSelectedA.find(c) != clientsInSelectedA.end();
                        }))
                  - static_cast<int>(std::count_if(
                        parents.second->chromR[(startB - 1 + nOfRoutesB) % nOfRoutesB].begin(),
                        parents.second->chromR[(startB - 1 + nOfRoutesB) % nOfRoutesB].end(),
                        [&clientsInSelectedA](int c) {
                            return clientsInSelectedA.find(c) != clientsInSelectedA.end();
                        }));

        // Difference for moving 'right' in parent B
        const int differenceBRight
                = static_cast<int>(std::count_if(
                        parents.second->chromR[startB].begin(), parents.second->chromR[startB].end(),
                        [&clientsInSelectedA](int c) {
                            return clientsInSelectedA.find(c) != clientsInSelectedA.end();
                        }))
                  - static_cast<int>(std::count_if(
                        parents.second->chromR[(startB + nOfMovedRoutes) % nOfRoutesB].begin(),
                        parents.second->chromR[(startB + nOfMovedRoutes) % nOfRoutesB].end(),
                        [&clientsInSelectedA](int c) {
                            return clientsInSelectedA.find(c) != clientsInSelectedA.end();
                        }));

        const int bestDifference
                = std::min({differenceALeft, differenceARight, differenceBLeft, differenceBRight});

        if (bestDifference < 0) {
            if (bestDifference == differenceALeft) {
                for (int c: parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA]) {
                    clientsInSelectedA.erase(clientsInSelectedA.find(c));
                }
                startA = (startA - 1 + nOfRoutesA) % nOfRoutesA;
                for (int c: parents.first->chromR[startA]) {
                    clientsInSelectedA.insert(c);
                }
            } else if (bestDifference == differenceARight) {
                for (int c: parents.first->chromR[startA]) {
                    clientsInSelectedA.erase(clientsInSelectedA.find(c));
                }
                startA = (startA + 1) % nOfRoutesA;
                for (int c: parents.first->chromR[(startA + nOfMovedRoutes - 1) % nOfRoutesA]) {
                    clientsInSelectedA.insert(c);
                }
            } else if (bestDifference == differenceBLeft) {
                for (int c: parents.second->chromR[(startB + nOfMovedRoutes - 1) % nOfRoutesB]) {
                    clientsInSelectedB.erase(clientsInSelectedB.find(c));
                }
                startB = (startB - 1 + nOfRoutesB) % nOfRoutesB;
                for (int c: parents.second->chromR[startB]) {
                    clientsInSelectedB.insert(c);
                }
            } else if (bestDifference == differenceBRight) {
                for (int c: parents.second->chromR[startB]) {
                    clientsInSelectedB.erase(clientsInSelectedB.find(c));
                }
                startB = (startB + 1) % nOfRoutesB;
                for (int c: parents.second->chromR[(startB + nOfMovedRoutes - 1) % nOfRoutesB]) {
                    clientsInSelectedB.insert(c);
                }
            }
        } else {
            improved = false;
        }
    }

    // Identify differences between route sets
    std::unordered_set<int> clientsInSelectedANotB;  // Customers in routes A but not in routes B
    std::copy_if(clientsInSelectedA.begin(), clientsInSelectedA.end(),
                 std::inserter(clientsInSelectedANotB, clientsInSelectedANotB.end()),
                 [&clientsInSelectedB](int c) { return !clientsInSelectedB.contains(c); });

    std::unordered_set<int> clientsInSelectedBNotA;  // Customers in routes B but not in A
    std::copy_if(clientsInSelectedB.begin(), clientsInSelectedB.end(),
                 std::inserter(clientsInSelectedBNotA, clientsInSelectedBNotA.end()),
                 [&clientsInSelectedA](int c) { return !clientsInSelectedA.contains(c); });

    // Replace selected routes from parent A with routes from parent B
    for (int r = 0; r < nOfMovedRoutes; r++) {
        int indexA = (startA + r) % nOfRoutesA;
        int indexB = (startB + r) % nOfRoutesB;
        candidateOffsprings[0]->chromR[indexA].clear();
        candidateOffsprings[1]->chromR[indexA].clear();

        for (int c: parents.second->chromR[indexB]) {
            // Replaces route from parent A with route from parent B (Step 2 Type I)
            candidateOffsprings[0]->chromR[indexA].push_back(c);
            // Step 1 Type II
            // Replace route from parent A with route from parent B minus customers that are in B
            // but not in A
            if (!clientsInSelectedBNotA.contains(c)) {
                candidateOffsprings[1]->chromR[indexA].push_back(c);
            }
        }
    }

    // Move routes from parent A that are kept
    for (int r = nOfMovedRoutes; r < nOfRoutesA; r++) {
        int indexA = (startA + r) % nOfRoutesA;
        candidateOffsprings[0]->chromR[indexA].clear();
        candidateOffsprings[1]->chromR[indexA].clear();

        for (int c: parents.first->chromR[indexA]) {
            // Copy route A minus customers in B but not in A
            if (!clientsInSelectedBNotA.contains(c)) {
                candidateOffsprings[0]->chromR[indexA].push_back(c);
            }
            // Copy route from A
            candidateOffsprings[1]->chromR[indexA].push_back(c);
        }
    }

    // Step 3: Insert unplanned clients (those that were in the removed routes of A but not the
    // inserted routes of B)
    for (auto i = 0; i < 2; ++i) {
        // Customers required in Parent A should be required in the offspring.
        // Problem: Some customers may be superfluous, some may be missing
        // First, remove all superfluous customers
        candidateOffsprings[i]->removeSuperfluousCustomers();
        // Then determine which customers are missing
        auto missing_customers = candidateOffsprings[i]->findMissingCustomers();
        insert_unplanned_tasks(candidateOffsprings[i], missing_customers.begin(),
                               missing_customers.end(), params);
        assert(candidateOffsprings[i]->isValid());
        candidateOffsprings[i]->evaluateCompleteCost();
    }

    auto bestOffspring
            = std::min_element(candidateOffsprings.begin(), candidateOffsprings.begin() + 2,
                               [](const Individual *lhs, const Individual *rhs) {
                                   return lhs->myCostSol.penalizedCost < rhs->myCostSol.penalizedCost;
                               });

    return *bestOffspring;
}

Genetic::Genetic(Params *params, Population *population, LocalSearch *localSearch)
        : params(params), population(population), localSearch(localSearch) {
    // After initializing the parameters of the Genetic object, also generate new individuals in the
    // array candidateOffsprings
    std::generate(candidateOffsprings.begin(), candidateOffsprings.end(),
                  [&] { return new Individual(params); });
}

Genetic::~Genetic(void) {
    // Destruct the Genetic object by deleting all the individuals of the candidateOffsprings
    for (Individual *candidateOffspring: candidateOffsprings) {
        delete candidateOffspring;
    }
}
