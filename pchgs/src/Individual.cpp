#include "Individual.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "Params.h"

std::vector<int> compute_included_customers(const Individual *indiv) {
    std::vector<int> includedCustomers;
    includedCustomers.reserve(indiv->getNumberOfServedCustomers());
    for (unsigned int i = 1; i <= indiv->params->nbClients; ++i) {
        if (indiv->isRequired(i)) {
            includedCustomers.push_back(i);
        }
    }
    // Account for depot
    assert(includedCustomers.size() == indiv->getNumberOfServedCustomers());
    return includedCustomers;
}

void Individual::removeSuperfluousCustomers() {
    // Iterate over solution any remove any customers not included
    std::vector<int> customersToRemove;
    for (const auto &route: chromR) {
        std::copy_if(route.begin(), route.end(), std::back_inserter(customersToRemove),
                     [this](auto client_id) { return !isRequired(client_id); });
    }

    for (auto client_id: customersToRemove) {
        // call private method to not modify includedCustomers.
        _removeCustomer(client_id);
    }
}

std::vector<int> Individual::getUnservedCustomers() const {
    std::vector<int> not_included_customer_vec;
    not_included_customer_vec.reserve(includedCustomers.size());
    auto notIncludedCustomers = ~includedCustomers;
    for (auto nextNotIncludedCustomer = notIncludedCustomers.find_first();
         nextNotIncludedCustomer != sul::dynamic_bitset<>::npos;
         nextNotIncludedCustomer = notIncludedCustomers.find_next(nextNotIncludedCustomer)) {
        not_included_customer_vec.push_back(static_cast<int>(nextNotIncludedCustomer));
    }

    assert(not_included_customer_vec.size() == params->nbClients - (includedCustomers.count() - 1)
           && not_included_customer_vec.size() == notIncludedCustomers.count());

    return not_included_customer_vec;
}

std::vector<int> Individual::findMissingCustomers() const {
    sul::dynamic_bitset<> seenCustomers(params->nbClients + 1, 1);
    for (const auto &route: chromR) {
        for (auto c: route) {
            seenCustomers.set(c);
        }
    }

    assert(seenCustomers.is_subset_of(includedCustomers));
    auto missing_customer_set = seenCustomers ^ includedCustomers;

    std::vector<int> missing_customers;
    missing_customers.reserve(missing_customer_set.count());

    for (auto i = missing_customer_set.find_first(); i != sul::dynamic_bitset<>::npos;
         i = missing_customer_set.find_next(i)) {
        missing_customers.push_back(static_cast<int>(i));
    }
    return missing_customers;
}

void Individual::evaluateCompleteCost() {
    // Create an object to store all information regarding solution costs
    myCostSol = CostSol();
    // Loop over all routes that are not empty
    for (int r = 0; r < params->nbVehicles; r++) {
        if (!chromR[r].empty()) {
            int latestReleaseTime = params->cli[chromR[r][0]].releaseTime;
            for (int i = 1; i < static_cast<int>(chromR[r].size()); i++) {
                latestReleaseTime
                        = std::max(latestReleaseTime, params->cli[chromR[r][i]].releaseTime);
            }
            // Get the distance, load, serviceDuration and time associated with the vehicle
            // traveling from the depot to the first client Assume depot has service time 0 and
            // earliestArrival 0
            int distance = params->getCost(0, chromR[r][0]);
            int cost = params->getDiscountedCost(0, chromR[r][0]);
            int load = params->cli[chromR[r][0]].demand;
            // Running time excludes service of current node. This is the time that runs with the
            // vehicle traveling We start the route at the latest release time (or later but then we
            // can just wait and there is no penalty for waiting)
            int time = latestReleaseTime + params->getTravelTime(0, chromR[r][0]);
            int waitTime = 0;
            int timeWarp = 0;
            // Add possible waiting time
            if (time < params->cli[chromR[r][0]].earliestArrival) {
                // Don't add wait time since we can start route later
                // (doesn't really matter since there is no penalty anyway)
                // waitTime += params->cli[chromR[r][0]].earliestArrival - time;
                time = params->cli[chromR[r][0]].earliestArrival;
            }
                // Add possible time warp
            else if (time > params->cli[chromR[r][0]].latestArrival) {
                timeWarp += time - params->cli[chromR[r][0]].latestArrival;
                time = params->cli[chromR[r][0]].latestArrival;
            }
            setPredecessor(chromR[r][0], 0);

            // Loop over all clients for this vehicle
            for (int i = 1; i < static_cast<int>(chromR[r].size()); i++) {
                auto current_client_id = chromR[r][i], prev_client_id = chromR[r][i - 1];
                const Client &current_client = params->cli[current_client_id];
                // Sum the distance, load, serviceDuration and time associated with the vehicle
                // traveling from the depot to the next client
                distance += params->getCost(prev_client_id, current_client_id);
                cost += params->getDiscountedCost(prev_client_id, current_client_id);
                load += current_client.demand;
                time = time + params->cli[chromR[r][i - 1]].serviceDuration
                       + params->getTravelTime(prev_client_id, current_client_id);

                // Add possible waiting time
                if (time < current_client.earliestArrival) {
                    waitTime += current_client.earliestArrival - time;
                    time = current_client.earliestArrival;
                }
                    // Add possible time warp
                else if (time > current_client.latestArrival) {
                    timeWarp += time - current_client.latestArrival;
                    time = current_client.latestArrival;
                }

                // Update predecessors and successors
                setPredecessor(current_client_id, prev_client_id);
                setSuccessor(prev_client_id, current_client_id);
            }

            // For the last client, the successors is the depot. Also update the distance and time
            setSuccessor(chromR[r][chromR[r].size() - 1], 0);
            distance += params->getCost(chromR[r][chromR[r].size() - 1], 0);
            cost += params->getDiscountedCost(chromR[r][chromR[r].size() - 1], 0);
            time = time + params->cli[chromR[r][chromR[r].size() - 1]].serviceDuration
                   + params->getTravelTime(chromR[r][chromR[r].size() - 1], 0);

            // For the depot, we only need to check the end of the time window (add possible time
            // warp)
            if (time > params->cli[0].latestArrival) {
                timeWarp += time - params->cli[0].latestArrival;
                time = params->cli[0].latestArrival;
            }
            // Update variables that track stats on the whole solution (all vehicles combined)
            myCostSol.distance += distance;
            myCostSol.cost += cost;
            myCostSol.waitTime += waitTime;
            myCostSol.timeWarp += timeWarp;
            myCostSol.nbRoutes++;
            if (load > params->vehicleCapacity) {
                myCostSol.capacityExcess += load - params->vehicleCapacity;
            }
        }
    }

    // When all vehicles are dealt with, calculated total penalized cost and check if the solution
    // is feasible. (Wait time does not affect feasibility)
    myCostSol.penalizedCost = myCostSol.cost + myCostSol.capacityExcess * params->penaltyCapacity
                              + myCostSol.timeWarp * params->penaltyTimeWarp
                              + myCostSol.waitTime * params->penaltyWaitTime;
    isFeasible = (myCostSol.capacityExcess < MY_EPSILON && myCostSol.timeWarp < MY_EPSILON);
}

void Individual::removeProximity(Individual *indiv) {
    // Get the first individual in indivsPerProximity
    auto it = indivsPerProximity.begin();
    // Loop over all individuals in indivsPerProximity until indiv is found
    while (it->second != indiv) {
        ++it;
    }
    // Remove indiv from indivsPerProximity
    indivsPerProximity.erase(it);
}

double Individual::brokenPairsDistance(Individual *indiv2) {
    // Initialize the difference to zero. Then loop over all clients of this individual
    int differences = 0;
    for (int j = 1; j <= params->nbClients; j++) {
        // Handle case where either solution does not require the customer
        if (!isRequired(j) || !indiv2->isRequired(j)) {
            // Differs only if one requires and the other one does not
            differences += (isRequired(j) != indiv2->isRequired(j));
            continue;
        }
        // Increase the difference if the successor of j in this individual is not directly linked
        // to j in indiv2
        if (getSuccessor(j) != indiv2->getSuccessor(j)
            && getSuccessor(j) != indiv2->getPredecessor(j)) {
            differences++;
        }
        // Last loop covers all but the first arc. Increase the difference if the predecessor of j
        // in this individual is not directly linked to j in indiv2
        if (getPredecessor(j) == 0 && indiv2->getPredecessor(j) != 0
            && indiv2->getSuccessor(j) != 0) {
            differences++;
        }
    }
    return static_cast<double>(differences) / params->nbClients;
}

double Individual::averageBrokenPairsDistanceClosest(int nbClosest) {
    double result = 0;
    int maxSize = std::min(nbClosest, static_cast<int>(indivsPerProximity.size()));
    auto it = indivsPerProximity.begin();
    for (int i = 0; i < maxSize; i++) {
        result += it->first;
        ++it;
    }
    return result / maxSize;
}

// TODO Refactor to take stream
void Individual::exportCVRPLibFormat(std::string fileName) {
    if (!params->config.quiet) {
        std::cout << "----- WRITING SOLUTION WITH VALUE " << myCostSol.penalizedCost
                  << " IN : " << fileName << std::endl;
    }
    std::ofstream myfile(fileName);
    if (myfile.is_open()) {
        for (int k = 0; k < params->nbVehicles; k++) {
            if (!chromR[k].empty()) {
                myfile << "Route #" << k + 1 << ":";  // Route IDs start at 1 in the file format
                for (int i: chromR[k]) {
                    myfile << " " << i;
                }
                myfile << std::endl;
            }
        }
        myfile << "Cost " << myCostSol.penalizedCost << std::endl;
        myfile << "Time " << params->getTimeElapsedSeconds() << std::endl;
    } else
        std::cout << "----- IMPOSSIBLE TO OPEN: " << fileName << std::endl;
}

// TODO Refactor to use export function
void Individual::printCVRPLibFormat() {
    std::cout << "----- PRINTING SOLUTION WITH VALUE " << myCostSol.distance << " ("
              << myCostSol.penalizedCost << ")" << std::endl;
    std::cout << "Removed customers: ";
    const char *delim = " ";
    for (auto i = 1; i <= params->nbClients; ++i) {
        if (isRequired(i)) continue;
        std::cout << delim << i << " (" << params->cli[i].profit << ")";
        delim = ", ";
    }
    std::cout << std::endl;
    for (int k = 0; k < params->nbVehicles; k++) {
        if (!chromR[k].empty()) {
            std::cout << "Route #" << k + 1 << ":";  // Route IDs start at 1 in the file format
            for (int i: chromR[k]) {
                std::cout << " " << i;
            }
            std::cout << std::endl;
        }
    }
    std::cout << "Cost " << myCostSol.penalizedCost << std::endl;
    std::cout << "Time " << params->getTimeElapsedSeconds() << std::endl;
    fflush(stdout);
}

bool Individual::readCVRPLibFormat(std::string fileName,
                                   std::vector<std::vector<int>> &readSolution, double &readCost) {
    readSolution.clear();
    std::ifstream inputFile(fileName);
    if (inputFile.is_open()) {
        std::string inputString;
        inputFile >> inputString;
        // Loops as long as the first line keyword is "Route"
        for (int r = 0; inputString == "Route"; r++) {
            readSolution.push_back(std::vector<int>());
            inputFile >> inputString;
            getline(inputFile, inputString);
            std::stringstream ss(inputString);
            int inputCustomer;
            // Loops as long as there is an integer to read
            while (ss >> inputCustomer) {
                readSolution[r].push_back(inputCustomer);
            }
            inputFile >> inputString;
        }
        if (inputString == "Cost") {
            inputFile >> readCost;
            return true;
        } else
            std::cout << "----- UNEXPECTED WORD IN SOLUTION FORMAT: " << inputString << std::endl;
    } else
        std::cout << "----- IMPOSSIBLE TO OPEN: " << fileName << std::endl;
    return false;
}

Individual::Individual(Params *params, bool initialize)
        : includedCustomers(params->nbClients + 1),
          params(params),
          isFeasible(false),
          biasedFitness(0) {
    _successors = std::vector<int>(params->nbClients + 1);
    _predecessors = std::vector<int>(params->nbClients + 1);
    chromR = std::vector<std::vector<int>>(params->nbVehicles);
    // Mark all potentially profitable customers + the depot as required.
    includedCustomers |= ~(params->getCertainlyUnprofitableCustomerMask());
#ifndef NDEBUG
    for (int c = 0; c <= params->nbClients; ++c) {
        assert(isRequired(c) == !params->isCertainlyUnprofitable(c));
    }
#endif

    assert(isRequired(0));
    if (initialize) {
        _fillRoutesRandomly();
        evaluateCompleteCost();
    }
}

Individual::Individual() : params(nullptr), isFeasible(false), biasedFitness(0) {
    myCostSol.penalizedCost = 1.e30;
}

void Individual::fillIncludedCustomers() {
    includedCustomers.reset();
    // Mark depot
    includedCustomers.set(0);
    for (const auto &route: chromR) {
        for (int customerID: route) {
            assert(!includedCustomers.test(customerID));
            includedCustomers.set(customerID);
        }
    }
}

size_t Individual::getNumberOfServedCustomers() const {
    // Account for depot.
    return includedCustomers.count() - 1;
}

bool Individual::isValid() {
    assert(includedCustomers.test(0));
    // Ensure that each customer in includedCustomers is present in the solution
    sul::dynamic_bitset<> seenCustomers(params->nbClients + 1, 1);

    for (const auto &route: chromR) {
        for (int customerID: route) {
            if (seenCustomers.test(customerID)) {
                return false;
            }
            seenCustomers.set(customerID);
        }
    }

    if (seenCustomers != includedCustomers) {
        return false;
    }

    for (int c = 1; c < params->nbClients; ++c) {
        if (params->isCertainlyUnprofitable(c) && isRequired(c)) {
            return false;
        }
        if (params->isCertainlyProfitable((c)) && !isRequired(c)) {
            return false;
        }
    }

    return true;
}

void Individual::removeCustomer(int id, bool assert_validity) {
    // Do not remove the depot
    assert(id > 0);
    assert(isRequired(id));
    assert(!assert_validity || isValid());
    _removeCustomer(id);

    // Mark as removed
    includedCustomers.reset(id);
    assert(!assert_validity || isValid());
}

void Individual::_removeCustomer(int id) {
    for (auto &route: chromR) {
        route.erase(std::remove(route.begin(), route.end(), id), route.end());
    }

    if (isRequired(id)) {
        // Update predecessor/successor of previous nodes
        setSuccessor(getPredecessor(id), getSuccessor(id));
        setPredecessor(getSuccessor(id), getPredecessor(id));

        // Unset for removed node
        setPredecessor(id, 0);
        setSuccessor(id, 0);
    } else {
        assert(_successors[id] == 0);
        assert(_predecessors[id] == 0);
    }
}

bool Individual::isRequired(int i) const { return includedCustomers.test(i); }

void Individual::clearRoutes() {
    std::for_each(chromR.begin(), chromR.end(), [](auto &r) { r.clear(); });
}

void Individual::_fillRoutesRandomly() {
    // depot is always included but never part of chromR
    std::vector<int> customers = findMissingCustomers();
    std::shuffle(customers.begin(), customers.end(), params->rng);
    while (!customers.empty()) {
        // Choose random route
        unsigned int route_id = params->rng() % chromR.size();
        chromR[route_id].push_back(customers.back());
        customers.pop_back();
    }
    assert(std::all_of(chromR.begin(), chromR.end(), [this](const auto &route) {
        return std::none_of(route.begin(), route.end(),
                            [this](int c) { return params->isCertainlyUnprofitable(c); });
    }));
}

void Individual::addCustomer(int id, bool markOnly) {
    assert(!isRequired(id));
    assert(!params->isCertainlyUnprofitable(id));
    includedCustomers.set(id);
    if (markOnly) {
        return;
    }
    throw std::runtime_error("Not implemented");
}

size_t Individual::getNumberOfUnservedCustomers() const {
    // params->nbClients does not include the depot. Hence, this is safe.
    assert(getUnservedCustomers().size() == (params->nbClients - getNumberOfServedCustomers()));
    return params->nbClients - getNumberOfServedCustomers();
}

bool Individual::checkEvaluated() {
#ifdef NDEBUG
    return true;
#endif
    auto prevCostSol = myCostSol;
    evaluateCompleteCost();
    return prevCostSol == myCostSol;
}

bool CostSol::operator==(const CostSol &rhs) const {
    return penalizedCost == rhs.penalizedCost && nbRoutes == rhs.nbRoutes && cost == rhs.cost
           && distance == rhs.distance && capacityExcess == rhs.capacityExcess
           && waitTime == rhs.waitTime && timeWarp == rhs.timeWarp;
}

bool CostSol::operator!=(const CostSol &rhs) const { return !(rhs == *this); }
