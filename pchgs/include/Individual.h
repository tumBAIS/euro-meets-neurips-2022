/*MIT License

Original HGS-CVRP code: Copyright(c) 2020 Thibaut Vidal
Additional contributions: Copyright(c) 2022 ORTEC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <algorithm>
#include <set>
#include <string>
#include <sul/dynamic_bitset.hpp>
#include <vector>

#include "Params.h"

// Object to store all relevant information that may be needed to calculate some cost corresponding
// to a solution
struct CostSol {
    double penalizedCost;  // Penalized cost of the solution
    int nbRoutes;          // Number of routes
    int cost;              // Cost, including profits from customers
    int distance;          // Total distance
    int capacityExcess;    // Total excess load over all routes
    int waitTime;  // Total wait time (time to wait to meet earliest possible arrival) over all
                   // routes
    int timeWarp;  // Total time warp (going back in time to meet latest possible arrival) over all
                   // routes
    [[nodiscard]] bool operator==(const CostSol& rhs) const;
    [[nodiscard]] bool operator!=(const CostSol& rhs) const;

    // Constructor, initialize everything with 0
    CostSol()
        : penalizedCost(0.),
          nbRoutes(0),
          cost(0),
          distance(0),
          capacityExcess(0),
          waitTime(0),
          timeWarp(0) {}
};

// Object to represent one individual/solution of a population.
class Individual {
  private:
    std::vector<int> _successors;  // For each node, the successor in the solution (can be the depot
                                   // 0). Size is nbClients+1
    std::vector<int> _predecessors;  // For each node, the predecessor in the solution (can be the
                                     // depot 0). Size is nbClients+1

    // Customers present in the current solution
    sul::dynamic_bitset<> includedCustomers;

    void _fillRoutesRandomly();
    void _removeCustomer(int id);

  public:
    Params* params;     // Problem parameters
    CostSol myCostSol;  // Information on the cost of the solution
    std::vector<std::vector<int>>
        chromR;  // For each vehicle, the associated sequence of deliveries (complete solution).
                 // Size is nbVehicles. Routes are stored starting index maxVehicles-1, so the first
                 // indices will likely be empty
    std::multiset<std::pair<double, Individual*>>
        indivsPerProximity;  // The other individuals in the population (can not be the depot 0),
                             // ordered by increasing proximity (the set container follows a natural
                             // ordering based on the value of the first pair)
    bool isFeasible;         // Feasibility status of the individual
    double biasedFitness;    // Biased fitness of the solution

    [[nodiscard]] int getSuccessor(int id) const {
        assert(includedCustomers.test(id));
        return _successors[id];
    };
    [[nodiscard]] int getPredecessor(int id) const {
        assert(includedCustomers.test(id));
        return _predecessors[id];
    };

    void setPredecessor(int id, int value) {
        assert(includedCustomers.test(id));
        _predecessors[id] = value;
    }
    void setSuccessor(int id, int value) {
        assert(includedCustomers.test(id));
        _successors[id] = value;
    }

    /**
     * Marks and removes customer @param id from the solution.
     * @param id
     */
    void removeCustomer(int id, bool assert_validity);
    /**
     * Marks and adds customer @param id from the solution.
     * @param id
     */
    void addCustomer(int id, bool markOnly);

    bool isValid();
    bool isRequired(int i) const;
    void fillIncludedCustomers();
    // The number of *customers* which are part of this solution. This does not include the depot.
    [[nodiscard]] size_t getNumberOfServedCustomers() const;
    // The number of customers which are (deliberately) not part of this solution. Depot is always
    // required.
    [[nodiscard]] size_t getNumberOfUnservedCustomers() const;

    // Measuring cost of a solution from the information of chromR
    void evaluateCompleteCost();

    // Removing the individual indiv from the structure of proximity
    void removeProximity(Individual* indiv);

    // Distance measure with another individual, based on the number of arcs that differ between two
    // solutions
    double brokenPairsDistance(Individual* indiv2);

    // Returns the average distance of this individual with the nbClosest individuals
    double averageBrokenPairsDistanceClosest(int nbClosest);

    // Exports a solution in CVRPLib format (adds a final line with the computational time)
    void exportCVRPLibFormat(std::string fileName);

    // Prints a solution in CVRPLib format (adds a final line with the computational time)
    void printCVRPLibFormat();

    // Reads a solution in CVRPLib format, returns TRUE if the process worked, or FALSE if the file
    // does not exist or is not readable
    static bool readCVRPLibFormat(std::string fileName, std::vector<std::vector<int>>& readSolution,
                                  double& readCost);

    // Constructor: create a random individual
    Individual(Params* params, bool initialize = true);

    // Constructor: create an empty individual
    Individual();
    void clearRoutes();
    // Finds all customers which are required but currently not part of the solution.
    std::vector<int> findMissingCustomers() const;
    // Finds all customers which are not required.
    std::vector<int> getUnservedCustomers() const;

    void removeSuperfluousCustomers();

    [[nodiscard]] bool checkEvaluated();
};

std::vector<int> compute_included_customers(const Individual* indiv);

void remove_non_required_customers(Individual& indiv);
#endif
