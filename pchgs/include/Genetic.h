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

#ifndef GENETIC_H
#define GENETIC_H

#include <array>
#include <unordered_set>

#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "Population.h"

std::pair<int, int> optimizeCustomers(Params& param, LocalSearch& ls, Individual* offspring);

// Class to run the genetic algorithm, which incorporates functionality of population management,
// doing crossovers and updating parameters.
class Genetic {
  public:
    // Running the genetic algorithm until maxIterNonProd consecutive iterations without improvement
    // or a time limit (in seconds) is reached
    void run(int maxIterNonProd, int timeLimit);

    // Constructor
    Genetic(Params* params, Population* population, LocalSearch* localSearch);

    // Destructor
    ~Genetic();

  private:
    // The number of new potential offspring created from one individual
    static const int numberOfCandidateOffsprings = 4;

    Params* params;            // Problem parameters
    Population* population;    // Population
    LocalSearch* localSearch;  // Local Search structure

    // Pointers for offspring to edit new offspring in place:
    // 0 and 1 are reserved for SREX, 2 and 3 are reserved for OX
    std::array<Individual*, numberOfCandidateOffsprings> candidateOffsprings;

    // Function to do two SREX Crossovers for a pair of individuals (the two parents) and return the
    // best individual based on penalizedCost
    Individual* crossoverSREX(std::pair<const Individual*, const Individual*> parents);

  private:
    // Remove any customers that are not profitable from the current solution and add any
    // that would be. Assumes an exported/evaluateCompleteCost() solution.
    bool mutate(Individual* offspring);
};

#endif
