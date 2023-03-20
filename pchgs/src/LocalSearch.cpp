#include "LocalSearch.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "CircleSector.h"
#include "Individual.h"
#include "Params.h"

namespace {
    class RouteIterator {
      public:
        RouteIterator() = default;
        explicit RouteIterator(Node* pNode) noexcept : m_pCurrentNode(pNode) {}

        RouteIterator& operator=(Node* pNode) {
            this->m_pCurrentNode = pNode;
            return *this;
        }

        // Prefix ++ overload
        RouteIterator& operator++() {
            if (m_pCurrentNode) m_pCurrentNode = m_pCurrentNode->next;
            return *this;
        }

        // Postfix ++ overload
        RouteIterator operator++(int) {
            RouteIterator iterator = *this;
            ++*this;
            return iterator;
        }

        bool operator!=(const RouteIterator& iterator) const {
            return m_pCurrentNode != iterator.m_pCurrentNode;
        }

        bool operator==(const RouteIterator& iterator) const {
            return m_pCurrentNode == iterator.m_pCurrentNode;
        }

        Node& operator*() { return *m_pCurrentNode; }

        Node* operator->() { return m_pCurrentNode; }

      private:
        Node* m_pCurrentNode;
    };

    std::vector<int> bitset_to_vector(const sul::dynamic_bitset<>& bitset) {
        std::vector<int> vec;
        vec.reserve(bitset.size());
        for (auto next_set_bit = bitset.find_first(); next_set_bit != sul::dynamic_bitset<>::npos;
             next_set_bit = bitset.find_next(next_set_bit)) {
            vec.push_back(static_cast<int>(next_set_bit));
        }
        return vec;
    }

    bool operator==(const TimeWindowData& twData1, const TimeWindowData& twData2) {
        return twData1.firstNodeIndex == twData2.firstNodeIndex
               && twData1.lastNodeIndex == twData2.lastNodeIndex
               && twData1.duration == twData2.duration && twData1.timeWarp == twData2.timeWarp
               && twData1.earliestArrival == twData2.earliestArrival
               && twData1.latestArrival == twData2.latestArrival;
    }
}  // namespace

namespace std {
    template <> class iterator_traits<RouteIterator> {
      public:
        using difference_type = std::ptrdiff_t;
        using value_type = Node;
        using pointer = Node*;
        using reference = Node&;
        using iterator_category = std::forward_iterator_tag;
        using iterator_concept = std::forward_iterator_tag;
    };
}  // namespace std

void LocalSearch::initializeConstruction(Individual* indiv,
                                         std::vector<NodeToInsert>* nodesToInsert) {
    // Initialize datastructures relevant for constructions.
    // Local search-related data structures are not initialized.
    emptyRoutes.clear();
    TimeWindowData depotTwData;
    depotTwData.firstNodeIndex = 0;
    depotTwData.lastNodeIndex = 0;
    depotTwData.duration = 0;
    depotTwData.timeWarp = 0;
    depotTwData.earliestArrival = params->cli[0].earliestArrival;
    depotTwData.latestArrival = params->cli[0].latestArrival;

    // Initializing time window data for clients
    for (int i = 1; i <= params->nbClients; i++) {
        TimeWindowData* myTwData = &clients[i].twData;
        myTwData->firstNodeIndex = i;
        myTwData->lastNodeIndex = i;
        myTwData->duration = params->cli[i].serviceDuration;
        myTwData->earliestArrival = params->cli[i].earliestArrival;
        myTwData->latestArrival = params->cli[i].latestArrival;
    }

    // Initialize routes
    for (int r = 0; r < params->nbVehicles; r++) {
        Node* myDepot = &depots[r];
        Node* myDepotFin = &depotsEnd[r];
        myDepot->prev = myDepotFin;
        myDepotFin->next = myDepot;
        myDepot->next = myDepotFin;
        myDepotFin->prev = myDepot;

        myDepot->twData = depotTwData;
        myDepot->prefixTwData = depotTwData;
        myDepot->postfixTwData = depotTwData;

        myDepotFin->twData = depotTwData;
        myDepotFin->prefixTwData = depotTwData;
        myDepotFin->postfixTwData = depotTwData;

        updateRouteData(&routes[r]);
    }

    // Initialize clients.
    for (int i = 1; i <= params->nbClients; i++) {
        if (!indiv->isRequired(i)) {
            continue;
        }

        NodeToInsert nodeToInsert;
        nodeToInsert.clientIdx = i;
        nodeToInsert.twData = clients[i].twData;
        nodeToInsert.load = params->cli[i].demand;
        nodeToInsert.angleFromDepot = atan2(params->cli[i].coordY - params->cli[0].coordY,
                                            params->cli[i].coordX - params->cli[0].coordX);
        nodeToInsert.serviceDuration = params->cli[i].serviceDuration;
        nodesToInsert->push_back(nodeToInsert);
    }
}

void LocalSearch::constructIndividualBySweep(int fillPercentage, Individual* indiv) {
    std::vector<NodeToInsert> nodesToInsert;
    initializeConstruction(indiv, &nodesToInsert);

    std::vector<std::vector<int>> nodeIndicesPerRoute;

    // Sort nodes according to angle with depot.
    std::sort(std::begin(nodesToInsert), std::end(nodesToInsert),
              [](NodeToInsert a, NodeToInsert b) { return a.angleFromDepot < b.angleFromDepot; });

    // Distribute clients over routes.
    int load = 0;
    std::vector<int> nodeIndicesInRoute;
    for (int i = 0; i < static_cast<int>(nodesToInsert.size()); i++) {
        if (load > 0
            && load + nodesToInsert[i].load > fillPercentage * params->vehicleCapacity / 100
            && nodeIndicesPerRoute.size() + 1 < routes.size()) {
            nodeIndicesPerRoute.push_back(nodeIndicesInRoute);
            nodeIndicesInRoute.clear();
            load = 0;
        }

        load += nodesToInsert[i].load;
        nodeIndicesInRoute.push_back(i);
    }

    nodeIndicesPerRoute.push_back(nodeIndicesInRoute);

    // Construct routes
    for (int r = 0; r < static_cast<int>(nodeIndicesPerRoute.size()); r++) {
        int depotOpeningDuration
            = depots[r].twData.latestArrival - depots[r].twData.earliestArrival;
        std::vector<int> nodeIndicesToInsertShortTw;
        std::vector<int> nodeIndicesToInsertLongTw;
        for (int idx : nodeIndicesPerRoute[r]) {
            // Arbitrary division, but for all instances time windows are either much shorter than
            // half of depotOpeningDuration, or much larger.
            if ((nodesToInsert[idx].twData.latestArrival
                 - nodesToInsert[idx].twData.earliestArrival)
                    * 2
                > depotOpeningDuration)
                nodeIndicesToInsertLongTw.push_back(idx);
            else
                nodeIndicesToInsertShortTw.push_back(idx);
        }

        // Sort routes with short time window in increasing end of time window.
        std::sort(std::begin(nodeIndicesToInsertShortTw), std::end(nodeIndicesToInsertShortTw),
                  [&nodesToInsert](int a, int b) {
                      return nodesToInsert[a].twData.latestArrival
                             < nodesToInsert[b].twData.latestArrival;
                  });

        // Insert nodes with short time window in order in the route.
        Node* prev = routes[r].depot;
        for (int i = 0; i < static_cast<int>(nodeIndicesToInsertShortTw.size()); i++) {
            Node* toInsert = &clients[nodesToInsert[nodeIndicesToInsertShortTw[i]].clientIdx];
            Node* insertionPoint = prev;
            toInsert->prev = insertionPoint;
            toInsert->next = insertionPoint->next;
            insertionPoint->next->prev = toInsert;
            insertionPoint->next = toInsert;
            prev = toInsert;
        }

        updateRouteData(&routes[r]);

        // Insert remaining nodes according to best distance
        for (int i = 0; i < static_cast<int>(nodeIndicesToInsertLongTw.size()); i++) {
            double bestCost = std::numeric_limits<double>::max();
            Node* bestPred = nullptr;
            Node* prev = routes[r].depot;
            for (int j = 0; j <= routes[r].nbCustomers; j++) {
                // Compute insertion cost
                double insertionCost
                    = params->getCost(prev->cour,
                                      nodesToInsert[nodeIndicesToInsertLongTw[i]].clientIdx)
                      + params->getCost(nodesToInsert[nodeIndicesToInsertLongTw[i]].clientIdx,
                                        prev->next->cour)
                      - params->getCost(prev->cour, prev->next->cour);

                if (insertionCost < bestCost) {
                    bestCost = insertionCost;
                    bestPred = prev;
                }

                prev = prev->next;
            }

            Node* toInsert = &clients[nodesToInsert[nodeIndicesToInsertLongTw[i]].clientIdx];
            Node* insertionPoint = bestPred;
            toInsert->prev = insertionPoint;
            toInsert->next = insertionPoint->next;
            insertionPoint->next->prev = toInsert;
            insertionPoint->next = toInsert;
            updateRouteData(&routes[r]);
        }
    }

    // Register the solution produced by the construction heuristic in the individual.
    exportIndividual(indiv);
}

void LocalSearch::constructIndividualWithSeedOrder(int toleratedCapacityViolation,
                                                   int toleratedTimeWarp,
                                                   bool useSeedClientFurthestFromDepot,
                                                   Individual* indiv) {
    std::vector<NodeToInsert> nodesToInsert;

    auto mark_inserted = [&nodesToInsert](decltype(nodesToInsert)::iterator iter) {
        std::iter_swap(iter, std::prev(nodesToInsert.end()));
        nodesToInsert.pop_back();
    };

    initializeConstruction(indiv, &nodesToInsert);

    // Construct routes
    for (int r = 0; r < static_cast<int>(routes.size()) && nodesToInsert.size() > 0; r++) {
        // Note that if the seed client is the unassigned client closest to the depot, we do not
        // have to do any initialization and can just start inserting nodes that are best according
        // to distance in the main loop.
        if (useSeedClientFurthestFromDepot) {
            auto furthestNodeIter = nodesToInsert.end();
            double furthestNodeCost = std::numeric_limits<double>::lowest();
            for (auto node_iter = nodesToInsert.begin(); node_iter != nodesToInsert.end();
                 ++node_iter) {
                double insertionCost
                    = params->getCost(routes[r].depot->cour, node_iter->clientIdx)
                      + params->getCost(node_iter->clientIdx, routes[r].depot->next->cour)
                      - params->getCost(routes[r].depot->cour, routes[r].depot->next->cour);

                if (insertionCost > furthestNodeCost) {
                    furthestNodeCost = insertionCost;
                    furthestNodeIter = node_iter;
                }
            }
            if (furthestNodeIter == nodesToInsert.cend()) {
                throw std::runtime_error("No clients to insert!");
            }

            Node* toInsert = &clients[furthestNodeIter->clientIdx];
            toInsert->prev = routes[r].depot;
            toInsert->next = routes[r].depot->next;
            routes[r].depot->next->prev = toInsert;
            routes[r].depot->next = toInsert;
            updateRouteData(&routes[r]);
            mark_inserted(furthestNodeIter);
        }

        bool insertedNode = true;
        while (insertedNode) {
            insertedNode = false;
            double bestCost = std::numeric_limits<double>::max();
            Node* bestPred = nullptr;
            auto best_node_iter = nodesToInsert.begin();
            for (auto node_iter = nodesToInsert.begin(); node_iter != nodesToInsert.end();
                 ++node_iter) {
                // Do not allow insertion if capacity is exceeded more than tolerance.
                if (routes[r].load + node_iter->load
                    > params->vehicleCapacity + toleratedCapacityViolation)
                    continue;

                Node* prev = routes[r].depot;
                for (int j = 0; j <= routes[r].nbCustomers; j++) {
                    // Do not allow insertions if time windows are violated more than tolerance
                    TimeWindowData routeTwData = MergeTWDataRecursive(
                        prev->prefixTwData, node_iter->twData, prev->next->postfixTwData);
                    if (routeTwData.timeWarp > toleratedTimeWarp) {
                        prev = prev->next;
                        continue;
                    }

                    // Compute insertion cost
                    double insertionCost = params->getCost(prev->cour, node_iter->clientIdx)
                                           + params->getCost(node_iter->clientIdx, prev->next->cour)
                                           - params->getCost(prev->cour, prev->next->cour);

                    if (insertionCost < bestCost) {
                        bestCost = insertionCost;
                        bestPred = prev;
                        best_node_iter = node_iter;
                    }

                    prev = prev->next;
                }
            }

            if (bestCost < std::numeric_limits<double>::max()) {
                Node* toInsert = &clients[best_node_iter->clientIdx];
                toInsert->prev = bestPred;
                toInsert->next = bestPred->next;
                bestPred->next->prev = toInsert;
                bestPred->next = toInsert;
                assert(indiv->isRequired(toInsert->cour));
                updateRouteData(&routes[r]);
                insertedNode = true;
                mark_inserted(best_node_iter);
            }
        }
    }

    // Insert all unassigned nodes at the back of the last route. We assume that typically there
    // are no unassigned nodes left, because there are plenty routes, but we have to make sure that
    // all nodes are assigned.
    if (nodesToInsert.size() > 0) {
        int lastRouteIdx = routes.size() - 1;
        Node* prevNode
            = depotsEnd[lastRouteIdx].prev;  // Last node before finish depot in last route.

        while (nodesToInsert.size() > 0) {
            Node* toInsert = &clients[nodesToInsert.back().clientIdx];
            toInsert->prev = prevNode;
            toInsert->next = prevNode->next;
            prevNode->next->prev = toInsert;
            prevNode->next = toInsert;
            nodesToInsert.pop_back();
        }

        updateRouteData(&routes[lastRouteIdx]);
    }

    // Register the solution produced by the construction heuristic in the individual.
    exportIndividual(indiv);
}

void LocalSearch::run(Individual* indiv, double penaltyCapacityLS, double penaltyTimeWarpLS) {
    static const bool neverIntensify = params->config.intensificationProbabilityLS == 0;
    static const bool alwaysIntensify = params->config.intensificationProbabilityLS == 100;
    const bool runLS_INT
        = params->rng() % 100 < (unsigned int)params->config.intensificationProbabilityLS;

    this->penaltyCapacityLS = penaltyCapacityLS;
    this->penaltyTimeWarpLS = penaltyTimeWarpLS;

    loadIndividual(indiv);

    // Shuffling the order of the nodes explored by the LS to allow for more diversity in the search
    std::shuffle(orderNodes.begin(), orderNodes.end(), params->rng);
    std::shuffle(orderRoutes.begin(), orderRoutes.end(), params->rng);
    if (params->rng() % MaxNumberOfCorrelatedVertices() == 0) {
        for (auto& correlated_vertices : correlatedVertices) {
            std::shuffle(correlated_vertices.begin(), correlated_vertices.end(), params->rng);
        }
    }

    searchCompleted = false;
    for (loopID = 0; loopID < 2 * params->nbClients && !searchCompleted; loopID++) {
        if (loopID > 1) {
            // Allows at least two loops since some moves involving empty routes are not checked at
            // the first loop
            searchCompleted = true;
        }

        /* CLASSICAL ROUTE IMPROVEMENT (RI) MOVES SUBJECT TO A PROXIMITY RESTRICTION */
        for (int posU = 0; posU < params->nbClients; posU++) {
            nodeU = &clients[orderNodes[posU]];

            // Never consider nodes which are not part of the current solution as nodeU.
            if (!indiv->isRequired(nodeU->cour)) {
                continue;
            }

            int lastTestRINodeU = nodeU->whenLastTestedRI;
            nodeU->whenLastTestedRI = nbMoves;

            const auto& correlated = correlatedVertices[nodeU->cour];
            unsigned int num_evaluated_correlated_vertices = 0;

            // TODO Skipping is fine, but make sure to try at least min(config.nbGraunlar,
            //  correlated.size()) vertices.
            for (const auto& v : correlated) {
                if (!indiv->isRequired(v)) {
                    continue;
                }
                if (num_evaluated_correlated_vertices >= MaxNumberOfCorrelatedVertices()) {
                    break;
                }
                num_evaluated_correlated_vertices += 1;

                nodeV = &clients[v];
                assert(indiv->isRequired(nodeV->cour));
                if (loopID == 0
                    || std::max(nodeU->route->whenLastModified, nodeV->route->whenLastModified)
                           > lastTestRINodeU)  // only evaluate moves involving routes that have
                                               // been modified since last move evaluations for
                                               // nodeU
                {
                    // Randomizing the order of the neighborhoods within this loop does not matter
                    // much as we are already randomizing the order of the node pairs (and it's not
                    // very common to find improving moves of different types for the same node
                    // pair)
                    setLocalVariablesRouteU();
                    setLocalVariablesRouteV();
                    if (MoveSingleClient()) continue;                                   // RELOCATE
                    if (MoveTwoClients()) continue;                                     // RELOCATE
                    if (MoveTwoClientsReversed()) continue;                             // RELOCATE
                    if (nodeUIndex < nodeVIndex && SwapTwoSingleClients()) continue;    // SWAP
                    if (SwapTwoClientsForOne()) continue;                               // SWAP
                    if (nodeUIndex < nodeVIndex && SwapTwoClientPairs()) continue;      // SWAP
                    if (routeU->cour < routeV->cour && TwoOptBetweenTrips()) continue;  // 2-OPT*
                    if (routeU == routeV && TwoOptWithinTrip()) continue;               // 2-OPT

                    // Trying moves that insert nodeU directly after the depot
                    if (nodeV->prev->isDepot) {
                        nodeV = nodeV->prev;
                        setLocalVariablesRouteV();
                        if (MoveSingleClient()) continue;        // RELOCATE
                        if (MoveTwoClients()) continue;          // RELOCATE
                        if (MoveTwoClientsReversed()) continue;  // RELOCATE
                        if (routeU->cour < routeV->cour && TwoOptBetweenTrips())
                            continue;  // 2-OPT*
                    }
                }
            }

            /* MOVES INVOLVING AN EMPTY ROUTE -- NOT TESTED IN THE FIRST LOOP TO AVOID INCREASING
             * TOO MUCH THE FLEET SIZE */
            if (loopID > 0 && !emptyRoutes.empty()) {
                nodeV = routes[*emptyRoutes.begin()].depot;
                setLocalVariablesRouteU();
                setLocalVariablesRouteV();
                if (MoveSingleClient()) continue;        // RELOCATE
                if (MoveTwoClients()) continue;          // RELOCATE
                if (MoveTwoClientsReversed()) continue;  // RELOCATE
                if (TwoOptBetweenTrips()) continue;      // 2-OPT*
            }

            /* MOVES INVOLVING REMOVING/ADDING A CUSTOMER -- NOT TESTED IN THE FIRST LOOP TO AVOID
             * REMOVING TOO MANY CUSTOMERS */
            if (loopID >= 0) {
                setLocalVariablesRouteU();

                num_evaluated_correlated_vertices = 0;
                for (const auto& v : correlated) {
                    if (indiv->isRequired(v) || params->isCertainlyUnprofitable(v)) {
                        continue;
                    }
                    num_evaluated_correlated_vertices += 1;
                    if (num_evaluated_correlated_vertices >= MaxNumberOfCorrelatedVertices()) {
                        break;
                    }
                    if (InsertSingleClient(indiv, v)) {
                        // This modifies route U and changes node X. Need to reload the vars.
                        setLocalVariablesRouteU();
                    }
                }

                if (!params->isCertainlyProfitable(nodeUIndex)) {
                    if (RemoveSingleClient(indiv)) continue;
                    if (!nodeX->isDepot && !params->isCertainlyProfitable(nodeXIndex)
                        && RemoveTwoClients(indiv)) {
                        continue;
                    }
                }
            }

            /* (SWAP*) MOVES LIMITED TO ROUTE PAIRS WHOSE CIRCLE SECTORS OVERLAP */
            if (!neverIntensify && searchCompleted && (alwaysIntensify || runLS_INT)) {
                for (int rU = 0; rU < params->nbVehicles; rU++) {
                    routeU = &routes[orderRoutes[rU]];
                    if (routeU->nbCustomers == 0) {
                        continue;
                    }

                    int lastTestLargeNbRouteU = routeU->whenLastTestedLargeNb;
                    routeU->whenLastTestedLargeNb = nbMoves;
                    for (int rV = 0; rV < params->nbVehicles; rV++) {
                        routeV = &routes[orderRoutes[rV]];
                        if (routeV->nbCustomers == 0 || routeU->cour >= routeV->cour) {
                            continue;
                        }

                        if (loopID > 0
                            && std::max(routeU->whenLastModified, routeV->whenLastModified)
                                   <= lastTestLargeNbRouteU) {
                            continue;
                        }

                        if (!CircleSector::overlap(routeU->sector, routeV->sector,
                                                   params->circleSectorOverlapTolerance)) {
                            continue;
                        }

                        if (!RelocateStar()) {
                            if (params->config.skipSwapStarDist || !swapStar(false)) {
                                if (params->config.useSwapStarTW) {
                                    swapStar(true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Register the solution produced by the LS in the individual
    exportIndividual(indiv);
}

void LocalSearch::setLocalVariablesRouteU() {
    routeU = nodeU->route;
    nodeX = nodeU->next;
    nodeXNextIndex = nodeX->next->cour;
    nodeUIndex = nodeU->cour;
    nodeUPrevIndex = nodeU->prev->cour;
    nodeXIndex = nodeX->cour;
    loadU = params->cli[nodeUIndex].demand;
    loadX = params->cli[nodeXIndex].demand;
    routeUTimeWarp = routeU->twData.timeWarp > 0;
    routeULoadPenalty = routeU->load > params->vehicleCapacity;
}

void LocalSearch::setLocalVariablesRouteV() {
    routeV = nodeV->route;
    nodeY = nodeV->next;
    nodeYNextIndex = nodeY->next->cour;
    nodeVIndex = nodeV->cour;
    nodeVPrevIndex = nodeV->prev->cour;
    nodeYIndex = nodeY->cour;
    loadV = params->cli[nodeVIndex].demand;
    loadY = params->cli[nodeYIndex].demand;
    routeVTimeWarp = routeV->twData.timeWarp > 0;
    routeVLoadPenalty = routeV->load > params->vehicleCapacity;
}

bool LocalSearch::MoveSingleClient() {
    // If U already comes directly after V, this move has no effect
    if (nodeUIndex == nodeYIndex) {
        // Skipping only makes sense if routeU != routeV
        assert(routeU == routeV);
        return false;
    }

    // Cost of removing U, will likely be negative
    double costSuppU = params->getCost(nodeUPrevIndex, nodeXIndex)
                       - params->getCost(nodeUPrevIndex, nodeUIndex)
                       - params->getCost(nodeUIndex, nodeXIndex);
    // Cost of inserting U at V
    double costSuppV = params->getCost(nodeVIndex, nodeUIndex)
                       + params->getCost(nodeUIndex, nodeYIndex)
                       - params->getCost(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        // RouteU is feasible and it incurs extra cost to move U to V => No improvement
        if (!routeULoadPenalty && !routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeX->postfixTwData);
        routeVTwData
            = MergeTWDataRecursive(nodeV->prefixTwData, nodeU->twData, nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load - loadU) + penaltyTimeWindows(routeUTwData)
                     - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU) + penaltyTimeWindows(routeVTwData)
                     - routeV->penalty;
    } else {
        // RouteU is feasible (Time window wise) and it incurs extra cost to move U to V => No
        // improvement
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Move within the same route
        if (nodeU->position < nodeV->position) {
            // Edge case V directly after U, so X == V, this works
            // start - ... - UPrev - X - ... - V - U - Y - ... - end
            routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData,
                                                getRouteSegmentTwData(nodeX, nodeV), nodeU->twData,
                                                nodeY->postfixTwData);
        } else {
            // Edge case U directly after V is excluded from beginning of function
            // start - ... - V - U - Y - ... - UPrev - X - ... - end
            routeUTwData = MergeTWDataRecursive(nodeV->prefixTwData, nodeU->twData,
                                                getRouteSegmentTwData(nodeY, nodeU->prev),
                                                nodeX->postfixTwData);
        }

        // Compute new total penalty
        costSuppU
            += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    insertNode(nodeU, nodeV);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool LocalSearch::MoveTwoClients() {
    if (nodeU == nodeY || nodeV == nodeX || nodeX->isDepot) return false;

    double costSuppU = params->getCost(nodeUPrevIndex, nodeXNextIndex)
                       - params->getCost(nodeUPrevIndex, nodeUIndex)
                       - params->getCost(nodeXIndex, nodeXNextIndex);
    double costSuppV = params->getCost(nodeVIndex, nodeUIndex)
                       + params->getCost(nodeXIndex, nodeYIndex)
                       - params->getCost(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prefixTwData, getEdgeTwData(nodeU, nodeX),
                                            nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load - loadU - loadX)
                     + penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU + loadX)
                     + penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Move within the same route
        if (nodeU->position < nodeV->position) {
            // Edge case V directly after U, so X == V is excluded, V directly after X so XNext == V
            // works start - ... - UPrev - XNext - ... - V - U - X - Y - ... - end
            routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData,
                                                getRouteSegmentTwData(nodeX->next, nodeV),
                                                getEdgeTwData(nodeU, nodeX), nodeY->postfixTwData);
        } else {
            // Edge case U directly after V is excluded from beginning of function
            // start - ... - V - U - X - Y - ... - UPrev - XNext - ... - end
            routeUTwData = MergeTWDataRecursive(nodeV->prefixTwData, getEdgeTwData(nodeU, nodeX),
                                                getRouteSegmentTwData(nodeY, nodeU->prev),
                                                nodeX->next->postfixTwData);
        }

        // Compute new total penalty
        costSuppU
            += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    insertNode(nodeU, nodeV);
    insertNode(nodeX, nodeU);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool LocalSearch::MoveTwoClientsReversed() {
    if (nodeU == nodeY || nodeX == nodeV || nodeX->isDepot) return false;

    double costSuppU = params->getCost(nodeUPrevIndex, nodeXNextIndex)
                       - params->getCost(nodeUPrevIndex, nodeUIndex)
                       - params->getCost(nodeUIndex, nodeXIndex)
                       - params->getCost(nodeXIndex, nodeXNextIndex);
    double costSuppV
        = params->getCost(nodeVIndex, nodeXIndex) + params->getCost(nodeXIndex, nodeUIndex)
          + params->getCost(nodeUIndex, nodeYIndex) - params->getCost(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prefixTwData, getEdgeTwData(nodeX, nodeU),
                                            nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load - loadU - loadX)
                     + penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU + loadX)
                     + penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Move within the same route
        if (nodeU->position < nodeV->position) {
            // Edge case V directly after U, so X == V is excluded, V directly after X so XNext == V
            // works start - ... - UPrev - XNext - ... - V - X - U - Y - ... - end
            routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData,
                                                getRouteSegmentTwData(nodeX->next, nodeV),
                                                getEdgeTwData(nodeX, nodeU), nodeY->postfixTwData);
        } else {
            // Edge case U directly after V is excluded from beginning of function
            // start - ... - V - X - U - Y - ... - UPrev - XNext - ... - end
            routeUTwData = MergeTWDataRecursive(nodeV->prefixTwData, getEdgeTwData(nodeX, nodeU),
                                                getRouteSegmentTwData(nodeY, nodeU->prev),
                                                nodeX->next->postfixTwData);
        }

        // Compute new total penalty
        costSuppU
            += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    insertNode(nodeX, nodeV);
    insertNode(nodeU, nodeX);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool LocalSearch::SwapTwoSingleClients() {
    if (nodeUIndex == nodeVPrevIndex || nodeUIndex == nodeYIndex) return false;

    double costSuppU
        = params->getCost(nodeUPrevIndex, nodeVIndex) + params->getCost(nodeVIndex, nodeXIndex)
          - params->getCost(nodeUPrevIndex, nodeUIndex) - params->getCost(nodeUIndex, nodeXIndex);
    double costSuppV
        = params->getCost(nodeVPrevIndex, nodeUIndex) + params->getCost(nodeUIndex, nodeYIndex)
          - params->getCost(nodeVPrevIndex, nodeVIndex) - params->getCost(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty && !routeVTimeWarp
            && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData
            = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeV->twData, nodeX->postfixTwData);
        routeVTwData
            = MergeTWDataRecursive(nodeV->prev->prefixTwData, nodeU->twData, nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load + loadV - loadU)
                     + penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU - loadV)
                     + penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Swap within the same route
        if (nodeU->position < nodeV->position) {
            // Edge case V directly after U, so X == V is excluded, V directly after X so XNext == V
            // works start - ... - UPrev - V - X - ... - VPrev - U - Y - ... - end
            routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeV->twData,
                                                getRouteSegmentTwData(nodeX, nodeV->prev),
                                                nodeU->twData, nodeY->postfixTwData);
        } else {
            // Edge case U directly after V is excluded from beginning of function
            // start - ... - VPrev - U - Y - ... - UPrev - V - X - ... - end
            routeUTwData = MergeTWDataRecursive(nodeV->prev->prefixTwData, nodeU->twData,
                                                getRouteSegmentTwData(nodeY, nodeU->prev),
                                                nodeV->twData, nodeX->postfixTwData);
        }

        // Compute new total penalty
        costSuppU
            += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    swapNode(nodeU, nodeV);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool LocalSearch::SwapTwoClientsForOne() {
    if (nodeU == nodeV->prev || nodeX == nodeV->prev || nodeU == nodeY || nodeX->isDepot)
        return false;

    double costSuppU = params->getCost(nodeUPrevIndex, nodeVIndex)
                       + params->getCost(nodeVIndex, nodeXNextIndex)
                       - params->getCost(nodeUPrevIndex, nodeUIndex)
                       - params->getCost(nodeXIndex, nodeXNextIndex);
    double costSuppV
        = params->getCost(nodeVPrevIndex, nodeUIndex) + params->getCost(nodeXIndex, nodeYIndex)
          - params->getCost(nodeVPrevIndex, nodeVIndex) - params->getCost(nodeVIndex, nodeYIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty && !routeVTimeWarp
            && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeV->twData,
                                            nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prev->prefixTwData, getEdgeTwData(nodeU, nodeX),
                                            nodeY->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load + loadV - loadU - loadX)
                     + penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU + loadX - loadV)
                     + penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Swap within the same route
        if (nodeU->position < nodeV->position) {
            // start - ... - UPrev - V - XNext - ... - VPrev - U - X - Y - ... - end
            routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeV->twData,
                                                getRouteSegmentTwData(nodeX->next, nodeV->prev),
                                                getEdgeTwData(nodeU, nodeX), nodeY->postfixTwData);
        } else {
            // start - ... - VPrev - U - X - Y - ... - UPrev - V - XNext - ... - end
            routeUTwData
                = MergeTWDataRecursive(nodeV->prev->prefixTwData, getEdgeTwData(nodeU, nodeX),
                                       getRouteSegmentTwData(nodeY, nodeU->prev), nodeV->twData,
                                       nodeX->next->postfixTwData);
        }

        // Compute new total penalty
        costSuppU
            += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    // Note: next two lines are a bit inefficient but we only update occasionally and
    // updateRouteData is much more costly anyway, efficient checks are more important
    swapNode(nodeU, nodeV);
    insertNode(nodeX, nodeU);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool LocalSearch::SwapTwoClientPairs() {
    if (nodeX->isDepot || nodeY->isDepot || nodeY == nodeU->prev || nodeU == nodeY || nodeX == nodeV
        || nodeV == nodeX->next)
        return false;

    double costSuppU = params->getCost(nodeUPrevIndex, nodeVIndex)
                       + params->getCost(nodeYIndex, nodeXNextIndex)
                       - params->getCost(nodeUPrevIndex, nodeUIndex)
                       - params->getCost(nodeXIndex, nodeXNextIndex);
    double costSuppV = params->getCost(nodeVPrevIndex, nodeUIndex)
                       + params->getCost(nodeXIndex, nodeYNextIndex)
                       - params->getCost(nodeVPrevIndex, nodeVIndex)
                       - params->getCost(nodeYIndex, nodeYNextIndex);
    TimeWindowData routeUTwData, routeVTwData;

    if (routeU != routeV) {
        if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty && !routeVTimeWarp
            && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        routeUTwData = MergeTWDataRecursive(nodeU->prev->prefixTwData, getEdgeTwData(nodeV, nodeY),
                                            nodeX->next->postfixTwData);
        routeVTwData = MergeTWDataRecursive(nodeV->prev->prefixTwData, getEdgeTwData(nodeU, nodeX),
                                            nodeY->next->postfixTwData);

        costSuppU += penaltyExcessLoad(routeU->load + loadV + loadY - loadU - loadX)
                     + penaltyTimeWindows(routeUTwData) - routeU->penalty;

        costSuppV += penaltyExcessLoad(routeV->load + loadU + loadX - loadV - loadY)
                     + penaltyTimeWindows(routeVTwData) - routeV->penalty;
    } else {
        if (!routeUTimeWarp && costSuppU + costSuppV > -MY_EPSILON) {
            return false;
        }

        // Swap within the same route
        if (nodeU->position < nodeV->position) {
            // start - ... - UPrev - V - Y - XNext - ... - VPrev - U - X - YNext  - ... - end
            routeUTwData
                = MergeTWDataRecursive(nodeU->prev->prefixTwData, getEdgeTwData(nodeV, nodeY),
                                       getRouteSegmentTwData(nodeX->next, nodeV->prev),
                                       getEdgeTwData(nodeU, nodeX), nodeY->next->postfixTwData);
        } else {
            // start - ... - VPrev - U - X - YNext - ... - UPrev - V - Y - XNext - ... - end
            routeUTwData
                = MergeTWDataRecursive(nodeV->prev->prefixTwData, getEdgeTwData(nodeU, nodeX),
                                       getRouteSegmentTwData(nodeY->next, nodeU->prev),
                                       getEdgeTwData(nodeV, nodeY), nodeX->next->postfixTwData);
        }

        // Compute new total penalty
        costSuppU
            += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeUTwData) - routeU->penalty;
    }

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    swapNode(nodeU, nodeV);
    swapNode(nodeX, nodeY);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    if (routeU != routeV) updateRouteData(routeV);

    return true;
}

bool LocalSearch::TwoOptWithinTrip() {
    if (nodeU->position >= nodeV->position - 1) return false;

    double cost = params->getCost(nodeUIndex, nodeVIndex) + params->getCost(nodeXIndex, nodeYIndex)
                  - params->getCost(nodeUIndex, nodeXIndex)
                  - params->getCost(nodeVIndex, nodeYIndex) + nodeV->cumulatedReversalDistance
                  - nodeX->cumulatedReversalDistance;

    if (!routeUTimeWarp && cost > -MY_EPSILON) {
        return false;
    }

    TimeWindowData routeTwData = nodeU->prefixTwData;
    Node* itRoute = nodeV;
    while (itRoute != nodeU) {
        routeTwData = MergeTWDataRecursive(routeTwData, itRoute->twData);
        itRoute = itRoute->prev;
    }
    routeTwData = MergeTWDataRecursive(routeTwData, nodeY->postfixTwData);

    // Compute new total penalty
    cost += penaltyExcessLoad(routeU->load) + penaltyTimeWindows(routeTwData) - routeU->penalty;

    if (cost > -MY_EPSILON) {
        return false;
    }

    itRoute = nodeV;
    Node* insertionPoint = nodeU;
    while (itRoute != nodeX)  // No need to move x, we pivot around it
    {
        Node* current = itRoute;
        itRoute = itRoute->prev;
        insertNode(current, insertionPoint);
        insertionPoint = current;
    }

    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);

    return true;
}

bool LocalSearch::TwoOptBetweenTrips() {
    double costSuppU
        = params->getCost(nodeUIndex, nodeYIndex) - params->getCost(nodeUIndex, nodeXIndex);
    double costSuppV
        = params->getCost(nodeVIndex, nodeXIndex) - params->getCost(nodeVIndex, nodeYIndex);

    if (!routeULoadPenalty && !routeUTimeWarp && !routeVLoadPenalty && !routeVTimeWarp
        && costSuppU + costSuppV > -MY_EPSILON) {
        return false;
    }

    TimeWindowData routeUTwData, routeVTwData;

    routeUTwData = MergeTWDataRecursive(nodeU->prefixTwData, nodeY->postfixTwData);
    routeVTwData = MergeTWDataRecursive(nodeV->prefixTwData, nodeX->postfixTwData);

    costSuppU += penaltyExcessLoad(nodeU->cumulatedLoad + routeV->load - nodeV->cumulatedLoad)
                 + penaltyTimeWindows(routeUTwData) - routeU->penalty;

    costSuppV += penaltyExcessLoad(nodeV->cumulatedLoad + routeU->load - nodeU->cumulatedLoad)
                 + penaltyTimeWindows(routeVTwData) - routeV->penalty;

    if (costSuppU + costSuppV > -MY_EPSILON) return false;

    Node* itRouteV = nodeY;
    Node* insertLocation = nodeU;
    while (!itRouteV->isDepot) {
        Node* current = itRouteV;
        itRouteV = itRouteV->next;
        insertNode(current, insertLocation);
        insertLocation = current;
    }

    Node* itRouteU = nodeX;
    insertLocation = nodeV;
    while (!itRouteU->isDepot) {
        Node* current = itRouteU;
        itRouteU = itRouteU->next;
        insertNode(current, insertLocation);
        insertLocation = current;
    }

    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(routeV);

    return true;
}

bool LocalSearch::RemoveSingleClient(Individual* individual) {
    assert(individual->isRequired(nodeUIndex));
    auto improvement = calculateRemovalImprov(nodeU) - params->cli[nodeUIndex].profit;
    if (improvement < MY_EPSILON) return false;

    // Remove U from route
    nodeU->next->prev = nodeU->prev;
    nodeU->prev->next = nodeU->next;
    individual->removeCustomer(nodeUIndex, false);

    searchCompleted = false;
    nbMoves++;

    updateRouteData(routeU);

    return true;
}

bool LocalSearch::RemoveTwoClients(Individual* individual) {
    assert(individual->isRequired(nodeUIndex));
    // Second condition is actually superfluous.
    assert(!nodeX->isDepot && individual->isRequired(nodeXIndex));
    auto improvement = calculateRemovalImprov(nodeU, nodeX) - params->cli[nodeUIndex].profit
                       - params->cli[nodeXIndex].profit;
    if (improvement < MY_EPSILON) return false;

    // Remove U from route
    nodeX->next->prev = nodeU->prev;
    nodeU->prev->next = nodeX->next;

    individual->removeCustomer(nodeUIndex, false);
    individual->removeCustomer(nodeXIndex, false);

    nbMoves++;
    searchCompleted = false;

    updateRouteData(routeU);

    return true;
}

bool LocalSearch::InsertSingleClient(Individual* individual, int v) {
    assert(!params->isCertainlyUnprofitable(v));
    assert(!individual->isRequired(v));
    assert(&depotsEnd[routeU->cour] != nodeU);
    assert(nodeX == nodeU->next);
    // Try to insert customer V after customer U
    auto improvement = calculateInsertionImprov(nodeU, v, nodeX) + params->cli[v].profit;
    if (improvement < MY_EPSILON) return false;

    Node* inserted_node = &clients[v];
    assert(inserted_node->cour == v);
    //  Insert at correct position
    inserted_node->prev = nodeU;
    inserted_node->next = nodeX;
    inserted_node->route = routeU;
    nodeU->next = inserted_node;
    nodeX->prev = inserted_node;
    //  Mark as added
    individual->addCustomer(v, true);
    //  Set search completed to false and increase nbMoves
    nbMoves++;
    searchCompleted = false;
    updateRouteData(routeU);

    return true;
}

bool LocalSearch::swapStar(const bool withTW) {
    SwapStarElement myBestSwapStar;

    if (!bestInsertInitializedForRoute[routeU->cour]) {
        bestInsertInitializedForRoute[routeU->cour] = true;
        for (int i = 1; i <= params->nbClients; i++) {
            bestInsertClient[routeU->cour][i].whenLastCalculated = -1;
            bestInsertClientTW[routeU->cour][i].whenLastCalculated = -1;
        }
    }
    if (!bestInsertInitializedForRoute[routeV->cour]) {
        bestInsertInitializedForRoute[routeV->cour] = true;
        for (int i = 1; i <= params->nbClients; i++) {
            bestInsertClient[routeV->cour][i].whenLastCalculated = -1;
            bestInsertClientTW[routeV->cour][i].whenLastCalculated = -1;
        }
    }

    // Preprocessing insertion costs
    if (withTW) {
        preprocessInsertionsWithTW(routeU, routeV);
        preprocessInsertionsWithTW(routeV, routeU);
    } else {
        preprocessInsertions(routeU, routeV);
        preprocessInsertions(routeV, routeU);
    }

    // Evaluating the moves
    for (nodeU = routeU->depot->next; !nodeU->isDepot; nodeU = nodeU->next) {
        for (nodeV = routeV->depot->next; !nodeV->isDepot; nodeV = nodeV->next) {
            // We cannot determine impact on timewarp without adding too much complexity (O(n^3)
            // instead of O(n^2))
            const double loadPenU = penaltyExcessLoad(routeU->load + params->cli[nodeV->cour].demand
                                                      - params->cli[nodeU->cour].demand);
            const double loadPenV = penaltyExcessLoad(routeV->load + params->cli[nodeU->cour].demand
                                                      - params->cli[nodeV->cour].demand);
            const double deltaLoadPen = loadPenU + loadPenV - penaltyExcessLoad(routeU->load)
                                        - penaltyExcessLoad(routeV->load);
            const int deltaRemoval = withTW ? nodeU->deltaRemovalTW + nodeV->deltaRemovalTW
                                            : nodeU->deltaRemoval + nodeV->deltaRemoval;

            // Quick filter: possibly early elimination of many SWAP* due to the capacity
            // constraints/penalties and bounds on insertion costs
            if (deltaLoadPen + deltaRemoval <= 0) {
                SwapStarElement mySwapStar;
                mySwapStar.U = nodeU;
                mySwapStar.V = nodeV;

                int extraV, extraU;
                if (withTW) {
                    // Evaluate best reinsertion cost of U in the route of V where V has been
                    // removed
                    extraV = getCheapestInsertSimultRemovalWithTW(nodeU, nodeV,
                                                                  mySwapStar.bestPositionU);

                    // Evaluate best reinsertion cost of V in the route of U where U has been
                    // removed
                    extraU = getCheapestInsertSimultRemovalWithTW(nodeV, nodeU,
                                                                  mySwapStar.bestPositionV);
                } else {
                    // Evaluate best reinsertion cost of U in the route of V where V has been
                    // removed
                    extraV = getCheapestInsertSimultRemoval(nodeU, nodeV, mySwapStar.bestPositionU);

                    // Evaluate best reinsertion cost of V in the route of U where U has been
                    // removed
                    extraU = getCheapestInsertSimultRemoval(nodeV, nodeU, mySwapStar.bestPositionV);
                }

                // Evaluating final cost
                mySwapStar.moveCost = deltaLoadPen + deltaRemoval + extraU + extraV;

                if (mySwapStar.moveCost < myBestSwapStar.moveCost) {
                    myBestSwapStar = mySwapStar;
                    myBestSwapStar.loadPenU = loadPenU;
                    myBestSwapStar.loadPenV = loadPenV;
                }
            }
        }
    }

    if (!myBestSwapStar.bestPositionU || !myBestSwapStar.bestPositionV) {
        return false;
    }

    // Compute actual cost including TimeWarp penalty
    double costSuppU = params->getCost(myBestSwapStar.bestPositionV->cour, myBestSwapStar.V->cour)
                       - params->getCost(myBestSwapStar.U->prev->cour, myBestSwapStar.U->cour)
                       - params->getCost(myBestSwapStar.U->cour, myBestSwapStar.U->next->cour);
    double costSuppV = params->getCost(myBestSwapStar.bestPositionU->cour, myBestSwapStar.U->cour)
                       - params->getCost(myBestSwapStar.V->prev->cour, myBestSwapStar.V->cour)
                       - params->getCost(myBestSwapStar.V->cour, myBestSwapStar.V->next->cour);

    if (myBestSwapStar.bestPositionV == myBestSwapStar.U->prev) {
        // Insert in place of U
        costSuppU += params->getCost(myBestSwapStar.V->cour, myBestSwapStar.U->next->cour);
    } else {
        costSuppU
            += params->getCost(myBestSwapStar.V->cour, myBestSwapStar.bestPositionV->next->cour)
               + params->getCost(myBestSwapStar.U->prev->cour, myBestSwapStar.U->next->cour)
               - params->getCost(myBestSwapStar.bestPositionV->cour,
                                 myBestSwapStar.bestPositionV->next->cour);
    }

    if (myBestSwapStar.bestPositionU == myBestSwapStar.V->prev) {
        // Insert in place of V
        costSuppV += params->getCost(myBestSwapStar.U->cour, myBestSwapStar.V->next->cour);
    } else {
        costSuppV
            += params->getCost(myBestSwapStar.U->cour, myBestSwapStar.bestPositionU->next->cour)
               + params->getCost(myBestSwapStar.V->prev->cour, myBestSwapStar.V->next->cour)
               - params->getCost(myBestSwapStar.bestPositionU->cour,
                                 myBestSwapStar.bestPositionU->next->cour);
        ;
    }

    TimeWindowData routeUTwData, routeVTwData;
    // It is not possible to have bestPositionU == V or bestPositionV == U, so the positions are
    // always strictly different
    if (myBestSwapStar.bestPositionV->position == myBestSwapStar.U->position - 1) {
        // Special case
        routeUTwData
            = MergeTWDataRecursive(myBestSwapStar.bestPositionV->prefixTwData,
                                   myBestSwapStar.V->twData, myBestSwapStar.U->next->postfixTwData);
    } else if (myBestSwapStar.bestPositionV->position < myBestSwapStar.U->position) {
        routeUTwData = MergeTWDataRecursive(
            myBestSwapStar.bestPositionV->prefixTwData, myBestSwapStar.V->twData,
            getRouteSegmentTwData(myBestSwapStar.bestPositionV->next, myBestSwapStar.U->prev),
            myBestSwapStar.U->next->postfixTwData);
    } else {
        routeUTwData = MergeTWDataRecursive(
            myBestSwapStar.U->prev->prefixTwData,
            getRouteSegmentTwData(myBestSwapStar.U->next, myBestSwapStar.bestPositionV),
            myBestSwapStar.V->twData, myBestSwapStar.bestPositionV->next->postfixTwData);
    }

    if (myBestSwapStar.bestPositionU->position == myBestSwapStar.V->position - 1) {
        // Special case
        routeVTwData
            = MergeTWDataRecursive(myBestSwapStar.bestPositionU->prefixTwData,
                                   myBestSwapStar.U->twData, myBestSwapStar.V->next->postfixTwData);
    } else if (myBestSwapStar.bestPositionU->position < myBestSwapStar.V->position) {
        routeVTwData = MergeTWDataRecursive(
            myBestSwapStar.bestPositionU->prefixTwData, myBestSwapStar.U->twData,
            getRouteSegmentTwData(myBestSwapStar.bestPositionU->next, myBestSwapStar.V->prev),
            myBestSwapStar.V->next->postfixTwData);
    } else {
        routeVTwData = MergeTWDataRecursive(
            myBestSwapStar.V->prev->prefixTwData,
            getRouteSegmentTwData(myBestSwapStar.V->next, myBestSwapStar.bestPositionU),
            myBestSwapStar.U->twData, myBestSwapStar.bestPositionU->next->postfixTwData);
    }

    costSuppU += myBestSwapStar.loadPenU + penaltyTimeWindows(routeUTwData) - routeU->penalty;

    costSuppV += myBestSwapStar.loadPenV + penaltyTimeWindows(routeVTwData) - routeV->penalty;

    if (costSuppU + costSuppV > -MY_EPSILON) {
        return false;
    }

    // Applying the best move in case of improvement
    insertNode(myBestSwapStar.U, myBestSwapStar.bestPositionU);
    insertNode(myBestSwapStar.V, myBestSwapStar.bestPositionV);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(routeV);

    return true;
}

bool LocalSearch::RelocateStar() {
    double bestCost = 0;
    Node* insertionPoint = nullptr;
    Node* nodeToInsert = nullptr;
    for (nodeU = routeU->depot->next; !nodeU->isDepot; nodeU = nodeU->next) {
        setLocalVariablesRouteU();

        const TimeWindowData routeUTwData
            = MergeTWDataRecursive(nodeU->prev->prefixTwData, nodeX->postfixTwData);
        const double costSuppU = params->getCost(nodeUPrevIndex, nodeXIndex)
                                 - params->getCost(nodeUPrevIndex, nodeUIndex)
                                 - params->getCost(nodeUIndex, nodeXIndex)
                                 + penaltyExcessLoad(routeU->load - loadU)
                                 + penaltyTimeWindows(routeUTwData) - routeU->penalty;

        for (Node* V = routeV->depot->next; !V->isDepot; V = V->next) {
            const TimeWindowData routeVTwData
                = MergeTWDataRecursive(V->prefixTwData, nodeU->twData, V->next->postfixTwData);
            double costSuppV = params->getCost(V->cour, nodeUIndex)
                               + params->getCost(nodeUIndex, V->next->cour)
                               - params->getCost(V->cour, V->next->cour)
                               + penaltyExcessLoad(routeV->load + loadU)
                               + penaltyTimeWindows(routeVTwData) - routeV->penalty;
            if (costSuppU + costSuppV < bestCost - MY_EPSILON) {
                bestCost = costSuppU + costSuppV;
                insertionPoint = V;
                nodeToInsert = nodeU;
            }
        }
    }

    if (!insertionPoint) {
        return false;
    }

    routeU = nodeToInsert->route;
    insertNode(nodeToInsert, insertionPoint);
    nbMoves++;  // Increment move counter before updating route data
    searchCompleted = false;
    updateRouteData(routeU);
    updateRouteData(insertionPoint->route);

    return true;
}

int LocalSearch::getCheapestInsertSimultRemoval(Node* U, Node* V, Node*& bestPosition) {
    ThreeBestInsert* myBestInsert = &bestInsertClient[V->route->cour][U->cour];
    bool found = false;

    // Find best insertion in the route such that V is not next or pred (can only belong to the top
    // three locations)
    bestPosition = myBestInsert->bestLocation[0];
    int bestCost = myBestInsert->bestCost[0];
    found = (bestPosition != V && bestPosition->next != V);
    if (!found && myBestInsert->bestLocation[1] != nullptr) {
        bestPosition = myBestInsert->bestLocation[1];
        bestCost = myBestInsert->bestCost[1];
        found = (bestPosition != V && bestPosition->next != V);
        if (!found && myBestInsert->bestLocation[2] != nullptr) {
            bestPosition = myBestInsert->bestLocation[2];
            bestCost = myBestInsert->bestCost[2];
            found = true;
        }
    }

    // Compute insertion in the place of V
    int deltaCost = params->getCost(V->prev->cour, U->cour)
                    + params->getCost(U->cour, V->next->cour)
                    - params->getCost(V->prev->cour, V->next->cour);
    if (!found || deltaCost < bestCost) {
        bestPosition = V->prev;
        bestCost = deltaCost;
    }

    return bestCost;
}

// TODO make this double as cost??
int LocalSearch::getCheapestInsertSimultRemovalWithTW(Node* U, Node* V, Node*& bestPosition) {
    // TODO ThreeBestInsert must also use double as cost?
    ThreeBestInsert* myBestInsert = &bestInsertClientTW[V->route->cour][U->cour];
    bool found = false;

    // Find best insertion in the route such that V is not next or pred (can only belong to the top
    // three locations)
    bestPosition = myBestInsert->bestLocation[0];
    int bestCost = myBestInsert->bestCost[0];
    found = (bestPosition != V && bestPosition->next != V);
    if (!found && myBestInsert->bestLocation[1] != nullptr) {
        bestPosition = myBestInsert->bestLocation[1];
        bestCost = myBestInsert->bestCost[1];
        found = (bestPosition != V && bestPosition->next != V);
        if (!found && myBestInsert->bestLocation[2] != nullptr) {
            bestPosition = myBestInsert->bestLocation[2];
            bestCost = myBestInsert->bestCost[2];
            found = true;
        }
    }

    // Compute insertion in the place of V
    TimeWindowData twData
        = MergeTWDataRecursive(V->prev->prefixTwData, U->twData, V->next->postfixTwData);
    int deltaCost = params->getCost(V->prev->cour, U->cour)
                    + params->getCost(U->cour, V->next->cour)
                    - params->getCost(V->prev->cour, V->next->cour)
                    + deltaPenaltyTimeWindows(twData, V->route->twData);
    if (!found || deltaCost < bestCost) {
        bestPosition = V->prev;
        bestCost = deltaCost;
    }

    return bestCost;
}

void LocalSearch::preprocessInsertions(Route* R1, Route* R2) {
    for (Node* U = R1->depot->next; !U->isDepot; U = U->next) {
        // Performs the preprocessing
        U->deltaRemoval = params->getCost(U->prev->cour, U->next->cour)
                          - params->getCost(U->prev->cour, U->cour)
                          - params->getCost(U->cour, U->next->cour);
        auto& currentOption = bestInsertClient[R2->cour][U->cour];
        if (R2->whenLastModified > currentOption.whenLastCalculated) {
            currentOption.reset();
            currentOption.whenLastCalculated = nbMoves;
            currentOption.bestCost[0] = params->getCost(0, U->cour)
                                        + params->getCost(U->cour, R2->depot->next->cour)
                                        - params->getCost(0, R2->depot->next->cour);
            currentOption.bestLocation[0] = R2->depot;
            for (Node* V = R2->depot->next; !V->isDepot; V = V->next) {
                int deltaCost = params->getCost(V->cour, U->cour)
                                + params->getCost(U->cour, V->next->cour)
                                - params->getCost(V->cour, V->next->cour);
                currentOption.compareAndAdd(deltaCost, V);
            }
        }
    }
}

void LocalSearch::preprocessInsertionsWithTW(Route* R1, Route* R2) {
    TimeWindowData twData;
    for (Node* U = R1->depot->next; !U->isDepot; U = U->next) {
        // Performs the preprocessing
        // Note: when removing U and adding V to a route, the timewarp penalties may interact,
        // however in most cases it will hold that the reduced timewarp from removing U + added
        // timewarp from adding V will be bigger than the actual delta timewarp such that assuming
        // independence gives a conservative estimate

        if (R1->isDeltaRemovalTWOutdated) {
            twData = MergeTWDataRecursive(U->prev->prefixTwData, U->next->postfixTwData);
            U->deltaRemovalTW = params->getCost(U->prev->cour, U->next->cour)
                                - params->getCost(U->prev->cour, U->cour)
                                - params->getCost(U->cour, U->next->cour)
                                + deltaPenaltyTimeWindows(twData, R1->twData);
        }
        auto& currentOption = bestInsertClientTW[R2->cour][U->cour];
        if (R2->whenLastModified > currentOption.whenLastCalculated) {
            currentOption.reset();
            currentOption.whenLastCalculated = nbMoves;

            // Compute additional timewarp we get when inserting U in R2, this may be actually less
            // if we remove U but we ignore this to have a conservative estimate
            twData = MergeTWDataRecursive(R2->depot->prefixTwData, U->twData,
                                          R2->depot->next->postfixTwData);

            currentOption.bestCost[0] = params->getCost(0, U->cour)
                                        + params->getCost(U->cour, R2->depot->next->cour)
                                        - params->getCost(0, R2->depot->next->cour)
                                        + deltaPenaltyTimeWindows(twData, R2->twData);

            currentOption.bestLocation[0] = R2->depot;
            for (Node* V = R2->depot->next; !V->isDepot; V = V->next) {
                twData = MergeTWDataRecursive(V->prefixTwData, U->twData, V->next->postfixTwData);
                int deltaCost = params->getCost(V->cour, U->cour)
                                + params->getCost(U->cour, V->next->cour)
                                - params->getCost(V->cour, V->next->cour)
                                + deltaPenaltyTimeWindows(twData, R2->twData);
                currentOption.compareAndAdd(deltaCost, V);
            }
        }
    }
    R1->isDeltaRemovalTWOutdated = false;
}

TimeWindowData LocalSearch::getEdgeTwData(Node* U, Node* V) {
    // TODO this could be cached?
    return MergeTWDataRecursive(U->twData, V->twData);
}

TimeWindowData LocalSearch::getRouteSegmentTwData(Node* U, Node* V) {
    if (U->isDepot) return V->prefixTwData;
    if (V->isDepot) return U->postfixTwData;

    // Struct so this makes a copy
    TimeWindowData twData = U->twData;

    Node* mynode = U;
    const int targetPos = V->position;
    while (!(mynode == V)) {
        if (mynode->isSeed && mynode->position + 4 <= targetPos) {
            twData = MergeTWDataRecursive(twData, mynode->toNextSeedTwD);
            mynode = mynode->nextSeed;
        } else {
            mynode = mynode->next;
            twData = MergeTWDataRecursive(twData, mynode->twData);
        }
    }
    return twData;
}

TimeWindowData LocalSearch::MergeTWDataRecursive(const TimeWindowData& twData1,
                                                 const TimeWindowData& twData2) const {
    TimeWindowData mergedTwData;
    // Note, assume time equals cost
    int deltaDuration = params->getTravelTime(twData1.lastNodeIndex, twData2.firstNodeIndex);
    int delta = twData1.duration - twData1.timeWarp + deltaDuration;
    int deltaWaitTime = std::max(twData2.earliestArrival - delta - twData1.latestArrival, 0);
    int deltaTimeWarp = std::max(twData1.earliestArrival + delta - twData2.latestArrival, 0);
    mergedTwData.firstNodeIndex = twData1.firstNodeIndex;
    mergedTwData.lastNodeIndex = twData2.lastNodeIndex;
    mergedTwData.duration = twData1.duration + twData2.duration + deltaDuration + deltaWaitTime;
    mergedTwData.timeWarp = twData1.timeWarp + twData2.timeWarp + deltaTimeWarp;
    mergedTwData.earliestArrival
        = std::max(twData2.earliestArrival - delta, twData1.earliestArrival) - deltaWaitTime;
    mergedTwData.latestArrival
        = std::min(twData2.latestArrival - delta, twData1.latestArrival) + deltaTimeWarp;
    mergedTwData.latestReleaseTime = std::max(twData1.latestReleaseTime, twData2.latestReleaseTime);
    return mergedTwData;
}

void LocalSearch::insertNode(Node* toInsert, Node* insertionPoint) {
    toInsert->prev->next = toInsert->next;
    toInsert->next->prev = toInsert->prev;
    insertionPoint->next->prev = toInsert;
    toInsert->prev = insertionPoint;
    toInsert->next = insertionPoint->next;
    insertionPoint->next = toInsert;
    toInsert->route = insertionPoint->route;
}

void LocalSearch::swapNode(Node* U, Node* V) {
    Node* myVPred = V->prev;
    Node* myVSuiv = V->next;
    Node* myUPred = U->prev;
    Node* myUSuiv = U->next;
    Route* myRouteU = U->route;
    Route* myRouteV = V->route;

    myUPred->next = V;
    myUSuiv->prev = V;
    myVPred->next = U;
    myVSuiv->prev = U;

    U->prev = myVPred;
    U->next = myVSuiv;
    V->prev = myUPred;
    V->next = myUSuiv;

    U->route = myRouteV;
    V->route = myRouteU;
}

void LocalSearch::updateRouteData(Route* myRoute) {
    int myplace = 0;
    int myload = 0;
    int mytime = 0;
    int myReversalDistance = 0;
    int cumulatedX = 0;
    int cumulatedY = 0;
    int myDiscountedCost = 0;
    int mydistance = 0;

    Node* mynode = myRoute->depot;
    mynode->position = 0;
    mynode->cumulatedLoad = 0;
    mynode->cumulatedReversalDistance = 0;
    mynode->cumulatedCost = 0;

    bool firstIt = true;
    TimeWindowData seedTwD;
    Node* seedNode = nullptr;
    while (!mynode->isDepot || firstIt) {
        mynode = mynode->next;
        myplace++;
        mynode->position = myplace;

        int cur_node_id = mynode->cour;
        int prev_node_id = mynode->prev->cour;

        myload += params->cli[cur_node_id].demand;
        mytime += params->getTravelTime(prev_node_id, cur_node_id)
                  + params->cli[cur_node_id].serviceDuration;
        myReversalDistance += params->getCost(cur_node_id, prev_node_id)
                              - params->getCost(prev_node_id, cur_node_id);
        mydistance += params->getCost(prev_node_id, cur_node_id);

        mynode->cumulatedLoad = myload;
        mynode->cumulatedReversalDistance = myReversalDistance;
        mynode->cumulatedCost = mydistance;
        mynode->prefixTwData = MergeTWDataRecursive(mynode->prev->prefixTwData, mynode->twData);
        mynode->isSeed = false;
        mynode->nextSeed = nullptr;
        if (!mynode->isDepot) {
            cumulatedX += params->cli[cur_node_id].coordX;
            cumulatedY += params->cli[cur_node_id].coordY;
            if (firstIt)
                myRoute->sector.initialize(params->cli[cur_node_id].polarAngle);
            else
                myRoute->sector.extend(params->cli[cur_node_id].polarAngle);
            if (myplace % 4 == 0) {
                if (seedNode != nullptr) {
                    seedNode->isSeed = true;
                    seedNode->toNextSeedTwD = MergeTWDataRecursive(seedTwD, mynode->twData);
                    seedNode->nextSeed = mynode;
                }
                seedNode = mynode;
            } else if (myplace % 4 == 1) {
                seedTwD = mynode->twData;
            } else {
                seedTwD = MergeTWDataRecursive(seedTwD, mynode->twData);
            }
        }
        firstIt = false;
    }

    myRoute->duration
        = mytime;  // Driving duration + service duration, excluding waiting time / time warp
    myRoute->load = myload;
    myRoute->twData = mynode->prefixTwData;
    myRoute->penalty = penaltyExcessLoad(myload) + penaltyTimeWindows(myRoute->twData);
    myRoute->nbCustomers = myplace - 1;
    // Remember "when" this route has been last modified (will be used to filter unnecessary move
    // evaluations)
    myRoute->whenLastModified = nbMoves;
    myRoute->isDeltaRemovalTWOutdated = true;

    myRoute->discountedCost = myDiscountedCost;
    myRoute->distance = mydistance;

    // Time window data in reverse direction, mynode should be end depot now
    assert(mynode == &depotsEnd[myRoute->cour]);
    firstIt = true;
    while (!mynode->isDepot || firstIt) {
        mynode = mynode->prev;
        mynode->postfixTwData = MergeTWDataRecursive(mynode->twData, mynode->next->postfixTwData);
        firstIt = false;
    }

    if (myRoute->nbCustomers == 0) {
        myRoute->polarAngleBarycenter = 1.e30;
        emptyRoutes.insert(myRoute->cour);
    } else {
        myRoute->polarAngleBarycenter
            = atan2(cumulatedY / static_cast<double>(myRoute->nbCustomers) - params->cli[0].coordY,
                    cumulatedX / static_cast<double>(myRoute->nbCustomers) - params->cli[0].coordX);
        // Enforce minimum size of circle sector
        if (params->minCircleSectorSize > 0) {
            const int growSectorBy
                = (params->minCircleSectorSize
                   - myRoute->sector.positive_mod(myRoute->sector.end - myRoute->sector.start) + 1)
                  / 2;
            if (growSectorBy > 0) {
                myRoute->sector.extend(myRoute->sector.start - growSectorBy);
                myRoute->sector.extend(myRoute->sector.end + growSectorBy);
            }
        }
        emptyRoutes.erase(myRoute->cour);
    }
}

CostSol LocalSearch::getCostSol(bool usePenaltiesLS) {
    CostSol myCostSol;

    myCostSol.nbRoutes = 0;  // TODO
    for (int r = 0; r < params->nbVehicles; r++) {
        Route* myRoute = &routes[r];
        myCostSol.cost += myRoute->discountedCost;
        myCostSol.distance += myRoute->distance;
        myCostSol.capacityExcess += std::max(0, myRoute->load - params->vehicleCapacity);
        myCostSol.waitTime += 0;  // TODO
        myCostSol.timeWarp += myRoute->twData.timeWarp;
    }

    if (usePenaltiesLS) {
        myCostSol.penalizedCost = myCostSol.cost + myCostSol.capacityExcess * penaltyCapacityLS
                                  + myCostSol.timeWarp * penaltyTimeWarpLS
                                  + myCostSol.waitTime * params->penaltyWaitTime;
    } else {
        myCostSol.penalizedCost = myCostSol.cost
                                  + myCostSol.capacityExcess * params->penaltyCapacity
                                  + myCostSol.timeWarp * params->penaltyTimeWarp
                                  + myCostSol.waitTime * params->penaltyWaitTime;
    }
    return myCostSol;
}

void LocalSearch::setCorrelatedVertices() {
    // Calculation of the correlated vertices for each client (for the granular restriction)
    // Loop over all clients (excluding the depot)
    for (int i = 1; i <= params->nbClients; i++) {
        const auto& orderProximity = params->orderProximities[i];

        auto& currentCorrelatedVertices = correlatedVertices[i];
        currentCorrelatedVertices.resize(orderProximity.size());

        std::transform(orderProximity.begin(), orderProximity.end(),
                       currentCorrelatedVertices.begin(),
                       [](const auto& proximity_pair) { return proximity_pair.second; });
    }

    /*if (total_num_correlated_vertices < static_cast<size_t>(std::max(
            1., static_cast<double>(std::min(params->config.nbGranular,
                                             static_cast<int>(individual.getNumberOfRequiredCustomers())))
                    * static_cast<double>(individual.getNumberOfRequiredCustomers()) / 10.0))) {
        std::cerr << "Warning: Very few correlated vertices! LS performance will suffer"
                  << std::endl;
    }*/
}

void LocalSearch::loadIndividual(Individual* indiv) {
    assert(indiv->isValid());

    emptyRoutes.clear();
    nbMoves = 0;
    TimeWindowData depotTwData;
    depotTwData.firstNodeIndex = 0;
    depotTwData.lastNodeIndex = 0;
    depotTwData.duration = 0;
    depotTwData.timeWarp = 0;
    depotTwData.earliestArrival = params->cli[0].earliestArrival;
    depotTwData.latestArrival = params->cli[0].latestArrival;
    depotTwData.latestReleaseTime = params->cli[0].releaseTime;

    // Initializing time window data (before loop since it is needed in update route)
    for (int i = 1; i <= params->nbClients; i++) {
        TimeWindowData* myTwData = &clients[i].twData;
        myTwData->firstNodeIndex = i;
        myTwData->lastNodeIndex = i;
        myTwData->duration = params->cli[i].serviceDuration;
        myTwData->earliestArrival = params->cli[i].earliestArrival;
        myTwData->latestArrival = params->cli[i].latestArrival;
        myTwData->latestReleaseTime = params->cli[i].releaseTime;
        // myTwData->load = params->cli[i].demand;
    }

    for (int r = 0; r < params->nbVehicles; r++) {
        Node* myDepot = &depots[r];
        Node* myDepotFin = &depotsEnd[r];
        Route* myRoute = &routes[r];
        myDepot->prev = myDepotFin;
        myDepotFin->next = myDepot;
        if (!indiv->chromR[r].empty()) {
            Node* myClient = &clients[indiv->chromR[r][0]];
            myClient->route = myRoute;
            myClient->prev = myDepot;
            myDepot->next = myClient;
            for (int i = 1; i < static_cast<int>(indiv->chromR[r].size()); i++) {
                Node* myClientPred = myClient;
                myClient = &clients[indiv->chromR[r][i]];
                myClient->prev = myClientPred;
                myClientPred->next = myClient;
                myClient->route = myRoute;
            }
            myClient->next = myDepotFin;
            myDepotFin->prev = myClient;
        } else {
            myDepot->next = myDepotFin;
            myDepotFin->prev = myDepot;
        }

        myDepot->twData = depotTwData;
        myDepot->prefixTwData = depotTwData;
        myDepot->postfixTwData = depotTwData;
        myDepot->isSeed = false;

        myDepotFin->twData = depotTwData;
        myDepotFin->prefixTwData = depotTwData;
        myDepotFin->postfixTwData = depotTwData;
        myDepotFin->isSeed = false;

        updateRouteData(&routes[r]);
        routes[r].whenLastTestedLargeNb = -1;
        bestInsertInitializedForRoute[r] = false;
    }

    for (int i = 1; i <= params->nbClients; i++)  // Initializing memory structures
        clients[i].whenLastTestedRI = -1;
}

void LocalSearch::exportIndividual(Individual* indiv) {
    std::vector<std::pair<double, int>> routePolarAngles;
    for (int r = 0; r < params->nbVehicles; r++)
        routePolarAngles.push_back(std::pair<double, int>(routes[r].polarAngleBarycenter, r));
    std::sort(routePolarAngles.begin(),
              routePolarAngles.end());  // empty routes have a polar angle of 1.e30, and therefore
                                        // will always appear at the end

    for (int r = 0; r < params->nbVehicles; r++) {
        indiv->chromR[r].clear();
        Node* node = depots[routePolarAngles[r].second].next;
        while (!node->isDepot) {
            indiv->chromR[r].push_back(node->cour);
            node = node->next;
        }
    }

    indiv->evaluateCompleteCost();

    // Actually should not be required
    assert(indiv->isValid());
}

LocalSearch::LocalSearch(Params* params)
    : params(params),
      clients(std::vector<Node>(params->nbClients + 1)),
      depots(std::vector<Node>(params->nbVehicles)),
      depotsEnd(std::vector<Node>(params->nbVehicles)),
      routes(std::vector<Route>(params->nbVehicles)),
      bestInsertInitializedForRoute(params->nbVehicles, false),
      bestInsertClient(params->nbVehicles, std::vector<ThreeBestInsert>(params->nbClients + 1)),
      bestInsertClientTW(std::vector<std::vector<ThreeBestInsert>>(
          params->nbVehicles, std::vector<ThreeBestInsert>(params->nbClients + 1))),
      correlatedVertices(std::vector<std::vector<int>>(params->nbClients + 1)) {
    setCorrelatedVertices();

    for (int i = 0; i <= params->nbClients; i++) {
        clients[i].cour = i;
        clients[i].isDepot = false;
    }
    for (int i = 0; i < params->nbVehicles; i++) {
        routes[i].cour = i;
        routes[i].depot = &depots[i];
        depots[i].cour = 0;
        depots[i].isDepot = true;
        depots[i].route = &routes[i];
        depotsEnd[i].cour = 0;
        depotsEnd[i].isDepot = true;
        depotsEnd[i].route = &routes[i];
    }
    for (int i = 1; i <= params->nbClients; i++) orderNodes.push_back(i);
    for (int r = 0; r < params->nbVehicles; r++) orderRoutes.push_back(r);
}

std::pair<std::vector<int>, std::vector<int>> LocalSearch::optimizeIncludedCustomers(
    const Individual& individual, double minPerturbationRangeFactor,
    double maxPerturbationRangeFactor,
    const std::vector<unsigned int>& remove_subsequence_lengths) {
    auto get_perturbation_factor
        = [minPerturbationRangeFactor, maxPerturbationRangeFactor, this]() {
              return minPerturbationRangeFactor
                     + (static_cast<double>(params->rng())
                        / static_cast<double>(params->rng.max() - params->rng.min()))
                           * (maxPerturbationRangeFactor - minPerturbationRangeFactor);
          };
    auto perturb
        = [&get_perturbation_factor](auto cost) { return get_perturbation_factor() * cost; };

    // Customers which are not profitable but served anyway
    std::vector<int> unprofitableCustomers
        = find_unprofitable_customers(individual, perturb, remove_subsequence_lengths);
    // Customers which are profitable but not in the current solution
    std::vector<int> profitableCustomers;

    // Determine customers to insert
    auto missing_customers = individual.getUnservedCustomers();
    static_assert(std::is_same_v<decltype(missing_customers)::iterator::iterator_category,
                                 std::random_access_iterator_tag>);
    for (const auto& route : routes) {
        if (route.nbCustomers == 0) continue;
        for (const Node* node = route.depot->next; !node->isDepot; node = node->next) {
            for (auto customer_id = missing_customers.begin();
                 customer_id != missing_customers.end();) {
                if (params->isCertainlyUnprofitable(*customer_id)) {
                    ++customer_id;
                    continue;
                }
                // Test insertion right before node
                if (auto improv = perturb(calculateInsertionImprov(*customer_id, node))
                                  + params->cli[*customer_id].profit;
                    improv > 0) {
                    profitableCustomers.push_back(*customer_id);
                    std::iter_swap(customer_id, std::prev(missing_customers.end()));
                    missing_customers.pop_back();
                    continue;
                }
                ++customer_id;
            }
        }
    }

    return {std::move(unprofitableCustomers), std::move(profitableCustomers)};
}

std::vector<int> LocalSearch::find_unprofitable_customers(
    const Individual& individual, auto perturb,
    const std::vector<unsigned int>& cust_subsequence_lengths) {  // Iterate over all routes

    sul::dynamic_bitset<> unprofitable_customers(params->nbClients + 1);

    for (auto cust_subsequence_length : cust_subsequence_lengths) {
        for (auto& route : routes) {
            if (route.nbCustomers < cust_subsequence_length) continue;
            // TODO We can improve the access pattern here
            RouteIterator route_end = RouteIterator(&depotsEnd[route.cour]);
            RouteIterator seg_beg = RouteIterator(route.depot->next);
            RouteIterator seg_last = std::next(seg_beg, cust_subsequence_length - 1);
            // Seg end is past-the-end iterator
            RouteIterator seg_end = std::next(seg_beg, cust_subsequence_length);
            double seq_profit
                = std::accumulate(seg_beg, seg_end, 0, [this](int prev, const Node& node) {
                      return prev + params->cli[node.cour].profit;
                  });

            for (; seg_last != route_end; seq_profit -= params->cli[seg_beg->cour].profit,
                                          ++seg_end, ++seg_beg, ++seg_last,
                                          seq_profit += params->cli[seg_end->prev->cour].profit) {
                assert(std::accumulate(seg_beg, seg_end, 0,
                                       [this](int prev, const Node& node) {
                                           return prev + params->cli[node.cour].profit;
                                       })
                       == seq_profit);

                if (std::any_of(seg_beg, seg_end, [individual, this](const Node& node) {
                        return params->isCertainlyProfitable(node.cour);
                    })) {
                    continue;
                }
                // Whats the benefit/cost of removing the sequence?
                // Recall that seg_end is past the end iterator!
                if (auto improv
                    = perturb(calculateRemovalImprov(&*seg_beg, &*seg_last)) - seq_profit;
                    improv > 0) {
                    // Mark sequence for removal
                    for (auto node_iter = seg_beg; node_iter != seg_end; ++node_iter) {
                        unprofitable_customers.set(node_iter->cour);
                    }
                }
            }
        }
    }

    return bitset_to_vector(unprofitable_customers);
}

int LocalSearch::calculateRemovalImprov(const Node* seq_beg, const Node* seq_end) const {
    const Route* route = seq_beg->route;
    assert(route == seq_end->route);
    assert(!seq_beg->isDepot && !seq_end->isDepot);

    int prev_cost = route->distance + route->penalty;

    int load_saving = seq_end->cumulatedLoad - seq_beg->prev->cumulatedLoad;
    int distance_saving = (seq_end->next->cumulatedCost - seq_beg->prev->cumulatedCost)
                          - params->getCost(seq_beg->prev->cour, seq_end->next->cour);

    int new_cost = route->distance - distance_saving + penaltyExcessLoad(route->load - load_saving);

    auto removed_tw_data
        = MergeTWDataRecursive(seq_beg->prev->prefixTwData, seq_end->next->postfixTwData);
    // Compute new total penalty
    new_cost += penaltyTimeWindows(removed_tw_data);
    return prev_cost - new_cost;
}

int LocalSearch::calculateRemovalImprov(const Node* node) const {
    return calculateRemovalImprov(node, node);
}

int LocalSearch::calculateInsertionImprov(const Node* prev, int customerIndex,
                                          const Node* next) const {
    const Route* route = prev->route;

    int prev_cost = params->getCost(prev->cour, next->cour) + route->penalty;
    int new_cost
        = params->getCost(prev->cour, customerIndex) + params->getCost(customerIndex, next->cour);

    auto route_tw_data = MergeTWDataRecursive(prev->prefixTwData, clients[customerIndex].twData,
                                              next->postfixTwData);

    new_cost += penaltyExcessLoad(route->load + params->cli[customerIndex].demand)
                + penaltyTimeWindows(route_tw_data);

    return (prev_cost - new_cost);
}

int LocalSearch::calculateInsertionImprov(int customerIndex, const Node* at) const {
    return calculateInsertionImprov(at->prev, customerIndex, at);
}
unsigned int LocalSearch::MaxNumberOfCorrelatedVertices() const {
    return params->config.nbGranular;
}

bool Node::operator==(const Node& rhs) const {
    return isDepot == rhs.isDepot && cour == rhs.cour && position == rhs.position
           && whenLastTestedRI == rhs.whenLastTestedRI && next == rhs.next && prev == rhs.prev
           && route == rhs.route && cumulatedLoad == rhs.cumulatedLoad
           && cumulatedReversalDistance == rhs.cumulatedReversalDistance
           && cumulatedCost == rhs.cumulatedCost && deltaRemoval == rhs.deltaRemoval
           && deltaRemovalTW == rhs.deltaRemovalTW && twData == rhs.twData
           && prefixTwData == rhs.prefixTwData && postfixTwData == rhs.postfixTwData
           && isSeed == rhs.isSeed && toNextSeedTwD == rhs.toNextSeedTwD
           && nextSeed == rhs.nextSeed;
}

bool Node::operator!=(const Node& rhs) const { return !(rhs == *this); }

std::ostream& operator<<(std::ostream& os, const Route& route) {
    os << "0, ";
    Node* next = route.depot->next;
    while (!next->isDepot) {
        os << next->cour << ", ";
        next = next->next;
    }
    os << "0";
    return os;
}
