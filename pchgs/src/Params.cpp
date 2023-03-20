#include "Params.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

#include "CircleSector.h"
#include "Matrix.h"
#include "commandline.h"
#include "xorshift128.h"

Client parse_client(const nlohmann::json &json, int id) {
    return {id,
            json["coords"][0].get<int>(),
            json["coords"][1].get<int>(),
            json["service_time"].get<int>(),
            json["demand"].get<int>(),
            json["time_window"][0].get<int>(),
            json["time_window"][1].get<int>(),
            json.value("release_time", 0),
            json["profit"].get<int>(),
            json["request_idx"].get<int>()};
}

std::vector<Client> parse_customers_from_JSON(const nlohmann::json &json) {
    std::vector<Client> clients(0);

    int client_id = 0;
    for (const auto &node: json["nodes"]) {
        clients.emplace_back(parse_client(node, client_id));
        client_id++;
    }

    return clients;
}

namespace {
    Matrix _parse_matrix(const nlohmann::json &raw_matrix, int dimension) {
        Matrix matrix(dimension);
        if (raw_matrix.size() != dimension) {
            throw std::runtime_error(fmt::format(
                    "Matrix has wrong format. Expected {} rows, got {}", dimension, raw_matrix.size()));
        }
        for (int i = 0; i < raw_matrix.size(); ++i) {
            if (raw_matrix[i].size() != dimension) {
                throw std::runtime_error(fmt::format(
                        "Duration matrix has wrong format. Expected {} columns, got {} (row index {})",
                        dimension, raw_matrix.size(), i));
            }
            for (int j = 0; j < raw_matrix[i].size(); ++j) {
                int val = raw_matrix[i][j];
                matrix.set(i, j, val);
            }
        }
        return matrix;
    }
}  // namespace

void Params::_parse_json(std::istream &instance_input) {
    auto instance_json = nlohmann::json::parse(instance_input);
    // Parse capacity
    vehicleCapacity = instance_json["capacity"];
    // Parse customers
    cli = parse_customers_from_JSON(instance_json);
    // nbClients excludes depot
    nbClients = cli.size() - 1;

    // Parse distances
    _timeCost = _parse_matrix(instance_json["durations"], cli.size());
    if (instance_json.contains("cost")) {
        _distCost = _parse_matrix(instance_json["cost"], cli.size());
    } else {
        _distCost = _timeCost;
    }
}

void Params::_parse_vrplib(std::istream &inputFile) {
    // Initialize some parameter values
    int serviceTimeData = 0;
    int node;
    bool hasServiceTimeSection = false;
    bool hasDistCost = false;

    // Read INPUT dataset
    std::string content, content2, content3;
    // Read the instance name from the first line and remove \r
    getline(inputFile, content);
    instanceName = content;
    instanceName.erase(std::remove(instanceName.begin(), instanceName.end(), '\r'),
                       instanceName.end());

    // Read the next lines
    getline(inputFile, content);  // "Empty line" or "NAME : {instance_name}"
    getline(inputFile, content);  // VEHICLE or "COMMENT: {}"

    // Check if the next line has "VEHICLE"
    // CVRP or VRPTW according to VRPLib format
    for (inputFile >> content; content != "EOF"; inputFile >> content) {
        // Read the dimension of the problem (the number of clients)
        if (content == "DIMENSION") {
            // Need to substract the depot from the number of nodes
            inputFile >> content2 >> nbClients;
            nbClients--;
        }
            // Read the type of edge weights
        else if (content == "EDGE_WEIGHT_TYPE") {
            inputFile >> content2 >> content3;
            if (content3 != "EXPLICIT") {
                throw std::logic_error(
                        "Implicit (computed) distance matrix is not implemented!");
            }
        } else if (content == "EDGE_WEIGHT_FORMAT") {
            inputFile >> content2 >> content3;
            if (content3 != "FULL_MATRIX") {
                throw std::string("EDGE_WEIGHT_FORMAT only supports FULL_MATRIX");
            }
        } else if (content == "CAPACITY") {
            inputFile >> content2 >> vehicleCapacity;
        } else if (content == "VEHICLES" || content == "SALESMAN") {
            inputFile >> content2 >> nbVehicles;
        } else if (content == "DISTANCE") {
            inputFile >> content2 >> durationLimit;
            isDurationConstraint = true;
        }
            // Read the data on the service time (used when the service time is constant for all
            // clients)
        else if (content == "SERVICE_TIME") {
            inputFile >> content2 >> serviceTimeData;
        }
            // Read the edge weights of an explicit distance matrix
        else if (content == "EDGE_WEIGHT_SECTION") {
            _timeCost = Matrix(nbClients + 1);

            for (int i = 0; i <= nbClients; i++) {
                for (int j = 0; j <= nbClients; j++) {
                    // Keep track of the largest distance between two clients (or the depot)
                    int cost;
                    inputFile >> cost;
                    _timeCost.set(i, j, cost);
                }
            }
        }
            // Read the cost matrix
        else if (content == "EDGE_COST_SECTION") {
            hasDistCost = true;
            _distCost = Matrix(nbClients + 1);
            for (int i = 0; i <= nbClients; i++) {
                for (int j = 0; j <= nbClients; j++) {
                    // Keep track of the largest distance between two clients (or the depot)
                    int cost;
                    inputFile >> cost;
                    _distCost.set(i, j, cost);
                }
            }
        } else if (content == "NODE_COORD_SECTION") {
            // Reading client coordinates
            cli = std::vector<Client>(nbClients + 1);
            for (int i = 0; i <= nbClients; i++) {
                inputFile >> cli[i].custNum >> cli[i].coordX >> cli[i].coordY;

                // Check if the clients are in order
                if (cli[i].custNum != i + 1) {
                    throw std::string("Clients are not in order in the list of coordinates");
                }

                cli[i].custNum--;
            }
        }
            // Read the demand of each client (including the depot, which should have demand 0)
        else if (content == "DEMAND_SECTION") {
            for (int i = 0; i <= nbClients; i++) {
                int clientNr = 0;
                inputFile >> clientNr >> cli[i].demand;

                // Check if the clients are in order
                if (clientNr != i + 1) {
                    throw std::string("Clients are not in order in the list of demands");
                }
            }
        } else if (content == "DEPOT_SECTION") {
            inputFile >> content2 >> content3;
            if (content2 != "1") {
                throw std::string("Expected depot index 1 instead of " + content2);
            }
        } else if (content == "SERVICE_TIME_SECTION") {
            for (int i = 0; i <= nbClients; i++) {
                int clientNr = 0;
                inputFile >> clientNr >> cli[i].serviceDuration;

                // Check if the clients are in order
                if (clientNr != i + 1) {
                    throw std::string("Clients are not in order in the list of service times");
                }
            }
            hasServiceTimeSection = true;
        } else if (content == "RELEASE_TIME_SECTION") {
            for (int i = 0; i <= nbClients; i++) {
                int clientNr = 0;
                inputFile >> clientNr >> cli[i].releaseTime;

                // Check if the clients are in order
                if (clientNr != i + 1) {
                    throw std::string("Clients are not in order in the list of release times");
                }
            }
        }
            // Read the time windows of all the clients (the depot should have a time window
            // from 0 to max)
        else if (content == "TIME_WINDOW_SECTION") {
            isTimeWindowConstraint = true;
            for (int i = 0; i <= nbClients; i++) {
                int clientNr = 0;
                inputFile >> clientNr >> cli[i].earliestArrival >> cli[i].latestArrival;

                // Check if the clients are in order
                if (clientNr != i + 1) {
                    throw std::string("Clients are not in order in the list of time windows");
                }
            }
        } else if (content == "PROFIT_SECTION") {
            for (int i = 0; i <= nbClients; i++) {
                int clientNr = 0;
                inputFile >> clientNr >> cli[i].profit;

                // Check if the clients are in order
                if (clientNr != i + 1) {
                    throw std::string("Clients are not in order in the list of profits");
                }
            }
        } else {
            throw std::string("Unexpected data in input file: " + content);
        }
    }

    if (!hasServiceTimeSection) {
        for (int i = 0; i <= nbClients; i++) {
            cli[i].serviceDuration = (i == 0) ? 0 : serviceTimeData;
        }
    }

    if (!hasDistCost) {
        _distCost = _timeCost;
    }
}

void Params::_validate() const {
    const auto &depot = cli[0];

    if (nbClients < 0) {
        throw std::string("Number of nodes is undefined");
    }
    if (vehicleCapacity == INT_MAX) {
        throw std::string("Vehicle capacity is undefined");
    }
    // Check if the service duration of the depot is 0
    if (depot.releaseTime != 0) {
        throw std::string("Release time for depot should be 0");
    }
    // Check if the service duration of the depot is 0
    if (depot.serviceDuration != 0) {
        throw std::string("Service duration for depot should be 0");
    }
    // Check the start of the time window of the depot
    if (depot.earliestArrival != 0) {
        throw std::string("Time window for depot should start at 0");
    }
    // Check if the depot has proft 0
    if (depot.profit != 0) {
        throw std::string("Depot profit is not zero, but is instead: "
                          + std::to_string(depot.profit));
    }
    // Check if the depot has demand 0
    if (depot.demand != 0) {
        throw std::string("Depot demand is not zero, but is instead: "
                          + std::to_string(depot.serviceDuration));
    }

    // Check if each customer finishes before the depot
    for (const auto &client: cli) {
        if (client.latestArrival > depot.latestArrival) {
            throw fmt::format("Depot TW ends before client {} ends.", client.requestIdx);
        }
    }

    // Distance to self should be zero
    for (int i = 0; i <= nbClients; ++i) {
        if (_timeCost.get(i, i) != 0 || _distCost.get(i, i) != 0) {
            throw fmt::format("Customer {} has non-zero distance/duration to self: {}/{}", i,
                              _timeCost.get(i, i), _distCost.get(i, i));
        }
    }

    // Safeguards to avoid possible numerical instability in case of instances containing
    // arbitrarily small or large numerical values
    if (maxDist > 100000) {
        throw std::string(
                "The distances are of very small or large scale. This could impact numerical "
                "stability. "
                "Please rescale the dataset and run again.");
    }
    if (maxDemand > 100000) {
        throw std::string(
                "The demand quantities are of very small or large scale. This could impact numerical "
                "stability. Please rescale the dataset and run again.");
    }
    if (nbVehicles < std::ceil(totalDemand / vehicleCapacity)) {
        throw std::string("Fleet size is insufficient to service the considered clients.");
    }
    if (config.useSymmetricCorrelatedVertices) {
        throw std::string("Symmetric correlation is not supported!");
    }
}

Params::Params(const CommandLine &cl)
        : isDurationConstraint(false),
          durationLimit(INT_MAX),
          vehicleCapacity(INT_MAX),
          totalDemand(0),
          maxDemand(0),
          maxDist(0) {
    // Read and create some parameter values from the commandline
    config = cl.config;
    nbVehicles = config.nbVeh;
    rng = XorShift128(config.seed);
    startWallClockTime = std::chrono::system_clock::now();
    startCPUTime = std::clock();

    // Convert the circle sector parameters from degrees ([0,359]) to [0,65535] to allow for faster
    // calculations
    circleSectorOverlapTolerance
            = static_cast<int>(config.circleSectorOverlapToleranceDegrees / 360. * 65536);
    minCircleSectorSize = static_cast<int>(config.minCircleSectorSizeDegrees / 360. * 65536);

    if (config.pathInstance == "-") {
        _parse(std::cin);
    } else {
        std::ifstream inputFile(config.pathInstance);
        if (!inputFile.is_open()) {
            throw std::invalid_argument("Impossible to open instance file: " + config.pathInstance);
        }
        _parse(inputFile);
    }

    _prepare();
    _validate();

    _preprocess_included_customers();
    _compute_proximities();
}

void Params::_parse(std::istream &inputStream) {
    if (config.pathInstance.ends_with(".json") || config.assumeJSONInput) {
        _parse_json(inputStream);
    } else {
        _parse_vrplib(inputStream);
    }
}

void Params::_prepare() {
    totalDemand = std::accumulate(cli.begin(), cli.end(), 0,
                                  [](int sum, const auto &client) { return sum + client.demand; });
    maxDemand = std::max_element(cli.begin(), cli.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.demand < rhs.demand;
    })->demand;

    // calculate max distance
    for (auto i = 0u; i < cli.size(); ++i) {
        for (auto j = i + 1; j < cli.size(); ++j) {
            // Distances may be asymmetric
            maxDist = std::max({_timeCost.get(i, j), _timeCost.get(j, i), maxDist});
        }
    }

    // Calculate polar angles
    for (auto &client: cli) {
        client.polarAngle = CircleSector::positive_mod(static_cast<int>(
                                                               32768. * atan2(client.coordY - cli[0].coordY,
                                                                              client.coordX - cli[0].coordX) / PI));
    }

    // Fill request_idx map
    for (auto &client: cli) {
        auto [iter, inserted]
                = _request_idx_to_customer_idx.insert({client.requestIdx, client.custNum});
        if (!inserted) {
            throw std::runtime_error("Request IDX is not unique!");
        }
    }

    // Default initialization if the number of vehicles has not been provided by the user
    if (nbVehicles == INT_MAX) {
        // Safety margin: 30% + 3 more vehicles than the trivial bin packing LB
        nbVehicles = static_cast<int>(ceil(1.3 * totalDemand / vehicleCapacity) + 3.);
        if (!config.quiet) {
            std::cout << "----- FLEET SIZE WAS NOT SPECIFIED: DEFAULT INITIALIZATION TO "
                      << nbVehicles << " VEHICLES" << std::endl;
        }
    } else if (nbVehicles == -1) {
        nbVehicles = nbClients;
        if (!config.quiet) {
            std::cout << "----- FLEET SIZE UNLIMITED: SET TO UPPER BOUND OF " << nbVehicles
                      << " VEHICLES" << std::endl;
        }
    } else {
        if (!config.quiet) {
            std::cout << "----- FLEET SIZE SPECIFIED IN THE COMMANDLINE: SET TO " << nbVehicles
                      << " VEHICLES" << std::endl;
        }
    }

    // A reasonable scale for the initial values of the penalties
    penaltyCapacity = std::max(0.1, std::min(1000., static_cast<double>(maxDist) / maxDemand));

    // Initial parameter values of these two parameters are not argued
    penaltyWaitTime = 0.;
    penaltyTimeWarp = config.initialTimeWarpPenalty;

    // See Vidal 2012, HGS for VRPTW
    proximityWeightWaitTime = 0.2;
    proximityWeightTimeWarp = 1;

    _certainly_unprofitable_customers = sul::dynamic_bitset<>(nbClients + 1);
    _certainly_profitable_customers = sul::dynamic_bitset<>(nbClients + 1);
}

void Params::_compute_proximities() {  // Compute order proximities once
    orderProximities = std::vector<std::vector<std::pair<double, int>>>(nbClients + 1);
    // Loop over all clients (excluding the depot)
    for (int i = 1; i <= nbClients; i++) {
        // Remove all elements from the vector
        auto &orderProximity = orderProximities[i];
        orderProximity.clear();

        // Loop over all clients (excluding the depot and the specific client itself)
        for (int j = 1; j <= nbClients; j++) {
            if (i != j) {
                // Compute proximity using Eq. 4 in Vidal 2012, and append at the end of
                // orderProximity
                const int timeIJ = getTravelTime(i, j);
                orderProximity.emplace_back(
                        getCost(i, j)
                        + std::min(
                                proximityWeightWaitTime
                                * std::max(cli[j].earliestArrival - timeIJ
                                           - cli[i].serviceDuration - cli[i].latestArrival,
                                           0)
                                + proximityWeightTimeWarp
                                  * std::max(cli[i].earliestArrival + cli[i].serviceDuration
                                             + timeIJ - cli[j].latestArrival,
                                             0),
                                proximityWeightWaitTime
                                * std::max(cli[i].earliestArrival - timeIJ
                                           - cli[j].serviceDuration - cli[j].latestArrival,
                                           0)
                                + proximityWeightTimeWarp
                                  * std::max(cli[j].earliestArrival + cli[j].serviceDuration
                                             + timeIJ - cli[i].latestArrival,
                                             0)),
                        j);
            }
        }

        // Sort orderProximity (for the specific client)
        std::sort(orderProximity.begin(), orderProximity.end());
    }
}

double Params::getTimeElapsedSeconds() {
    if (config.useWallClockTime) {
        std::chrono::duration<double> wctduration
                = (std::chrono::system_clock::now() - startWallClockTime);
        return wctduration.count();
    }
    return (std::clock() - startCPUTime) / (double) CLOCKS_PER_SEC;
}

bool Params::isTimeLimitExceeded() { return getTimeElapsedSeconds() >= config.timeLimit; }

int Params::getTravelTime(const int row, const int col) const { return _timeCost.get(row, col); }

int Params::getCost(const int row, const int col) const { return _distCost.get(row, col); }

int Params::getDiscountedCost(const int row, const int col) const {
    return getCost(row, col) - cli[col].profit;
}

void Params::_preprocess_included_customers() {
    _certainly_profitable_customers.set(0);
    const auto max_dist_overall = maxDist;
    for (int c = 1; c <= nbClients; ++c) {
        // Asymmetric
        int min_from_dist, min_to_dist;
        min_from_dist = min_to_dist = std::numeric_limits<int>::max();
        int max_from_dist, max_to_dist;
        max_from_dist = max_to_dist = std::numeric_limits<int>::min();
        for (int i = 0; i <= nbClients; ++i) {
            min_from_dist = std::min(min_from_dist, getCost(i, c));
            min_to_dist = std::min(min_to_dist, getCost(c, i));
            max_from_dist = std::max(max_from_dist, getCost(i, c));
            max_to_dist = std::max(max_to_dist, getCost(c, i));
        }

        if (min_from_dist + min_to_dist - cli[c].profit >= max_dist_overall) {
            _certainly_unprofitable_customers.set(c);
        }

        if (max_from_dist + max_to_dist < cli[c].profit) {
            _certainly_profitable_customers.set(c);
        }
    }
}

bool Params::isCertainlyProfitable(int cust_id) const {
    return _certainly_profitable_customers.test(cust_id);
}

bool Params::isCertainlyUnprofitable(int cust_id) const {
    return _certainly_unprofitable_customers.test(cust_id);
}

int Params::getNumberOfCertainlyProfitableCustomers() const {
    return _certainly_profitable_customers.count();
}

int Params::getNumberOfCertainlyUnprofitableCustomers() const {
    return _certainly_unprofitable_customers.count();
}

int Params::getNumberOfPossibleClients() {
    return nbClients - getNumberOfCertainlyUnprofitableCustomers();
}

int Params::getClientIDXByRequestIDX(int request_idx) const {
    return _request_idx_to_customer_idx.at(request_idx);
}
