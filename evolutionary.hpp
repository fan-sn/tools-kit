#pragma once
#include "llm_math_toolkit.hpp"
#include <vector>
#include <functional>
#include <random>

namespace evo {

struct GAParams {
    size_t population_size = 50;
    double crossover_rate = 0.7;
    double mutation_rate = 0.01;
    int generations = 100;
};

class GeneticAlgorithm {
public:
    using Individual = llm::Vector;
    using FitnessFunc = std::function<double(const Individual&)>;
    GeneticAlgorithm(const GAParams& params, FitnessFunc fit_fn);
    void run();
    Individual best_individual() const;
private:
    std::vector<Individual> population_;
    FitnessFunc fitness_;
    GAParams params_;
};

class ParticleSwarmOptimization {
public:
    ParticleSwarmOptimization(size_t dim, size_t swarm_size = 30, double w = 0.5, double c1 = 1.0, double c2 = 2.0, int max_iter = 100);
    void optimize(const std::function<double(const llm::Vector&)>& fitness_fn);
    llm::Vector best_position() const;
private:
    // 粒子结构体等
};

class AntColonyOptimization {
public:
    AntColonyOptimization(int n_ants, double alpha = 1.0, double beta = 2.0, double rho = 0.5, int max_iter = 100);
    void optimize(const llm::Matrix& graph);
    std::vector<int> best_path() const;
    double best_length() const;
private:
    // 信息素矩阵等
};

class ArtificialFishSwarm {
public:
    ArtificialFishSwarm(size_t fish_count, size_t dim, int max_iter = 100);
    void optimize(const std::function<double(const llm::Vector&)>& fitness_fn);
    llm::Vector best_position() const;
private:
    // ...
};

class ArtificialBeeColony {
public:
    ArtificialBeeColony(size_t bee_count, size_t dim, int max_iter = 100);
    void optimize(const std::function<double(const llm::Vector&)>& fitness_fn);
    llm::Vector best_position() const;
private:
    // ...
};

} // namespace evo
