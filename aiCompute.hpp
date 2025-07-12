//```cpp name=aiCompute.h
#ifndef aiCompute_H
#define aiCompute_H

#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <cassert>
#include <iostream>

// 智能化工具集（aiCompute）
class aiCompute {
public:
    // ==== 通用工具函数 ====
    static double randomDouble(double min, double max) {
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> dist(min, max);
        return dist(rng);
    }

    static int randomInt(int min, int max) {
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng);
    }

    static std::vector<double> linspace(double start, double end, int num) {
        std::vector<double> result(num);
        if (num == 1) {
            result[0] = start;
            return result;
        }
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) {
            result[i] = start + step * i;
        }
        return result;
    }

    static double mean(const std::vector<double>& data) {
        double sum = 0;
        for (auto v : data) sum += v;
        return data.empty() ? 0.0 : sum / data.size();
    }

    static double stddev(const std::vector<double>& data) {
        double m = mean(data);
        double accum = 0.0;
        for (auto v : data) accum += (v - m) * (v - m);
        return data.size() > 1 ? std::sqrt(accum / (data.size() - 1)) : 0.0;
    }

    // ==== 粒子群优化算法（PSO） ====
    struct Particle {
        std::vector<double> position;
        std::vector<double> velocity;
        std::vector<double> best_position;
        double best_value;

        Particle(int dim)
            : position(dim, 0.0), velocity(dim, 0.0),
              best_position(dim, 0.0), best_value(std::numeric_limits<double>::max()) {}
    };

    static std::vector<double> particleSwarmOptimize(
        std::function<double(const std::vector<double>&)> func,
        int dim, int num_particles, int max_iter,
        double min_bound, double max_bound,
        double w = 0.5, double c1 = 1.5, double c2 = 1.5)
    {
        std::vector<Particle> swarm;
        std::vector<double> gbest_position(dim, 0.0);
        double gbest_value = std::numeric_limits<double>::max();

        // 初始化粒子
        for (int i = 0; i < num_particles; ++i) {
            Particle p(dim);
            for (int d = 0; d < dim; ++d) {
                p.position[d] = randomDouble(min_bound, max_bound);
                p.velocity[d] = randomDouble(-fabs(max_bound - min_bound), fabs(max_bound - min_bound));
            }
            p.best_position = p.position;
            p.best_value = func(p.position);

            if (p.best_value < gbest_value) {
                gbest_value = p.best_value;
                gbest_position = p.position;
            }
            swarm.push_back(p);
        }

        // 主循环
        for (int iter = 0; iter < max_iter; ++iter) {
            for (auto& p : swarm) {
                for (int d = 0; d < dim; ++d) {
                    double r1 = randomDouble(0, 1);
                    double r2 = randomDouble(0, 1);
                    p.velocity[d] = w * p.velocity[d]
                        + c1 * r1 * (p.best_position[d] - p.position[d])
                        + c2 * r2 * (gbest_position[d] - p.position[d]);
                    p.position[d] += p.velocity[d];
                    // 边界处理
                    if (p.position[d] < min_bound) p.position[d] = min_bound;
                    if (p.position[d] > max_bound) p.position[d] = max_bound;
                }
                double value = func(p.position);
                if (value < p.best_value) {
                    p.best_value = value;
                    p.best_position = p.position;
                }
                if (value < gbest_value) {
                    gbest_value = value;
                    gbest_position = p.position;
                }
            }
        }
        return gbest_position;
    }

    // ==== 人工鱼群算法（AFSA） ====
    struct ArtificialFish {
        std::vector<double> position;
        double fitness;

        ArtificialFish(int dim)
            : position(dim, 0.0), fitness(std::numeric_limits<double>::max()) {}
    };

    static std::vector<double> artificialFishSwarmOptimize(
        std::function<double(const std::vector<double>&)> func,
        int dim, int num_fish, int max_iter,
        double min_bound, double max_bound,
        double visual = 1.0, double step = 0.5, int try_number = 5, double crowd_factor = 0.618)
    {
        std::vector<ArtificialFish> swarm;
        std::vector<double> best_position(dim, 0.0);
        double best_fitness = std::numeric_limits<double>::max();

        // 初始化鱼群
        for (int i = 0; i < num_fish; ++i) {
            ArtificialFish fish(dim);
            for (int d = 0; d < dim; ++d) {
                fish.position[d] = randomDouble(min_bound, max_bound);
            }
            fish.fitness = func(fish.position);
            if (fish.fitness < best_fitness) {
                best_fitness = fish.fitness;
                best_position = fish.position;
            }
            swarm.push_back(fish);
        }

        // 主循环
        for (int iter = 0; iter < max_iter; ++iter) {
            for (auto& fish : swarm) {
                // 觅食行为
                ArtificialFish candidate = fish;
                for (int t = 0; t < try_number; ++t) {
                    std::vector<double> new_pos = fish.position;
                    for (int d = 0; d < dim; ++d) {
                        double delta = randomDouble(-visual, visual);
                        new_pos[d] += delta;
                        if (new_pos[d] < min_bound) new_pos[d] = min_bound;
                        if (new_pos[d] > max_bound) new_pos[d] = max_bound;
                    }
                    double new_fit = func(new_pos);
                    if (new_fit < candidate.fitness) {
                        candidate.position = new_pos;
                        candidate.fitness = new_fit;
                    }
                }
                // 更新位置
                if (candidate.fitness < fish.fitness) {
                    for (int d = 0; d < dim; ++d) {
                        fish.position[d] += step * (candidate.position[d] - fish.position[d]);
                        if (fish.position[d] < min_bound) fish.position[d] = min_bound;
                        if (fish.position[d] > max_bound) fish.position[d] = max_bound;
                    }
                    fish.fitness = func(fish.position);
                }
                // 更新全局最优
                if (fish.fitness < best_fitness) {
                    best_fitness = fish.fitness;
                    best_position = fish.position;
                }
            }
        }
        return best_position;
    }

    // ==== 遗传算法（GA） ====
    struct GAIndividual {
        std::vector<double> chromosome;
        double fitness;

        GAIndividual(int dim)
            : chromosome(dim, 0.0), fitness(std::numeric_limits<double>::max()) {}
    };

    static std::vector<double> geneticAlgorithmOptimize(
        std::function<double(const std::vector<double>&)> func,
        int dim, int pop_size, int max_iter,
        double min_bound, double max_bound,
        double crossover_rate = 0.8, double mutation_rate = 0.1)
    {
        std::vector<GAIndividual> population;
        std::vector<double> best_chromosome(dim, 0.0);
        double best_fitness = std::numeric_limits<double>::max();

        // 初始化种群
        for (int i = 0; i < pop_size; ++i) {
            GAIndividual ind(dim);
            for (int d = 0; d < dim; ++d) {
                ind.chromosome[d] = randomDouble(min_bound, max_bound);
            }
            ind.fitness = func(ind.chromosome);
            if (ind.fitness < best_fitness) {
                best_fitness = ind.fitness;
                best_chromosome = ind.chromosome;
            }
            population.push_back(ind);
        }

        // 主循环
        for (int iter = 0; iter < max_iter; ++iter) {
            // 选择（锦标赛）
            std::vector<GAIndividual> new_population;
            while (new_population.size() < population.size()) {
                GAIndividual &parent1 = tournamentSelect(population, 3);
                GAIndividual &parent2 = tournamentSelect(population, 3);

                GAIndividual child1 = parent1, child2 = parent2;

                // 交叉
                if (randomDouble(0, 1) < crossover_rate) {
                    int cross_point = randomInt(1, dim - 1);
                    for (int d = cross_point; d < dim; ++d) {
                        std::swap(child1.chromosome[d], child2.chromosome[d]);
                    }
                }
                // 变异
                for (int d = 0; d < dim; ++d) {
                    if (randomDouble(0, 1) < mutation_rate) {
                        child1.chromosome[d] = randomDouble(min_bound, max_bound);
                    }
                    if (randomDouble(0, 1) < mutation_rate) {
                        child2.chromosome[d] = randomDouble(min_bound, max_bound);
                    }
                }
                child1.fitness = func(child1.chromosome);
                child2.fitness = func(child2.chromosome);
                if (child1.fitness < best_fitness) {
                    best_fitness = child1.fitness;
                    best_chromosome = child1.chromosome;
                }
                if (child2.fitness < best_fitness) {
                    best_fitness = child2.fitness;
                    best_chromosome = child2.chromosome;
                }
                new_population.push_back(child1);
                if (new_population.size() < population.size()) {
                    new_population.push_back(child2);
                }
            }
            population = std::move(new_population);
        }
        return best_chromosome;
    }

private:
    static GAIndividual& tournamentSelect(std::vector<GAIndividual>& pop, int k) {
        int n = pop.size();
        int best = randomInt(0, n - 1);
        for (int i = 1; i < k; ++i) {
            int idx = randomInt(0, n - 1);
            if (pop[idx].fitness < pop[best].fitness) {
                best = idx;
            }
        }
        return pop[best];
    }

public:
    // ==== 蚁群算法（ACO），仅适合TSP等组合优化问题，示例TSP实现 ====
    // 距离矩阵必须为对称矩阵且无负边
    static std::vector<int> antColonyOptimizeTSP(
        const std::vector<std::vector<double>>& distance,
        int num_ants, int max_iter, double alpha = 1.0, double beta = 5.0, double rho = 0.5, double Q = 100.0)
    {
        int n = distance.size();
        std::vector<std::vector<double>> pheromone(n, std::vector<double>(n, 1.0));
        std::vector<int> best_path;
        double best_length = std::numeric_limits<double>::max();

        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<std::vector<int>> paths(num_ants, std::vector<int>(n));
            std::vector<double> lengths(num_ants, 0.0);

            for (int k = 0; k < num_ants; ++k) {
                std::vector<bool> visited(n, false);
                int current = randomInt(0, n - 1);
                paths[k][0] = current;
                visited[current] = true;
                for (int step = 1; step < n; ++step) {
                    std::vector<double> prob(n, 0.0);
                    double sum_prob = 0.0;
                    for (int j = 0; j < n; ++j) {
                        if (!visited[j]) {
                            prob[j] = std::pow(pheromone[current][j], alpha) * std::pow(1.0 / distance[current][j], beta);
                            sum_prob += prob[j];
                        }
                    }
                    double r = randomDouble(0, sum_prob);
                    double acc = 0.0;
                    int next = -1;
                    for (int j = 0; j < n; ++j) {
                        if (!visited[j]) {
                            acc += prob[j];
                            if (acc >= r) {
                                next = j;
                                break;
                            }
                        }
                    }
                    if (next == -1) { // fallback
                        for (int j = 0; j < n; ++j) {
                            if (!visited[j]) {
                                next = j;
                                break;
                            }
                        }
                    }
                    paths[k][step] = next;
                    visited[next] = true;
                    lengths[k] += distance[current][next];
                    current = next;
                }
                lengths[k] += distance[current][paths[k][0]];
                if (lengths[k] < best_length) {
                    best_length = lengths[k];
                    best_path = paths[k];
                }
            }
            // 信息素更新
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    pheromone[i][j] *= (1 - rho);
            for (int k = 0; k < num_ants; ++k) {
                for (int step = 0; step < n - 1; ++step) {
                    int from = paths[k][step];
                    int to = paths[k][step + 1];
                    pheromone[from][to] += Q / lengths[k];
                    pheromone[to][from] += Q / lengths[k];
                }
                pheromone[paths[k][n - 1]][paths[k][0]] += Q / lengths[k];
                pheromone[paths[k][0]][paths[k][n - 1]] += Q / lengths[k];
            }
        }
        return best_path;
    }

    // ==== 向量加减乘除 ====
    static std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        std::vector<double> res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
        return res;
    }
    static std::vector<double> sub(const std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        std::vector<double> res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] - b[i];
        return res;
    }
    static std::vector<double> mul(const std::vector<double>& a, double k) {
        std::vector<double> res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] * k;
        return res;
    }
    static std::vector<double> div(const std::vector<double>& a, double k) {
        assert(k != 0);
        std::vector<double> res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] / k;
        return res;
    }
};

#endif // aiCompute_H
