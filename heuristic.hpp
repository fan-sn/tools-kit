#pragma once
#include "llm_math_toolkit.hpp"
#include <vector>
#include <functional>

namespace heur {

llm::Vector simulated_annealing(const std::function<double(const llm::Vector&)>& f, llm::Vector init_state, double T0 = 1.0, double cooling = 0.95, int max_iter = 1000);

llm::Vector tabu_search(const std::function<double(const llm::Vector&)>& f, llm::Vector init_state, int tabu_len = 100, int max_iter = 1000);

llm::Vector greedy_algorithm(const std::function<double(const llm::Vector&)>& f, llm::Vector init_state);

} // namespace heur
