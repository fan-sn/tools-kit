#pragma once
#include "llm_math_toolkit.hpp"
#include <vector>

namespace other {

/** 支持向量机(SVM, 见ml::SVM) **/

/** Q-Learning */
class QLearning {
public:
    QLearning(size_t n_states, size_t n_actions, double alpha, double gamma, double epsilon);
    void train(int episodes, int max_steps = 100);
    int select_action(int state) const;
    void update(int state, int action, double reward, int next_state);
    llm::Matrix get_Q() const;
private:
    size_t n_states_, n_actions_;
    double alpha_, gamma_, epsilon_;
    llm::Matrix Q_;
};

/** 深度Q网络 DQN */
class DQN {
public:
    DQN(size_t state_dim, size_t action_dim, int hidden_dim = 64, double gamma = 0.99, double lr = 1e-3);
    void train(int episodes, int max_steps = 100);
    int select_action(const llm::Vector& state);
    void store_transition(const llm::Vector& state, int action, double reward, const llm::Vector& next_state, bool done);
private:
    // Q网络等
};

/** 蒙特卡洛树搜索(MCTS) */
class MCTS {
public:
    struct Node {
        int state;
        int parent;
        std::vector<int> children;
        double value;
        int visits;
    };
    MCTS(int root_state);
    int search(int iterations = 1000);
    void expand(int node_id);
    void update(int node_id, double reward);
private:
    std::vector<Node> tree_;
    int root_;
};

} // namespace other
