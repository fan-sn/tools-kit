#pragma once
#include "llm_math_toolkit.hpp"
#include <map>
#include <string>
#include <vector>

namespace fuzzy {

struct FuzzyRule {
    std::string antecedent;
    std::string consequent;
    double weight = 1.0;
};

class FuzzyInferenceSystem {
public:
    void add_rule(const FuzzyRule& rule);
    double infer(const std::map<std::string, double>& inputs) const;
private:
    std::vector<FuzzyRule> rules_;
};

class ExpertRuleSystem {
public:
    struct Rule {
        std::function<bool(const std::map<std::string, double>&)> condition;
        std::function<void()> action;
    };
    void add_rule(const Rule& rule);
    void evaluate(const std::map<std::string, double>& inputs);
private:
    std::vector<Rule> rules_;
};

class BayesianNetwork {
public:
    void add_node(const std::string& name, const std::vector<std::string>& parents);
    void set_cpt(const std::string& name, const llm::Matrix& cpt);
    double infer(const std::map<std::string, int>& evidence, const std::string& query) const;
private:
    // 网络结构等
};

} // namespace fuzzy
