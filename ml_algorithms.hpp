#pragma once
#include "llm_math_toolkit.hpp"
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace ml {

// ===== 监督学习 =====

class LinearRegression {
public:
    LinearRegression();
    void fit(const llm::Matrix& X, const llm::Vector& y, double lr = 1e-2, int epochs = 1000);
    llm::Vector predict(const llm::Matrix& X) const;
    double mse(const llm::Matrix& X, const llm::Vector& y) const;
private:
    llm::Vector weights_;
    double bias_;
};

class LogisticRegression {
public:
    LogisticRegression();
    void fit(const llm::Matrix& X, const llm::Vector& y, double lr = 1e-2, int epochs = 1000);
    llm::Vector predict_proba(const llm::Matrix& X) const;
    llm::Vector predict(const llm::Matrix& X, double threshold = 0.5) const;
    double accuracy(const llm::Matrix& X, const llm::Vector& y) const;
private:
    llm::Vector weights_;
    double bias_;
};

class SVM {
public:
    SVM(double C = 1.0, double tol = 1e-4, int max_iter = 1000);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    int predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
private:
    llm::Vector weights_;
    double bias_;
    double C_;
    double tol_;
    int max_iter_;
};

class DecisionTree {
public:
    DecisionTree(int max_depth = 10, int min_samples_split = 2);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    double predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
private:
    struct Node; // 内部递归结构
    std::shared_ptr<Node> root_;
    int max_depth_;
    int min_samples_split_;
};

class RandomForest {
public:
    RandomForest(int n_estimators = 10, int max_depth = 10);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    double predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
private:
    std::vector<std::shared_ptr<DecisionTree>> trees_;
    int n_estimators_;
    int max_depth_;
};

class KNN {
public:
    explicit KNN(int k = 5);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    double predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
private:
    int k_;
    llm::Matrix train_X_;
    llm::Vector train_y_;
};

class SimpleNeuralNetwork {
public:
    SimpleNeuralNetwork(int input_dim, int hidden_dim, int output_dim);
    void fit(const llm::Matrix& X, const llm::Matrix& Y, int epochs = 100, double lr = 1e-2);
    llm::Vector predict(const llm::Vector& x) const;
    llm::Matrix predict(const llm::Matrix& X) const;
private:
    std::vector<llm::Matrix> weights_;
    std::vector<llm::Vector> biases_;
};

// ===== 无监督学习 =====

class KMeans {
public:
    explicit KMeans(int n_clusters = 3, int max_iter = 100);
    void fit(const llm::Matrix& X);
    llm::Vector predict(const llm::Matrix& X) const;
    const std::vector<llm::Vector>& cluster_centers() const;
private:
    int n_clusters_;
    int max_iter_;
    std::vector<llm::Vector> centers_;
};

class HierarchicalClustering {
public:
    explicit HierarchicalClustering(int n_clusters = 2);
    void fit(const llm::Matrix& X);
    llm::Vector predict(const llm::Matrix& X) const;
private:
    int n_clusters_;
    // 树结构等
};

class PCA {
public:
    PCA(int n_components);
    void fit(const llm::Matrix& X);
    llm::Matrix transform(const llm::Matrix& X) const;
    llm::Matrix components() const;
private:
    int n_components_;
    llm::Matrix components_;
};

class AutoEncoder {
public:
    AutoEncoder(int input_dim, int hidden_dim);
    void fit(const llm::Matrix& X, int epochs = 100, double lr = 1e-3);
    llm::Matrix encode(const llm::Matrix& X) const;
    llm::Matrix decode(const llm::Matrix& X) const;
    llm::Matrix reconstruct(const llm::Matrix& X) const;
private:
    std::vector<llm::Matrix> encoder_weights_;
    std::vector<llm::Matrix> decoder_weights_;
};

// ===== 半监督学习（接口示例） =====

class SemiSupervisedLabelPropagation {
public:
    SemiSupervisedLabelPropagation(int max_iter = 1000);
    void fit(const llm::Matrix& X, const llm::Vector& y); // y中未标注可用-1
    llm::Vector predict(const llm::Matrix& X) const;
private:
    int max_iter_;
    // ...
};

// ===== 强化学习基类接口 =====

class RLAgent {
public:
    virtual void train(int episodes) = 0;
    virtual int select_action(int state) = 0;
    virtual ~RLAgent() {}
};

} // namespace ml
