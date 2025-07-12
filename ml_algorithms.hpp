#pragma once
#include "llm_math_toolkit.hpp"
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <map>
#include <random>

namespace ml {

// ===== 工具类 =====

class DatasetUtils {
public:
    static void train_test_split(const llm::Matrix& X, const llm::Vector& y, 
                                 double test_ratio, 
                                 llm::Matrix& X_train, llm::Matrix& X_test,
                                 llm::Vector& y_train, llm::Vector& y_test, 
                                 unsigned seed = 42);
    static llm::Vector one_hot(const llm::Vector& y, int n_classes);
    static llm::Matrix normalize(const llm::Matrix& X);
};

// ===== 线性回归 =====

class LinearRegression {
public:
    LinearRegression();
    void fit(const llm::Matrix& X, const llm::Vector& y, double lr = 1e-2, int epochs = 1000);
    llm::Vector predict(const llm::Matrix& X) const;
    double mse(const llm::Matrix& X, const llm::Vector& y) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
    const llm::Vector& weights() const { return weights_; }
    double bias() const { return bias_; }
private:
    llm::Vector weights_;
    double bias_;
};

// ===== 逻辑回归 =====

class LogisticRegression {
public:
    LogisticRegression();
    void fit(const llm::Matrix& X, const llm::Vector& y, double lr = 1e-2, int epochs = 1000);
    llm::Vector predict_proba(const llm::Matrix& X) const;
    llm::Vector predict(const llm::Matrix& X, double threshold = 0.5) const;
    double accuracy(const llm::Matrix& X, const llm::Vector& y) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
    const llm::Vector& weights() const { return weights_; }
    double bias() const { return bias_; }
private:
    llm::Vector weights_;
    double bias_;
    static double sigmoid(double z);
};

// ===== 支持向量机（简易二分类） =====

class SVM {
public:
    SVM(double C = 1.0, double tol = 1e-4, int max_iter = 1000);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    int predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
    const llm::Vector& weights() const { return weights_; }
    double bias() const { return bias_; }
private:
    llm::Vector weights_;
    double bias_;
    double C_;
    double tol_;
    int max_iter_;
};

// ===== 决策树 =====

class DecisionTree {
public:
    DecisionTree(int max_depth = 10, int min_samples_split = 2);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    double predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
private:
    struct Node {
        bool is_leaf;
        double value;
        int feature_index;
        double threshold;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        Node() : is_leaf(false), value(0), feature_index(-1), threshold(0) {}
    };
    std::unique_ptr<Node> root_;
    int max_depth_;
    int min_samples_split_;
    int n_features_;
    void build_tree(std::unique_ptr<Node>& node, const llm::Matrix& X, const llm::Vector& y, int depth);
    double predict_node(const Node* node, const llm::Vector& x) const;
};

// ===== 随机森林 =====

class RandomForest {
public:
    RandomForest(int n_estimators = 10, int max_depth = 10, int min_samples_split = 2);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    double predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
private:
    std::vector<std::unique_ptr<DecisionTree>> trees_;
    int n_estimators_;
    int max_depth_;
    int min_samples_split_;
    std::mt19937 rng_;
};

// ===== KNN =====

class KNN {
public:
    explicit KNN(int k = 5, const std::string& metric = "euclidean");
    void fit(const llm::Matrix& X, const llm::Vector& y);
    double predict(const llm::Vector& x) const;
    llm::Vector predict(const llm::Matrix& X) const;
    const std::string& metric() const { return metric_; }
private:
    int k_;
    std::string metric_;
    llm::Matrix train_X_;
    llm::Vector train_y_;
    double distance(const llm::Vector& a, const llm::Vector& b) const;
};

// ===== 简单神经网络 =====

class SimpleNeuralNetwork {
public:
    SimpleNeuralNetwork(int input_dim, int hidden_dim, int output_dim, const std::string& activation = "relu");
    void fit(const llm::Matrix& X, const llm::Matrix& Y, int epochs = 100, double lr = 1e-2);
    llm::Vector predict(const llm::Vector& x) const;
    llm::Matrix predict(const llm::Matrix& X) const;
    double accuracy(const llm::Matrix& X, const llm::Matrix& Y) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
private:
    std::vector<llm::Matrix> weights_;
    std::vector<llm::Vector> biases_;
    std::string activation_;
    static double relu(double x);
    static double sigmoid(double x);
    static double activate(double x, const std::string& activation);
};

// ===== KMeans =====

class KMeans {
public:
    explicit KMeans(int n_clusters = 3, int max_iter = 100, double tol = 1e-4, int seed = 42);
    void fit(const llm::Matrix& X);
    llm::Vector predict(const llm::Matrix& X) const;
    const std::vector<llm::Vector>& cluster_centers() const;
private:
    int n_clusters_;
    int max_iter_;
    double tol_;
    int seed_;
    std::vector<llm::Vector> centers_;
    static double distance(const llm::Vector& a, const llm::Vector& b);
};

// ===== PCA =====

class PCA {
public:
    explicit PCA(int n_components);
    void fit(const llm::Matrix& X);
    llm::Matrix transform(const llm::Matrix& X) const;
    const llm::Matrix& components() const;
    const llm::Vector& explained_variance() const;
private:
    int n_components_;
    llm::Matrix components_;
    llm::Vector variance_;
};

// ===== AutoEncoder =====

class AutoEncoder {
public:
    AutoEncoder(int input_dim, int hidden_dim);
    void fit(const llm::Matrix& X, int epochs = 100, double lr = 1e-3);
    llm::Matrix encode(const llm::Matrix& X) const;
    llm::Matrix decode(const llm::Matrix& X) const;
    llm::Matrix reconstruct(const llm::Matrix& X) const;
private:
    std::vector<llm::Matrix> encoder_weights_;
    std::vector<llm::Vector> encoder_biases_;
    std::vector<llm::Matrix> decoder_weights_;
    std::vector<llm::Vector> decoder_biases_;
};

// ===== 半监督学习 =====

class SemiSupervisedLabelPropagation {
public:
    SemiSupervisedLabelPropagation(int max_iter = 1000);
    void fit(const llm::Matrix& X, const llm::Vector& y);
    llm::Vector predict(const llm::Matrix& X) const;
private:
    int max_iter_;
    llm::Matrix label_distributions_;
};

// ===== 层次聚类 =====

class HierarchicalClustering {
public:
    explicit HierarchicalClustering(int n_clusters = 2);
    void fit(const llm::Matrix& X);
    llm::Vector predict(const llm::Matrix& X) const;
private:
    int n_clusters_;
    // 树结构等
};

// ===== 强化学习基类接口 =====

class RLAgent {
public:
    virtual void train(int episodes) = 0;
    virtual int select_action(int state) = 0;
    virtual ~RLAgent() {}
};

} // namespace ml
