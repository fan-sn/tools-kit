#pragma once
#include "llm_math_toolkit.hpp"
#include <string>
#include <vector>
#include <memory>

namespace dl {

/** 前馈神经网络 */
class FNN {
public:
    FNN(size_t input_dim, size_t hidden_dim, size_t output_dim, const std::string& activation = "relu");
    void fit(const llm::Matrix& X, const llm::Matrix& Y, int epochs = 100, double lr = 1e-3);
    llm::Vector predict(const llm::Vector& x) const;
    llm::Matrix predict(const llm::Matrix& X) const;
    double accuracy(const llm::Matrix& X, const llm::Matrix& Y) const;
private:
    std::vector<llm::Matrix> weights_;
    std::vector<llm::Vector> biases_;
    std::string activation_;
};

/** 卷积神经网络 */
class CNN {
public:
    CNN(int input_channels, int n_classes);
    void fit(const std::vector<llm::Tensor>& X, const llm::Matrix& Y, int epochs = 10, double lr = 1e-3);
    llm::Vector predict(const llm::Tensor& x) const;
    llm::Matrix predict(const std::vector<llm::Tensor>& X) const;
private:
    // 卷积核、全连接层等
};

/** 简单RNN */
class RNN {
public:
    RNN(size_t input_dim, size_t hidden_dim, size_t output_dim);
    void fit(const std::vector<llm::Matrix>& X, const llm::Matrix& Y, int epochs = 10, double lr = 1e-3);
    llm::Vector predict(const llm::Matrix& x) const;
private:
    // 权重等
};

/** LSTM */
class LSTM {
public:
    LSTM(size_t input_dim, size_t hidden_dim, size_t output_dim);
    void fit(const std::vector<llm::Matrix>& X, const llm::Matrix& Y, int epochs = 10, double lr = 1e-3);
    llm::Vector predict(const llm::Matrix& x) const;
private:
    // 权重等
};

/** GRU */
class GRU {
public:
    GRU(size_t input_dim, size_t hidden_dim, size_t output_dim);
    void fit(const std::vector<llm::Matrix>& X, const llm::Matrix& Y, int epochs = 10, double lr = 1e-3);
    llm::Vector predict(const llm::Matrix& x) const;
private:
    // 权重等
};

/** 生成对抗网络GAN */
class GAN {
public:
    GAN(size_t noise_dim, size_t data_dim, size_t hidden_dim);
    void train(const llm::Matrix& X, int epochs = 1000, double lr = 1e-4);
    llm::Vector generate() const;
private:
    // 生成器和判别器权重等
};

/** Transformer结构 */
class Transformer {
public:
    Transformer(size_t input_dim, size_t num_heads, size_t num_layers, size_t d_ff, size_t output_dim);
    void fit(const std::vector<llm::Matrix>& X, const llm::Matrix& Y, int epochs = 5, double lr = 1e-4);
    llm::Matrix predict(const std::vector<llm::Matrix>& X) const;
private:
    // 多头自注意力等
};

} // namespace dl
