#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include "matrix.h" // 自定义矩阵类

// 激活函数接口
class Activation {
public:
    virtual Matrix forward(const Matrix& x) = 0;
    virtual Matrix backward(const Matrix& x, const Matrix& grad) = 0;
    virtual ~Activation() = default;
};

	//新增激活函数
	//LeakyReLU - 解决 ReLU 的 "神经元死亡" 问题
	//Tanh - 双曲正切函数，输出范围为 [-1, 1]
	//Sigmoid - S 型函数，输出范围为 [0, 1]
	//新增损失函数
	//MSE (Mean Squared Error) - 均方误差损失
	//RMSE (Root Mean Squared Error) - 均方根误差损失
	//Huber Loss - 平滑的 L1 损失，对异常值不敏感
	//所有新添加的激活函数和损失函数都实现了相应的前向传播和反向传播方法。在 DenseLayer 类中也更新了构造函数，支持通过字符串参数选择不同的激活函数。
	//您可以在构建神经网络时使用这些新功能，例如：
	//net.add_layer(std::make_unique<DenseLayer>(784, 128, "LeakyReLU"));
	//net.compile(
	//    std::make_unique<AdamOptimizer>(0.001),
	//    std::make_unique<MSE>()
	//);
// ReLU激活函数
class ReLU : public Activation {
public:
    Matrix forward(const Matrix& x) override {
        return x.apply([](double val) { return std::max(0.0, val); });
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        return x.apply([](double val) { return val > 0 ? 1.0 : 0.0; }) * grad;
    }
};

// LeakyReLU激活函数
class LeakyReLU : public Activation {
private:
    double alpha;
public:
    LeakyReLU(double alpha = 0.01) : alpha(alpha) {}
    
    Matrix forward(const Matrix& x) override {
        return x.apply([this](double val) { 
            return val > 0 ? val : this->alpha * val; 
        });
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        return x.apply([this](double val) { 
            return val > 0 ? 1.0 : this->alpha; 
        }) * grad;
    }
};

// Tanh激活函数
class Tanh : public Activation {
public:
    Matrix forward(const Matrix& x) override {
        return x.apply([](double val) { return std::tanh(val); });
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        Matrix tanh_x = forward(x);
        return (Matrix(1.0) - tanh_x * tanh_x) * grad;
    }
};

// Sigmoid激活函数
class Sigmoid : public Activation {
public:
    Matrix forward(const Matrix& x) override {
        return x.apply([](double val) { 
            return 1.0 / (1.0 + std::exp(-val)); 
        });
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        Matrix sigmoid_x = forward(x);
        return sigmoid_x * (Matrix(1.0) - sigmoid_x) * grad;
    }
};

// Softmax激活函数
class Softmax : public Activation {
public:
    Matrix forward(const Matrix& x) override {
        // 数值稳定性处理
        Matrix max_vals = x.max_rows();
        Matrix shifted_x = x - max_vals.repeat(1, x.cols());
        
        Matrix exp_x = shifted_x.apply([](double val) { return std::exp(val); });
        Matrix sum_exp = exp_x.sum_rows();
        return exp_x / sum_exp;
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        // 简化的Softmax梯度计算
        Matrix softmax = forward(x);
        return softmax * (Matrix(1.0) - softmax) * grad;
    }
};
	//这些新添加的激活函数各有特点：
	//ELU：缓解了 ReLU 的死亡神经元问题，同时输出均值接近零
	//Swish：平滑的非单调激活函数，在多个任务上表现优于 ReLU
	//GELU：在 Transformer 架构中广泛使用的激活函数
	//Mish：自正则化的非单调激活函数，在多个基准测试中表现出色
	//您可以在构建神经网络时使用这些新的激活函数，例如：
	//net.add_layer(std::make_unique<DenseLayer>(784, 128, "GELU"));
	//net.add_layer(std::make_unique<DenseLayer>(128, 64, "Mish"));
// ELU (Exponential Linear Unit) 激活函数
class ELU : public Activation {
private:
    double alpha;
public:
    ELU(double alpha = 1.0) : alpha(alpha) {}
    
    Matrix forward(const Matrix& x) override {
        return x.apply([this](double val) { 
            return val > 0 ? val : this->alpha * (std::exp(val) - 1); 
        });
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        return x.apply([this](double val) { 
            return val > 0 ? 1.0 : this->alpha * std::exp(val); 
        }) * grad;
    }
};

// Swish 激活函数
class Swish : public Activation {
private:
    double beta;
public:
    Swish(double beta = 1.0) : beta(beta) {}
    
    Matrix forward(const Matrix& x) override {
        return x * x.apply([this](double val) { 
            return 1.0 / (1.0 + std::exp(-this->beta * val)); 
        });
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        Matrix sigmoid_bx = x.apply([this](double val) { 
            return 1.0 / (1.0 + std::exp(-this->beta * val)); 
        });
        return (sigmoid_bx * (Matrix(1.0) + this->beta * x * (Matrix(1.0) - sigmoid_bx))) * grad;
    }
};

// GELU (Gaussian Error Linear Unit) 激活函数
class GELU : public Activation {
public:
    Matrix forward(const Matrix& x) override {
        // 近似实现，使用 tanh 版本
        return 0.5 * x * (Matrix(1.0) + 
               (x * 0.7978845608 * (Matrix(1.0) + 0.044715 * x * x)).apply(tanh));
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        // 近似导数
        Matrix cdf = 0.5 * (Matrix(1.0) + 
                     (x * 0.7978845608 * (Matrix(1.0) + 0.044715 * x * x)).apply(tanh));
        Matrix pdf = 0.3989422804 * (-0.5 * x * x).apply(exp);
        return (cdf + x * pdf) * grad;
    }
};

// Mish 激活函数
class Mish : public Activation {
public:
    Matrix forward(const Matrix& x) override {
        return x * (x * 0.5).apply(tanh).apply(log).apply(exp);
    }

    Matrix backward(const Matrix& x, const Matrix& grad) override {
        Matrix sigmoid = (x * 0.5).apply(tanh).apply(exp);
        Matrix tanh_shifted = (x + 1.0).apply(tanh);
        return (sigmoid * (Matrix(1.0) + x * (Matrix(1.0) - tanh_shifted * tanh_shifted))) * grad;
    }
};

// 损失函数接口
class Loss {
public:
    virtual double compute(const Matrix& pred, const Matrix& target) = 0;
    virtual Matrix derivative(const Matrix& pred, const Matrix& target) = 0;
    virtual ~Loss() = default;
};

// 交叉熵损失函数
class CrossEntropy : public Loss {
public:
    double compute(const Matrix& pred, const Matrix& target) override {
        // 避免对数计算中的数值不稳定
        Matrix clipped = pred.apply([](double val) { 
            return std::max(std::min(val, 1 - 1e-10), 1e-10); 
        });
        return -(target * clipped.apply(log)).sum() / pred.rows();
    }

    Matrix derivative(const Matrix& pred, const Matrix& target) override {
        // 避免除以零
        Matrix clipped = pred.apply([](double val) { 
            return std::max(std::min(val, 1 - 1e-10), 1e-10); 
        });
        return -(target / clipped) / pred.rows();
    }
};

// 均方误差损失函数
class MSE : public Loss {
public:
    double compute(const Matrix& pred, const Matrix& target) override {
        Matrix diff = pred - target;
        return (diff * diff).sum() / (2.0 * pred.rows());
    }

    Matrix derivative(const Matrix& pred, const Matrix& target) override {
        return (pred - target) / pred.rows();
    }
};

// 均方根误差损失函数
class RMSE : public Loss {
public:
    double compute(const Matrix& pred, const Matrix& target) override {
        Matrix diff = pred - target;
        return std::sqrt((diff * diff).sum() / pred.rows());
    }

    Matrix derivative(const Matrix& pred, const Matrix& target) override {
        Matrix diff = pred - target;
        double rmse = compute(pred, target);
        return diff / (pred.rows() * rmse + 1e-10); // 避免除以零
    }
};

// Huber损失函数（平滑L1损失）
class HuberLoss : public Loss {
private:
    double delta;
public:
    HuberLoss(double delta = 1.0) : delta(delta) {}
    
    double compute(const Matrix& pred, const Matrix& target) override {
        Matrix diff = pred - target;
        Matrix abs_diff = diff.apply([](double val) { return std::abs(val); });
        
        // 分段计算损失
        Matrix loss = abs_diff.apply([this](double val) {
            return val <= this->delta 
                ? 0.5 * val * val 
                : this->delta * (val - 0.5 * this->delta);
        });
        
        return loss.sum() / pred.rows();
    }

    Matrix derivative(const Matrix& pred, const Matrix& target) override {
        Matrix diff = pred - target;
        Matrix abs_diff = diff.apply([](double val) { return std::abs(val); });
        
        // 分段计算梯度
        return diff.apply([this](double val) {
            return std::abs(val) <= this->delta 
                ? val 
                : this->delta * (val > 0 ? 1.0 : -1.0);
        }) / pred.rows();
    }
};

// 层接口
class Layer {
public:
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad) = 0;
    virtual std::vector<Matrix*> get_parameters() = 0;
    virtual std::vector<Matrix*> get_gradients() = 0;
    virtual ~Layer() = default;
};

// 全连接层
class DenseLayer : public Layer {
private:
    Matrix weights;
    Matrix bias;
    Matrix output_cache;
    Matrix input_cache;
    Matrix grad_weights;
    Matrix grad_bias;
    std::unique_ptr<Activation> activation;

public:
	DenseLayer(int input_size, int output_size, std::string activation_type = "linear") {
	    // Xavier初始化
	    double scale = std::sqrt(2.0 / (input_size + output_size));
	    weights = Matrix(output_size, input_size).randn() * scale;
	    bias = Matrix(output_size, 1).zeros();
	    grad_weights = Matrix(output_size, input_size).zeros();
	    grad_bias = Matrix(output_size, 1).zeros();
	
	    if (activation_type == "ReLU") {
	        activation = std::make_unique<ReLU>();
	    } else if (activation_type == "LeakyReLU") {
	        activation = std::make_unique<LeakyReLU>();
	    } else if (activation_type == "Tanh") {
	        activation = std::make_unique<Tanh>();
	    } else if (activation_type == "Sigmoid") {
	        activation = std::make_unique<Sigmoid>();
	    } else if (activation_type == "Softmax") {
	        activation = std::make_unique<Softmax>();
	    } else if (activation_type == "ELU") {
	        activation = std::make_unique<ELU>();
	    } else if (activation_type == "Swish") {
	        activation = std::make_unique<Swish>();
	    } else if (activation_type == "GELU") {
	        activation = std::make_unique<GELU>();
	    } else if (activation_type == "Mish") {
	        activation = std::make_unique<Mish>();
	    } else {
	        // 默认线性激活
	        class Linear : public Activation {
	        public:
	            Matrix forward(const Matrix& x) override { return x; }
	            Matrix backward(const Matrix& x, const Matrix& grad) override { return grad; }
	        };
	        activation = std::make_unique<Linear>();
	    }
	}

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        output_cache = weights * input + bias.repeat(1, input.cols());
        return activation->forward(output_cache);
    }

    Matrix backward(const Matrix& grad) override {
        Matrix activation_grad = activation->backward(output_cache, grad);
        grad_weights = activation_grad * input_cache.transpose();
        grad_bias = activation_grad.sum_cols();
        
        return weights.transpose() * activation_grad;
    }

    std::vector<Matrix*> get_parameters() override {
        return {&weights, &bias};
    }

    std::vector<Matrix*> get_gradients() override {
        return {&grad_weights, &grad_bias};
    }
};

// 优化器接口
class Optimizer {
public:
    virtual void update(Layer& layer) = 0;
    virtual ~Optimizer() = default;
};

// Adam优化器
class AdamOptimizer : public Optimizer {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::unordered_map<Matrix*, Matrix> m; // 一阶矩估计
    std::unordered_map<Matrix*, Matrix> v; // 二阶矩估计

public:
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void update(Layer& layer) override {
        t++;
        auto params = layer.get_parameters();
        auto grads = layer.get_gradients();

        for (size_t i = 0; i < params.size(); ++i) {
            Matrix& param = *params[i];
            Matrix& grad = *grads[i];

            // 初始化一阶和二阶矩缓存
            if (m.find(params[i]) == m.end()) {
                m[params[i]] = Matrix(param.rows(), param.cols()).zeros();
                v[params[i]] = Matrix(param.rows(), param.cols()).zeros();
            }

            // 更新矩估计
            m[params[i]] = beta1 * m[params[i]] + (1 - beta1) * grad;
            v[params[i]] = beta2 * v[params[i]] + (1 - beta2) * (grad * grad);

            // 偏差校正
            Matrix m_hat = m[params[i]] / (1 - std::pow(beta1, t));
            Matrix v_hat = v[params[i]] / (1 - std::pow(beta2, t));

            // 更新参数
            param = param - learning_rate * m_hat / (v_hat.apply(sqrt) + epsilon);
        }
    }
};

// 卷积层
class Conv2DLayer : public Layer {
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    Matrix weights;  // [out_channels, in_channels, kernel_size, kernel_size]
    Matrix bias;     // [out_channels]
    Matrix output_cache;
    Matrix input_cache;
    Matrix grad_weights;
    Matrix grad_bias;

public:
    Conv2DLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
        : in_channels(in_channels), out_channels(out_channels), 
          kernel_size(kernel_size), stride(stride), padding(padding) {
        
        // He初始化
        double scale = std::sqrt(2.0 / (in_channels * kernel_size * kernel_size));
        weights = Matrix(out_channels, in_channels * kernel_size * kernel_size).randn() * scale;
        bias = Matrix(out_channels, 1).zeros();
        grad_weights = Matrix(out_channels, in_channels * kernel_size * kernel_size).zeros();
        grad_bias = Matrix(out_channels, 1).zeros();
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        
        int batch_size = input.dim(0);
        int in_height = input.dim(2);
        int in_width = input.dim(3);
        
        int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
        int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
        
        output_cache = Matrix(batch_size, out_channels, out_height, out_width);
        
        // 实现卷积操作
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    // 对每个通道进行卷积
                    for (int i = 0; i < out_height; ++i) {
                        for (int j = 0; j < out_width; ++j) {
                            double sum = 0.0;
                            for (int ki = 0; ki < kernel_size; ++ki) {
                                for (int kj = 0; kj < kernel_size; ++kj) {
                                    int in_i = i * stride + ki - padding;
                                    int in_j = j * stride + kj - padding;
                                    
                                    // 处理padding
                                    double val = 0.0;
                                    if (in_i >= 0 && in_i < in_height && in_j >= 0 && in_j < in_width) {
                                        val = input(b, ic, in_i, in_j);
                                    }
                                    
                                    sum += val * weights(oc, ic * kernel_size * kernel_size + ki * kernel_size + kj);
                                }
                            }
                            output_cache(b, oc, i, j) += sum;
                        }
                    }
                }
                
                // 添加偏置
                for (int i = 0; i < out_height; ++i) {
                    for (int j = 0; j < out_width; ++j) {
                        output_cache(b, oc, i, j) += bias(oc, 0);
                    }
                }
            }
        }
        
        return output_cache;
    }

    Matrix backward(const Matrix& grad) override {
        int batch_size = input_cache.dim(0);
        int in_height = input_cache.dim(2);
        int in_width = input_cache.dim(3);
        int out_height = grad.dim(2);
        int out_width = grad.dim(3);
        
        Matrix input_grad = Matrix(batch_size, in_channels, in_height, in_width);
        grad_weights = Matrix::zeros_like(weights);
        grad_bias = Matrix::zeros_like(bias);
        
        // 计算梯度
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int i = 0; i < out_height; ++i) {
                        for (int j = 0; j < out_width; ++j) {
                            for (int ki = 0; ki < kernel_size; ++ki) {
                                for (int kj = 0; kj < kernel_size; ++kj) {
                                    int in_i = i * stride + ki - padding;
                                    int in_j = j * stride + kj - padding;
                                    
                                    if (in_i >= 0 && in_i < in_height && in_j >= 0 && in_j < in_width) {
                                        double grad_val = grad(b, oc, i, j);
                                        input_grad(b, ic, in_i, in_j) += weights(oc, ic * kernel_size * kernel_size + ki * kernel_size + kj) * grad_val;
                                        grad_weights(oc, ic * kernel_size * kernel_size + ki * kernel_size + kj) += input_cache(b, ic, in_i, in_j) * grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 计算偏置梯度
                for (int i = 0; i < out_height; ++i) {
                    for (int j = 0; j < out_width; ++j) {
                        grad_bias(oc, 0) += grad(b, oc, i, j);
                    }
                }
            }
        }
        
        return input_grad;
    }

    std::vector<Matrix*> get_parameters() override {
        return {&weights, &bias};
    }

    std::vector<Matrix*> get_gradients() override {
        return {&grad_weights, &grad_bias};
    }
};

// 最大池化层
class MaxPool2DLayer : public Layer {
private:
    int pool_size;
    int stride;
    Matrix output_cache;
    Matrix input_cache;

public:
    MaxPool2DLayer(int pool_size, int stride = 2) 
        : pool_size(pool_size), stride(stride) {}

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        
        int batch_size = input.dim(0);
        int channels = input.dim(1);
        int in_height = input.dim(2);
        int in_width = input.dim(3);
        
        int out_height = (in_height - pool_size) / stride + 1;
        int out_width = (in_width - pool_size) / stride + 1;
        
        output_cache = Matrix(batch_size, channels, out_height, out_width);
        
        // 实现最大池化
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_height; ++i) {
                    for (int j = 0; j < out_width; ++j) {
                        double max_val = -std::numeric_limits<double>::infinity();
                        
                        for (int ki = 0; ki < pool_size; ++ki) {
                            for (int kj = 0; kj < pool_size; ++kj) {
                                int in_i = i * stride + ki;
                                int in_j = j * stride + kj;
                                
                                if (in_i < in_height && in_j < in_width) {
                                    double val = input(b, c, in_i, in_j);
                                    if (val > max_val) {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                        
                        output_cache(b, c, i, j) = max_val;
                    }
                }
            }
        }
        
        return output_cache;
    }

    Matrix backward(const Matrix& grad) override {
        int batch_size = input_cache.dim(0);
        int channels = input_cache.dim(1);
        int in_height = input_cache.dim(2);
        int in_width = input_cache.dim(3);
        int out_height = grad.dim(2);
        int out_width = grad.dim(3);
        
        Matrix input_grad = Matrix::zeros_like(input_cache);
        
        // 计算梯度（只对最大值位置传递梯度）
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_height; ++i) {
                    for (int j = 0; j < out_width; ++j) {
                        double max_val = -std::numeric_limits<double>::infinity();
                        int max_i = -1, max_j = -1;
                        
                        // 找到最大值位置
                        for (int ki = 0; ki < pool_size; ++ki) {
                            for (int kj = 0; kj < pool_size; ++kj) {
                                int in_i = i * stride + ki;
                                int in_j = j * stride + kj;
                                
                                if (in_i < in_height && in_j < in_width) {
                                    double val = input_cache(b, c, in_i, in_j);
                                    if (val > max_val) {
                                        max_val = val;
                                        max_i = in_i;
                                        max_j = in_j;
                                    }
                                }
                            }
                        }
                        
                        // 只对最大值位置传递梯度
                        if (max_i != -1 && max_j != -1) {
                            input_grad(b, c, max_i, max_j) += grad(b, c, i, j);
                        }
                    }
                }
            }
        }
        
        return input_grad;
    }

    std::vector<Matrix*> get_parameters() override {
        return {}; // 无参数
    }

    std::vector<Matrix*> get_gradients() override {
        return {}; // 无参数梯度
    }
};

// 展平层：将多维输入展平为一维向量
class FlattenLayer : public Layer {
private:
    Matrix output_cache;
    Matrix input_cache;
    std::vector<int> input_shape;

public:
    FlattenLayer() {}

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        input_shape = input.shape();
        
        int batch_size = input.dim(0);
        int flattened_size = 1;
        for (size_t i = 1; i < input_shape.size(); ++i) {
            flattened_size *= input_shape[i];
        }
        
        output_cache = Matrix(batch_size, flattened_size);
        
        // 展平操作
        for (int b = 0; b < batch_size; ++b) {
            int idx = 0;
            for (int c = 0; c < input_shape[1]; ++c) {
                for (int i = 0; i < input_shape[2]; ++i) {
                    for (int j = 0; j < input_shape[3]; ++j) {
                        output_cache(b, idx++) = input(b, c, i, j);
                    }
                }
            }
        }
        
        return output_cache;
    }

    Matrix backward(const Matrix& grad) override {
        int batch_size = grad.dim(0);
        Matrix input_grad(batch_size, input_shape[1], input_shape[2], input_shape[3]);
        
        // 恢复原始形状
        for (int b = 0; b < batch_size; ++b) {
            int idx = 0;
            for (int c = 0; c < input_shape[1]; ++c) {
                for (int i = 0; i < input_shape[2]; ++i) {
                    for (int j = 0; j < input_shape[3]; ++j) {
                        input_grad(b, c, i, j) = grad(b, idx++);
                    }
                }
            }
        }
        
        return input_grad;
    }

    std::vector<Matrix*> get_parameters() override {
        return {}; // 无参数
    }

    std::vector<Matrix*> get_gradients() override {
        return {}; // 无参数梯度
    }
};

// MNIST数据集加载函数
Matrix load_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    
    // 读取文件头
    int magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);
    
    // 转换字节序
    magic_number = ((magic_number >> 24) & 0xFF) | ((magic_number << 8) & 0xFF0000) | 
                   ((magic_number >> 8) & 0xFF00) | ((magic_number << 24) & 0xFF000000);
    num_images = ((num_images >> 24) & 0xFF) | ((num_images << 8) & 0xFF0000) | 
                 ((num_images >> 8) & 0xFF00) | ((num_images << 24) & 0xFF000000);
    num_rows = ((num_rows >> 24) & 0xFF) | ((num_rows << 8) & 0xFF0000) | 
               ((num_rows >> 8) & 0xFF00) | ((num_rows << 24) & 0xFF000000);
    num_cols = ((num_cols >> 24) & 0xFF) | ((num_cols << 8) & 0xFF0000) | 
               ((num_cols >> 8) & 0xFF00) | ((num_cols << 24) & 0xFF000000);
    
    // 读取图像数据
    std::vector<unsigned char> buffer(num_images * num_rows * num_cols);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    
    // 创建矩阵并归一化
    Matrix images(num_images, 1, num_rows, num_cols);
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < num_rows; ++r) {
            for (int c = 0; c < num_cols; ++c) {
                images(i, 0, r, c) = static_cast<double>(buffer[i * num_rows * num_cols + r * num_cols + c]) / 255.0;
            }
        }
    }
    
    return images;
}
//接下来，添加 MNIST 数据集加载函数：
Matrix load_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    
    // 读取文件头
    int magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    
    // 转换字节序
    magic_number = ((magic_number >> 24) & 0xFF) | ((magic_number << 8) & 0xFF0000) | 
                   ((magic_number >> 8) & 0xFF00) | ((magic_number << 24) & 0xFF000000);
    num_labels = ((num_labels >> 24) & 0xFF) | ((num_labels << 8) & 0xFF0000) | 
                 ((num_labels >> 8) & 0xFF00) | ((num_labels << 24) & 0xFF000000);
    
    // 读取标签数据
    std::vector<unsigned char> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    
    // 创建矩阵
    Matrix labels(num_labels, 1);
    for (int i = 0; i < num_labels; ++i) {
        labels(i, 0) = static_cast<double>(buffer[i]);
    }
    
    return labels;
}

// 正则化接口
class Regularization {
public:
    virtual double compute(const std::vector<Matrix*>& params) = 0;
    virtual void apply_gradients(std::vector<Matrix*>& params, std::vector<Matrix*>& grads, double lr) = 0;
    virtual ~Regularization() = default;
};

// L2正则化（权重衰减）
class L2Regularization : public Regularization {
private:
    double lambda;  // 正则化强度
public:
    L2Regularization(double lambda = 0.001) : lambda(lambda) {}
    
    double compute(const std::vector<Matrix*>& params) override {
        double reg_loss = 0.0;
        for (auto param : params) {
            reg_loss += ((*param) * (*param)).sum();
        }
        return 0.5 * lambda * reg_loss;
    }
    
    void apply_gradients(std::vector<Matrix*>& params, std::vector<Matrix*>& grads, double lr) override {
        for (size_t i = 0; i < params.size(); ++i) {
            *grads[i] = *grads[i] + lambda * (*params[i]);  // 添加L2梯度
        }
    }
};

// L1正则化
class L1Regularization : public Regularization {
private:
    double lambda;  // 正则化强度
public:
    L1Regularization(double lambda = 0.001) : lambda(lambda) {}
    
    double compute(const std::vector<Matrix*>& params) override {
        double reg_loss = 0.0;
        for (auto param : params) {
            reg_loss += param->apply([](double val) { return std::abs(val); }).sum();
        }
        return lambda * reg_loss;
    }
    
    void apply_gradients(std::vector<Matrix*>& params, std::vector<Matrix*>& grads, double lr) override {
        for (size_t i = 0; i < params.size(); ++i) {
            *grads[i] = *grads[i] + lambda * params[i]->apply([](double val) { 
                return val > 0 ? 1.0 : (val < 0 ? -1.0 : 0.0); 
            });  // 添加L1梯度
        }
    }
};

// 组合正则化（L1 + L2）
class L1L2Regularization : public Regularization {
private:
    std::unique_ptr<L1Regularization> l1;
    std::unique_ptr<L2Regularization> l2;
public:
    L1L2Regularization(double l1_lambda = 0.001, double l2_lambda = 0.001) 
        : l1(std::make_unique<L1Regularization>(l1_lambda)),
          l2(std::make_unique<L2Regularization>(l2_lambda)) {}
    
    double compute(const std::vector<Matrix*>& params) override {
        return l1->compute(params) + l2->compute(params);
    }
    
    void apply_gradients(std::vector<Matrix*>& params, std::vector<Matrix*>& grads, double lr) override {
        l1->apply_gradients(params, grads, lr);
        l2->apply_gradients(params, grads, lr);
    }
};

//接下来，我们需要修改 Optimizer 接口和 AdamOptimizer 类以支持正则化：
// 优化器接口（更新以支持正则化）
class Optimizer {
protected:
    std::unique_ptr<Regularization> regularization;
public:
    Optimizer(std::unique_ptr<Regularization> reg = nullptr) 
        : regularization(std::move(reg)) {}
    
    virtual void update(Layer& layer, double learning_rate) = 0;
    virtual ~Optimizer() = default;
};

// Adam优化器（更新以支持正则化）
class AdamOptimizer : public Optimizer {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::unordered_map<Matrix*, Matrix> m; // 一阶矩估计
    std::unordered_map<Matrix*, Matrix> v; // 二阶矩估计

public:
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, 
                  double eps = 1e-8, std::unique_ptr<Regularization> reg = nullptr)
        : Optimizer(std::move(reg)), learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void update(Layer& layer) override {
        t++;
        auto params = layer.get_parameters();
        auto grads = layer.get_gradients();
        
        // 应用正则化（如果有）
        if (regularization) {
            regularization->apply_gradients(params, grads, learning_rate);
        }

        for (size_t i = 0; i < params.size(); ++i) {
            Matrix& param = *params[i];
            Matrix& grad = *grads[i];

            // 初始化一阶和二阶矩缓存
            if (m.find(params[i]) == m.end()) {
                m[params[i]] = Matrix(param.rows(), param.cols()).zeros();
                v[params[i]] = Matrix(param.rows(), param.cols()).zeros();
            }

            // 更新矩估计
            m[params[i]] = beta1 * m[params[i]] + (1 - beta1) * grad;
            v[params[i]] = beta2 * v[params[i]] + (1 - beta2) * (grad * grad);

            // 偏差校正
            Matrix m_hat = m[params[i]] / (1 - std::pow(beta1, t));
            Matrix v_hat = v[params[i]] / (1 - std::pow(beta2, t));

            // 更新参数
            param = param - learning_rate * m_hat / (v_hat.apply(sqrt) + epsilon);
        }
    }
};

// Dropout层
class DropoutLayer : public Layer {
private:
    double keep_prob;  // 保留神经元的概率
    Matrix mask;       // 用于存储丢弃掩码
    bool is_training;  // 训练/推理模式标志

public:
    DropoutLayer(double keep_prob = 0.5) 
        : keep_prob(keep_prob), is_training(true) {}

    void set_training_mode(bool mode) override {
        is_training = mode;
    }

    Matrix forward(const Matrix& input) override {
        if (is_training) {
            // 生成随机掩码
            mask = Matrix(input.rows(), input.cols()).rand() < keep_prob;
            // 应用掩码并缩放
            return input * mask / keep_prob;
        } else {
            // 在推理时不应用dropout，直接返回输入
            return input;
        }
    }

    Matrix backward(const Matrix& grad) override {
        if (is_training) {
            // 将梯度通过掩码传递
            return grad * mask / keep_prob;
        } else {
            return grad;
        }
    }

    std::vector<Matrix*> get_parameters() override {
        return {}; // 无参数
    }

    std::vector<Matrix*> get_gradients() override {
        return {}; // 无参数梯度
    }
};

// 早停法类
class EarlyStopping {
private:
    double patience;        // 等待轮数
    double min_delta;       // 最小改进阈值
    double best_score;      // 最佳分数
    int counter;            // 计数器
    bool early_stop;        // 是否停止标志
    std::string path;       // 模型保存路径

public:
    EarlyStopping(double patience = 10, double min_delta = 0.0, std::string path = "best_model.weights")
        : patience(patience), min_delta(min_delta), best_score(-std::numeric_limits<double>::infinity()),
          counter(0), early_stop(false), path(path) {}

    // 检查是否应该停止训练
    bool check(double validation_loss, NeuralNetwork& model) {
        double score = -validation_loss;  // 因为我们希望损失越小越好
        
        if (score > best_score + min_delta) {
            // 有改进，保存模型
            best_score = score;
            counter = 0;
            model.save_weights(path);  // 需要实现save_weights方法
        } else {
            // 没有改进，增加计数器
            counter++;
            std::cout << "EarlyStopping counter: " << counter << " out of " << patience << std::endl;
            
            if (counter >= patience) {
                early_stop = true;
                // 恢复最佳模型
                model.load_weights(path);  // 需要实现load_weights方法
            }
        }
        
        return early_stop;
    }

    // 获取是否应该停止的标志
    bool should_stop() const {
        return early_stop;
    }
};
//
// 批归一化层
class BatchNormLayer : public Layer {
private:
    double epsilon;      // 防止除零的小常数
    double momentum;     // 移动平均的动量
    Matrix gamma;        // 缩放参数
    Matrix beta;         // 偏移参数
    Matrix running_mean; // 训练期间计算的均值
    Matrix running_var;  // 训练期间计算的方差
    Matrix mean_cache;   // 存储当前批次的均值（用于反向传播）
    Matrix var_cache;    // 存储当前批次的方差（用于反向传播）
    Matrix x_norm_cache; // 存储归一化后的输入（用于反向传播）
    bool is_training;    // 训练/推理模式标志

public:
    BatchNormLayer(int input_size, double epsilon = 1e-5, double momentum = 0.9)
        : epsilon(epsilon), momentum(momentum), is_training(true) {
        // 初始化可学习参数
        gamma = Matrix(input_size, 1).ones();  // 缩放参数初始化为1
        beta = Matrix(input_size, 1).zeros();  // 偏移参数初始化为0
        
        // 初始化运行统计量
        running_mean = Matrix(input_size, 1).zeros();
        running_var = Matrix(input_size, 1).ones();
    }

    void set_training_mode(bool mode) override {
        is_training = mode;
    }

    Matrix forward(const Matrix& input) override {
        int batch_size = input.rows();
        
        if (is_training) {
            // 计算当前批次的均值
            mean_cache = input.sum_rows() / batch_size;
            
            // 计算当前批次的方差
            Matrix x_minus_mean = input - mean_cache.repeat(batch_size, 1);
            var_cache = (x_minus_mean * x_minus_mean).sum_rows() / batch_size;
            
            // 更新运行统计量
            running_mean = momentum * running_mean + (1 - momentum) * mean_cache;
            running_var = momentum * running_var + (1 - momentum) * var_cache;
            
            // 归一化
            x_norm_cache = x_minus_mean / (var_cache.repeat(batch_size, 1).apply(sqrt) + epsilon);
            
            // 应用缩放和偏移
            return x_norm_cache * gamma.repeat(batch_size, 1) + beta.repeat(batch_size, 1);
        } else {
            // 在推理时使用预计算的运行统计量
            Matrix x_minus_mean = input - running_mean.repeat(batch_size, 1);
            Matrix x_norm = x_minus_mean / (running_var.repeat(batch_size, 1).apply(sqrt) + epsilon);
            
            // 应用缩放和偏移
            return x_norm * gamma.repeat(batch_size, 1) + beta.repeat(batch_size, 1);
        }
    }

    Matrix backward(const Matrix& grad) override {
        int batch_size = grad.rows();
        
        // 恢复归一化的输入和统计量
        Matrix x_norm = x_norm_cache;
        Matrix var = var_cache;
        
        // 计算gamma和beta的梯度
        grad_gamma = (grad * x_norm).sum_rows();
        grad_beta = grad.sum_rows();
        
        // 计算输入的梯度
        Matrix x_minus_mean = x_norm * (var.repeat(batch_size, 1).apply(sqrt) + epsilon);
        Matrix std_inv = 1.0 / (var.repeat(batch_size, 1).apply(sqrt) + epsilon);
        
        Matrix d_x_norm = grad * gamma.repeat(batch_size, 1);
        Matrix d_var = (-0.5) * (d_x_norm * x_norm).sum_rows() * (var.apply([](double v) { return v * v * v; }));
        Matrix d_mean = (-1.0) * (d_x_norm * std_inv).sum_rows() - (2.0 / batch_size) * d_var * x_minus_mean.sum_rows();
        
        Matrix d_input = d_x_norm * std_inv + (2.0 / batch_size) * d_var.repeat(batch_size, 1) * x_minus_mean + d_mean.repeat(batch_size, 1) / batch_size;
        
        return d_input;
    }

    std::vector<Matrix*> get_parameters() override {
        return {&gamma, &beta};
    }

    std::vector<Matrix*> get_gradients() override {
        return {&grad_gamma, &grad_beta};
    }

private:
    Matrix grad_gamma;  // gamma的梯度
    Matrix grad_beta;   // beta的梯度
};

