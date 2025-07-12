//```cpp name=llm_math_toolkit.hpp
#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <functional>
#include <limits>
#include <tuple>
#include <string>
#include <memory>
#include <stdexcept>
#include <initializer_list>
#include <numeric>

namespace llm_math {

// ==================== 1. 泛型 Vector ==================== //
template<typename T>
struct GenVector {
    std::vector<T> data;
    GenVector() {}
    explicit GenVector(size_t n, T val = T()) : data(n, val) {}
    GenVector(std::initializer_list<T> l) : data(l) {}
    size_t size() const { return data.size(); }
    void resize(size_t n, T val = T()) { data.resize(n, val); }
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
    GenVector<T> operator+(const GenVector<T>& rhs) const {
        assert(data.size() == rhs.data.size());
        GenVector<T> ret(data.size());
        for (size_t i = 0; i < data.size(); ++i) ret.data[i] = data[i] + rhs.data[i];
        return ret;
    }
    GenVector<T> operator-(const GenVector<T>& rhs) const {
        assert(data.size() == rhs.data.size());
        GenVector<T> ret(data.size());
        for (size_t i = 0; i < data.size(); ++i) ret.data[i] = data[i] - rhs.data[i];
        return ret;
    }
    GenVector<T> operator*(T scalar) const {
        GenVector<T> ret(data.size());
        for (size_t i = 0; i < data.size(); ++i) ret.data[i] = data[i] * scalar;
        return ret;
    }
    GenVector<T> operator/(T scalar) const {
        GenVector<T> ret(data.size());
        for (size_t i = 0; i < data.size(); ++i) ret.data[i] = data[i] / scalar;
        return ret;
    }
    T dot(const GenVector<T>& rhs) const {
        assert(data.size() == rhs.data.size());
        T s = T();
        for (size_t i = 0; i < data.size(); ++i) s += data[i] * rhs.data[i];
        return s;
    }
    double norm(int p = 2) const {
        double sum = 0;
        for (size_t i = 0; i < data.size(); ++i)
            sum += std::pow(std::abs(static_cast<double>(data[i])), p);
        return std::pow(sum, 1.0 / p);
    }
    void fill(T val) { std::fill(data.begin(), data.end(), val); }
    void zero() { fill(T()); }
    typename std::vector<T>::iterator begin() { return data.begin(); }
    typename std::vector<T>::iterator end() { return data.end(); }
    typename std::vector<T>::const_iterator begin() const { return data.begin(); }
    typename std::vector<T>::const_iterator end() const { return data.end(); }
};

// ==================== 2. 泛型 Matrix ==================== //
template<typename T>
struct GenMatrix {
    std::vector<std::vector<T>> data;
    GenMatrix() : data() {}
    GenMatrix(size_t r, size_t c, T val = T()) : data(r, std::vector<T>(c, val)) {}
    GenMatrix(std::initializer_list<std::vector<T>> l) : data(l) {}
    size_t rows() const { return data.size(); }
    size_t cols() const { return data.empty() ? 0 : data[0].size(); }
    void resize(size_t r, size_t c, T val = T()) { data.resize(r, std::vector<T>(c, val)); }
    std::vector<T>& operator[](size_t i) { return data[i]; }
    const std::vector<T>& operator[](size_t i) const { return data[i]; }
    GenMatrix<T> operator+(const GenMatrix<T>& rhs) const {
        assert(rows() == rhs.rows() && cols() == rhs.cols());
        GenMatrix<T> ret(rows(), cols());
        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = 0; j < cols(); ++j)
                ret.data[i][j] = data[i][j] + rhs.data[i][j];
        return ret;
    }
    GenMatrix<T> operator-(const GenMatrix<T>& rhs) const {
        assert(rows() == rhs.rows() && cols() == rhs.cols());
        GenMatrix<T> ret(rows(), cols());
        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = 0; j < cols(); ++j)
                ret.data[i][j] = data[i][j] - rhs.data[i][j];
        return ret;
    }
    GenMatrix<T> operator*(T scalar) const {
        GenMatrix<T> ret(rows(), cols());
        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = 0; j < cols(); ++j)
                ret.data[i][j] = data[i][j] * scalar;
        return ret;
    }
    GenMatrix<T> operator/(T scalar) const {
        GenMatrix<T> ret(rows(), cols());
        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = 0; j < cols(); ++j)
                ret.data[i][j] = data[i][j] / scalar;
        return ret;
    }
    GenMatrix<T> operator*(const GenMatrix<T>& rhs) const {
        assert(cols() == rhs.rows());
        size_t m = rows(), n = cols(), p = rhs.cols();
        GenMatrix<T> ret(m, p, T());
        for (size_t i = 0; i < m; ++i)
            for (size_t k = 0; k < n; ++k)
                for (size_t j = 0; j < p; ++j)
                    ret.data[i][j] += data[i][k] * rhs.data[k][j];
        return ret;
    }
    GenVector<T> operator*(const GenVector<T>& v) const {
        assert(cols() == v.size());
        GenVector<T> ret(rows());
        for (size_t i = 0; i < rows(); ++i) {
            ret.data[i] = T();
            for (size_t j = 0; j < cols(); ++j)
                ret.data[i] += data[i][j] * v.data[j];
        }
        return ret;
    }
    GenMatrix<T> transpose() const {
        if (rows() == 0) return GenMatrix<T>();
        GenMatrix<T> ret(cols(), rows());
        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = 0; j < cols(); ++j)
                ret.data[j][i] = data[i][j];
        return ret;
    }
    void fill(T val) { for (size_t i = 0; i < rows(); ++i) std::fill(data[i].begin(), data[i].end(), val); }
    void zero() { fill(T()); }
    bool is_square() const { return rows() == cols(); }
    static GenMatrix<T> identity(size_t n) {
        GenMatrix<T> id(n, n, T());
        for (size_t i = 0; i < n; ++i) id.data[i][i] = T(1);
        return id;
    }
};

// ==================== 3. 稀疏 Vector ==================== //
template<typename T>
struct SparseVector {
    std::map<size_t, T> data; // index -> value
    size_t dim;

    SparseVector(size_t n) : dim(n) {}
    void set(size_t i, T val) { if (val == T()) data.erase(i); else data[i] = val; }
    T get(size_t i) const { auto it = data.find(i); return it == data.end() ? T() : it->second; }
    size_t size() const { return dim; }
    void clear() { data.clear(); }
    double norm(int p = 2) const {
        double sum = 0;
        for (auto& kv : data) sum += std::pow(std::abs(static_cast<double>(kv.second)), p);
        return std::pow(sum, 1.0 / p);
    }
    SparseVector<T> operator+(const SparseVector<T>& rhs) const {
        assert(dim == rhs.dim);
        SparseVector<T> ret(dim);
        for (auto& kv : data) ret.data[kv.first] = kv.second;
        for (auto& kv : rhs.data) ret.data[kv.first] += kv.second;
        for (auto it = ret.data.begin(); it != ret.data.end();) {
            if (it->second == T()) it = ret.data.erase(it); else ++it;
        }
        return ret;
    }
    SparseVector<T> operator-(const SparseVector<T>& rhs) const {
        assert(dim == rhs.dim);
        SparseVector<T> ret(dim);
        for (auto& kv : data) ret.data[kv.first] = kv.second;
        for (auto& kv : rhs.data) ret.data[kv.first] -= kv.second;
        for (auto it = ret.data.begin(); it != ret.data.end();) {
            if (it->second == T()) it = ret.data.erase(it); else ++it;
        }
        return ret;
    }
};

// ==================== 4. 稀疏 Matrix ==================== //
template<typename T>
struct SparseMatrix {
    std::map<size_t, std::map<size_t, T>> data; // row -> (col -> value)
    size_t rows_, cols_;

    SparseMatrix(size_t r, size_t c) : rows_(r), cols_(c) {}
    void set(size_t i, size_t j, T val) { if (val == T()) data[i].erase(j); else data[i][j] = val; }
    T get(size_t i, size_t j) const {
        auto row = data.find(i);
        if (row == data.end()) return T();
        auto col = row->second.find(j);
        return col == row->second.end() ? T() : col->second;
    }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    void clear() { data.clear(); }
    SparseVector<T> operator*(const SparseVector<T>& v) const {
        assert(cols_ == v.size());
        SparseVector<T> ret(rows_);
        for (auto& rowkv : data) {
            T sum = T();
            for (auto& colkv : rowkv.second)
                sum += colkv.second * v.get(colkv.first);
            if (sum != T()) ret.set(rowkv.first, sum);
        }
        return ret;
    }
    SparseMatrix<T> operator*(const SparseMatrix<T>& rhs) const {
        assert(cols_ == rhs.rows_);
        SparseMatrix<T> ret(rows_, rhs.cols_);
        for (auto& rowkv : data) {
            size_t i = rowkv.first;
            for (auto& colkv : rowkv.second) {
                size_t k = colkv.first;
                T a = colkv.second;
                auto rhsrow = rhs.data.find(k);
                if (rhsrow != rhs.data.end()) {
                    for (auto& rkv : rhsrow->second) {
                        size_t j = rkv.first;
                        ret.data[i][j] += a * rkv.second;
                    }
                }
            }
        }
        // remove zeros
        for (auto& row : ret.data)
            for (auto it = row.second.begin(); it != row.second.end();)
                if (it->second == T()) it = row.second.erase(it); else ++it;
        return ret;
    }
    SparseMatrix<T> transpose() const {
        SparseMatrix<T> ret(cols_, rows_);
        for (auto& rowkv : data)
            for (auto& colkv : rowkv.second)
                ret.data[colkv.first][rowkv.first] = colkv.second;
        return ret;
    }
};

// ========== 5. 稀疏张量(3D) ========== //
template<typename T>
struct SparseTensor3D {
    std::map<size_t, std::map<size_t, std::map<size_t, T>>> data;
    size_t d1, d2, d3;
    SparseTensor3D(size_t d1_, size_t d2_, size_t d3_) : d1(d1_), d2(d2_), d3(d3_) {}
    void set(size_t i, size_t j, size_t k, T val) {
        if (val == T())
            data[i][j].erase(k);
        else
            data[i][j][k] = val;
    }
    T get(size_t i, size_t j, size_t k) const {
        auto it1 = data.find(i);
        if (it1 == data.end()) return T();
        auto it2 = it1->second.find(j);
        if (it2 == it1->second.end()) return T();
        auto it3 = it2->second.find(k);
        return it3 == it2->second.end() ? T() : it3->second;
    }
    void clear() { data.clear(); }
    size_t dim1() const { return d1; }
    size_t dim2() const { return d2; }
    size_t dim3() const { return d3; }
};

// ================== 6. 高阶操作 ================== //
template<typename T, typename F>
GenVector<T> vector_map(const GenVector<T>& v, F func) {
    GenVector<T> ret(v.size());
    for (size_t i = 0; i < v.size(); ++i) ret[i] = func(v[i]);
    return ret;
}
template<typename T, typename F>
T vector_reduce(const GenVector<T>& v, T init, F func) {
    T acc = init;
    for (size_t i = 0; i < v.size(); ++i) acc = func(acc, v[i]);
    return acc;
}
template<typename T, typename F>
GenVector<T> vector_filter(const GenVector<T>& v, F pred) {
    GenVector<T> ret;
    for (size_t i = 0; i < v.size(); ++i)
        if (pred(v[i])) ret.data.push_back(v[i]);
    return ret;
}
template<typename T, typename F>
SparseVector<T> sparse_vector_map(const SparseVector<T>& v, F func) {
    SparseVector<T> ret(v.dim);
    for (auto& kv : v.data) {
        T val = func(kv.second);
        if (val != T()) ret.data[kv.first] = val;
    }
    return ret;
}
template<typename T, typename F>
SparseMatrix<T> sparse_matrix_map(const SparseMatrix<T>& m, F func) {
    SparseMatrix<T> ret(m.rows_, m.cols_);
    for (auto& rowkv : m.data)
        for (auto& colkv : rowkv.second) {
            T val = func(colkv.second);
            if (val != T()) ret.data[rowkv.first][colkv.first] = val;
        }
    return ret;
}
template<typename T, typename F>
void vector_apply_inplace(GenVector<T>& v, F func) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = func(v[i]);
}
template<typename T, typename F>
void matrix_apply_inplace(GenMatrix<T>& m, F func) {
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j)
            m[i][j] = func(m[i][j]);
}
template<typename T>
GenVector<T> sparse_mat_dense_vec(const SparseMatrix<T>& sm, const GenVector<T>& v) {
    assert(sm.cols_ == v.size());
    GenVector<T> ret(sm.rows_);
    for (auto& rowkv : sm.data) {
        T sum = T();
        for (auto& colkv : rowkv.second)
            sum += colkv.second * v.data[colkv.first];
        ret.data[rowkv.first] = sum;
    }
    return ret;
}
template<typename T>
GenVector<T> dense_mat_sparse_vec(const GenMatrix<T>& m, const SparseVector<T>& v) {
    assert(m.cols() == v.size());
    GenVector<T> ret(m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        T sum = T();
        for (auto& kv : v.data)
            sum += m.data[i][kv.first] * kv.second;
        ret.data[i] = sum;
    }
    return ret;
}
template<typename T>
GenMatrix<T> sparse_dense_matmul(const SparseMatrix<T>& sm, const GenMatrix<T>& m) {
    assert(sm.cols_ == m.rows());
    GenMatrix<T> ret(sm.rows_, m.cols());
    for (auto& rowkv : sm.data) {
        for (size_t j = 0; j < m.cols(); ++j) {
            T sum = T();
            for (auto& colkv : rowkv.second)
                sum += colkv.second * m.data[colkv.first][j];
            ret.data[rowkv.first][j] = sum;
        }
    }
    return ret;
}
template<typename T>
T sparse_dot(const SparseVector<T>& a, const SparseVector<T>& b) {
    assert(a.size() == b.size());
    T s = T();
    auto itA = a.data.begin(), itB = b.data.begin();
    while (itA != a.data.end() && itB != b.data.end()) {
        if (itA->first < itB->first) ++itA;
        else if (itA->first > itB->first) ++itB;
        else { s += itA->second * itB->second; ++itA; ++itB; }
    }
    return s;
}

// ================== 7. 线性代数专用类型和典型算法实现 ================== //
typedef GenVector<double> Vector;
typedef GenMatrix<double> Matrix;

// Eigen pair
struct EigenPair {
    double eigenvalue;
    Vector eigenvector;
};

// Power Iteration for dominant eigenvalue/vector
inline EigenPair power_iteration(const Matrix &A, int max_iter = 1000, double tol = 1e-8) {
    assert(A.is_square());
    size_t n = A.rows();
    Vector b_k(n, 1.0 / n);
    for (int it = 0; it < max_iter; ++it) {
        Vector b_k1 = A * b_k;
        double norm = b_k1.norm();
        for (size_t i = 0; i < n; ++i) b_k1[i] /= norm;
        if ((b_k1 - b_k).norm() < tol) return { (A * b_k1).dot(b_k1), b_k1 };
        b_k = b_k1;
    }
    return { (A * b_k).dot(b_k), b_k };
}

// ================== 8. 概率统计相关实现 ================== //
struct DiscreteDist {
    std::vector<double> probs;
    DiscreteDist() {}
    explicit DiscreteDist(const std::vector<double>& p) : probs(p) {}
    explicit DiscreteDist(size_t n, double val = 0.0) : probs(n, val) {}
    void normalize() {
        double s = std::accumulate(probs.begin(), probs.end(), 0.0);
        for (size_t i = 0; i < probs.size(); ++i) probs[i] /= s;
    }
    size_t argmax() const {
        return std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    }
    double entropy() const {
        double e = 0;
        for (auto x : probs) if (x > 0) e -= x * std::log(x);
        return e;
    }
    size_t sample(std::mt19937& rng) const {
        std::discrete_distribution<size_t> d(probs.begin(), probs.end());
        return d(rng);
    }
    bool valid(double epsilon = 1e-8) const {
        double s = std::accumulate(probs.begin(), probs.end(), 0.0);
        return std::abs(s - 1.0) < epsilon;
    }
};

inline Vector softmax(const Vector &x) {
    double maxval = *std::max_element(x.data.begin(), x.data.end());
    Vector exps(x.size());
    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i) sum += (exps[i] = std::exp(x[i] - maxval));
    for (size_t i = 0; i < x.size(); ++i) exps[i] /= sum;
    return exps;
}
inline double cross_entropy(const Vector &p, const Vector &q) {
    assert(p.size() == q.size());
    double ce = 0;
    for (size_t i = 0; i < p.size(); ++i)
        ce -= p[i] * std::log(q[i] + 1e-12);
    return ce;
}
inline double normal_pdf(double x, double mu = 0, double sigma = 1) {
    double z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2 * M_PI));
}
struct Gaussian1D {
    double mu, sigma2;
    Gaussian1D(double m = 0.0, double s2 = 1.0) : mu(m), sigma2(s2) {}
};
inline Gaussian1D bayes_gaussian_update(const Gaussian1D& prior, double obs, double obs_sigma2) {
    double post_sigma2 = 1.0 / (1.0 / prior.sigma2 + 1.0 / obs_sigma2);
    double post_mu = post_sigma2 * (prior.mu / prior.sigma2 + obs / obs_sigma2);
    return Gaussian1D(post_mu, post_sigma2);
}
inline double mle_mean(const Vector &x) {
    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i) sum += x[i];
    return sum / x.size();
}

// ================== 9. 数值优化典型算法实现 ================== //
struct GDResult {
    double x_min, f_min;
    int steps;
};
inline GDResult gradient_descent(std::function<double(double)> f, std::function<double(double)> df, double x0, double lr=0.01, int max_iter=1000, double tol=1e-8) {
    double x = x0;
    for (int i = 0; i < max_iter; ++i) {
        double g = df(x);
        double x_new = x - lr * g;
        if (std::abs(x_new - x) < tol) return {x_new, f(x_new), i+1};
        x = x_new;
    }
    return {x, f(x), max_iter};
}
struct SGDResult {
    Vector x_min;
    int steps;
};
inline SGDResult sgd(std::function<Vector(const Vector&)> grad_f, Vector x0, double lr=0.1, int max_iter=100) {
    Vector x = x0;
    for (int i = 0; i < max_iter; ++i) {
        Vector g = grad_f(x);
        for (size_t j = 0; j < x.size(); ++j) x[j] -= lr * g[j];
    }
    return {x, max_iter};
}
struct AdamResult {
    Vector x_min;
    int steps;
};
inline AdamResult adam(std::function<Vector(const Vector&)> grad_f, Vector x0, int max_iter = 1000, double lr=0.001, double beta1=0.9, double beta2=0.999, double eps=1e-8) {
    Vector x = x0, m(x.size(), 0.0), v(x.size(), 0.0);
    for (int i = 1; i <= max_iter; ++i) {
        Vector g = grad_f(x);
        for (size_t j = 0; j < x.size(); ++j) {
            m[j] = beta1 * m[j] + (1 - beta1) * g[j];
            v[j] = beta2 * v[j] + (1 - beta2) * g[j] * g[j];
            double m_hat = m[j] / (1 - std::pow(beta1, i));
            double v_hat = v[j] / (1 - std::pow(beta2, i));
            x[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
    return {x, max_iter};
}

// ================== 10. 简单MLP和反向传播 ================== //
struct SimpleMLP {
    Matrix W1, W2;
    Vector b1, b2;
    size_t in_dim, hidden_dim, out_dim;
    SimpleMLP(size_t in_dim_, size_t hidden, size_t out_dim_)
        : W1(hidden, in_dim_, 0.1), W2(out_dim_, hidden, 0.1),
          b1(hidden, 0.0), b2(out_dim_, 0.0),
          in_dim(in_dim_), hidden_dim(hidden), out_dim(out_dim_) {}
    Vector forward(const Vector& x) const {
        assert(x.size() == in_dim);
        Vector h(hidden_dim);
        for (size_t i = 0; i < hidden_dim; ++i) {
            h[i] = b1[i];
            for (size_t j = 0; j < in_dim; ++j) h[i] += W1[i][j] * x[j];
            h[i] = std::tanh(h[i]);
        }
        Vector y(out_dim);
        for (size_t i = 0; i < out_dim; ++i) {
            y[i] = b2[i];
            for (size_t j = 0; j < hidden_dim; ++j) y[i] += W2[i][j] * h[j];
        }
        return y;
    }
    void backward(const Vector& x, const Vector& y_true, double lr = 0.01) {
        // Forward
        Vector h(hidden_dim), h_raw(hidden_dim), y(out_dim);
        for (size_t i = 0; i < hidden_dim; ++i) {
            h_raw[i] = b1[i];
            for (size_t j = 0; j < in_dim; ++j) h_raw[i] += W1[i][j] * x[j];
            h[i] = std::tanh(h_raw[i]);
        }
        for (size_t i = 0; i < out_dim; ++i) {
            y[i] = b2[i];
            for (size_t j = 0; j < hidden_dim; ++j) y[i] += W2[i][j] * h[j];
        }
        // Loss = MSE
        Vector dy(out_dim);
        for (size_t i = 0; i < out_dim; ++i) dy[i] = 2.0 * (y[i] - y_true[i]);
        // Backprop
        std::vector<std::vector<double>> dW2(out_dim, std::vector<double>(hidden_dim, 0.0));
        Vector db2(out_dim, 0.0);
        Vector dh(hidden_dim, 0.0);
        for (size_t i = 0; i < out_dim; ++i) {
            db2[i] = dy[i];
            for (size_t j = 0; j < hidden_dim; ++j) {
                dW2[i][j] = dy[i] * h[j];
                dh[j] += dy[i] * W2[i][j];
            }
        }
        Vector dh_raw(hidden_dim, 0.0);
        for (size_t j = 0; j < hidden_dim; ++j)
            dh_raw[j] = dh[j] * (1 - std::tanh(h_raw[j]) * std::tanh(h_raw[j]));
        std::vector<std::vector<double>> dW1(hidden_dim, std::vector<double>(in_dim, 0.0));
        Vector db1(hidden_dim, 0.0);
        for (size_t i = 0; i < hidden_dim; ++i) {
            db1[i] = dh_raw[i];
            for (size_t j = 0; j < in_dim; ++j)
                dW1[i][j] = dh_raw[i] * x[j];
        }
        // Gradient step
        for (size_t i = 0; i < out_dim; ++i) {
            b2[i] -= lr * db2[i];
            for (size_t j = 0; j < hidden_dim; ++j)
                W2[i][j] -= lr * dW2[i][j];
        }
        for (size_t i = 0; i < hidden_dim; ++i) {
            b1[i] -= lr * db1[i];
            for (size_t j = 0; j < in_dim; ++j)
                W1[i][j] -= lr * dW1[i][j];
        }
    }
    void zero_grad() {}
    void random_init(std::mt19937& rng, double weight_scale = 0.1) {
        std::normal_distribution<double> nd(0.0, weight_scale);
        for (size_t i = 0; i < hidden_dim; ++i)
            for (size_t j = 0; j < in_dim; ++j)
                W1[i][j] = nd(rng);
        for (size_t i = 0; i < out_dim; ++i)
            for (size_t j = 0; j < hidden_dim; ++j)
                W2[i][j] = nd(rng);
        for (size_t i = 0; i < hidden_dim; ++i) b1[i] = nd(rng);
        for (size_t i = 0; i < out_dim; ++i) b2[i] = nd(rng);
    }
};

// ================== 11. 信息论 ================== //
inline double entropy(const Vector &p) {
    double e = 0;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0) e -= p[i] * std::log(p[i]);
    return e;
}
inline double kl_divergence(const Vector &p, const Vector &q) {
    assert(p.size() == q.size());
    double kl = 0;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0) kl += p[i] * std::log((p[i] + 1e-12) / (q[i] + 1e-12));
    return kl;
}

// ================== 12. 注意力/Transformer ================== //
struct AttentionResult {
    Matrix weights;
    Matrix output;
};
inline AttentionResult attention(const Matrix &Q, const Matrix &K, const Matrix &V, double scale = 1.0) {
    size_t n = Q.rows(), d = Q.cols();
    assert(K.cols() == d && V.rows() == K.rows());
    Matrix scores(n, K.rows());
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < K.rows(); ++j)
            for (size_t k = 0; k < d; ++k)
                scores[i][j] += Q[i][k] * K[j][k];
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < K.rows(); ++j)
            scores[i][j] *= scale;
    Matrix attn(n, K.rows());
    for (size_t i = 0; i < n; ++i) {
        double maxv = *std::max_element(scores[i].begin(), scores[i].end());
        double sum = 0;
        for (size_t j = 0; j < K.rows(); ++j) sum += (attn[i][j] = std::exp(scores[i][j] - maxv));
        for (size_t j = 0; j < K.rows(); ++j) attn[i][j] /= sum;
    }
    Matrix o(n, V.cols());
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < V.cols(); ++j)
            for (size_t k = 0; k < V.rows(); ++k)
                o[i][j] += attn[i][k] * V[k][j];
    return {attn, o};
}
struct SimpleTransformer {
    size_t dim;
    SimpleTransformer(size_t d) : dim(d) {}
    Matrix self_attention(const Matrix &X) const { return attention(X, X, X, 1.0 / std::sqrt(dim)).output; }
    Matrix mlp(const Matrix &X) const { return X; }
    Matrix forward(const Matrix &X) const { return mlp(self_attention(X)); }
};

// ================== 13. FSM/Markov ================== //
struct FSM {
    typedef std::pair<std::string, char> TransKey;
    std::map<TransKey, std::string> transitions;
    std::string state;
    FSM(const std::string& start) : state(start) {}
    void add_transition(const std::string& from, char symbol, const std::string& to) { transitions[TransKey(from, symbol)] = to; }
    void reset(const std::string& s) { state = s; }
    std::string step(char symbol) {
        auto it = transitions.find(TransKey(state, symbol));
        if (it != transitions.end()) state = it->second;
        return state;
    }
    std::string current_state() const { return state; }
    bool accepts(const std::string& input) const {
        std::string s = state;
        for (char c : input) {
            auto it = transitions.find(TransKey(s, c));
            if (it == transitions.end()) return false;
            s = it->second;
        }
        return true;
    }
};
struct MarkovChain {
    std::map<std::string, std::map<std::string, double>> trans_prob;
    void add_transition(const std::string &from, const std::string &to, double prob) { trans_prob[from][to] = prob; }
    std::string next(const std::string &from, double rnd) const {
        auto it = trans_prob.find(from);
        if (it == trans_prob.end()) return "";
        double s = 0;
        for (auto& kv : it->second) {
            s += kv.second;
            if (rnd < s) return kv.first;
        }
        return "";
    }
    std::string sample_next(const std::string &from, std::mt19937& rng) const {
        auto it = trans_prob.find(from);
        if (it == trans_prob.end()) return "";
        std::vector<std::string> keys;
        std::vector<double> weights;
        for (auto& kv : it->second) { keys.push_back(kv.first); weights.push_back(kv.second); }
        std::discrete_distribution<size_t> d(weights.begin(), weights.end());
        return keys[d(rng)];
    }
    bool valid(double epsilon = 1e-8) const {
        for (auto& kv : trans_prob) {
            double s = 0;
            for (auto& x : kv.second) s += x.second;
            if (std::abs(s - 1.0) > epsilon) return false;
        }
        return true;
    }
};

// ================== 14. 微积分 ================== //
inline double chain_rule(double x, std::function<double(double)> f, std::function<double(double)> g) {
    double df = (f(g(x) + 1e-8) - f(g(x) - 1e-8)) / 2e-8;
    double dg = (g(x + 1e-8) - g(x - 1e-8)) / 2e-8;
    return df * dg;
}
inline double numerical_derivative(std::function<double(double)> f, double x, double h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2 * h);
}
struct MinResult {
    double x_min, f_min;
};
inline MinResult find_minimum(std::function<double(double)> f, double a, double b, int steps = 1000) {
    double minx = a, minv = f(a);
    double step = (b - a) / steps;
    for (int i = 1; i <= steps; ++i) {
        double x = a + i * step;
        double v = f(x);
        if (v < minv) { minx = x; minv = v; }
    }
    return {minx, minv};
}

// ================== 15. 张量运算（稠密）================== //
struct Tensor3D {
    std::vector<std::vector<std::vector<double>>> data;
    Tensor3D() {}
    Tensor3D(size_t d1, size_t d2, size_t d3, double val = 0.0)
        : data(d1, std::vector<std::vector<double>>(d2, std::vector<double>(d3, val))) {}
    size_t dim1() const { return data.size(); }
    size_t dim2() const { return dim1() ? data[0].size() : 0; }
    size_t dim3() const { return (dim2() && dim1()) ? data[0][0].size() : 0; }
    void resize(size_t d1, size_t d2, size_t d3, double val = 0.0) {
        data.resize(d1, std::vector<std::vector<double>>(d2, std::vector<double>(d3, val)));
        for (size_t i = 0; i < d1; ++i) {
            data[i].resize(d2, std::vector<double>(d3, val));
            for (size_t j = 0; j < d2; ++j)
                data[i][j].resize(d3, val);
        }
    }
    std::vector<double>& operator()(size_t i, size_t j) { return data[i][j]; }
    const std::vector<double>& operator()(size_t i, size_t j) const { return data[i][j]; }
    double& at(size_t i, size_t j, size_t k) { return data[i][j][k]; }
    const double& at(size_t i, size_t j, size_t k) const { return data[i][j][k]; }
    Tensor3D operator+(const Tensor3D& rhs) const {
        assert(dim1() == rhs.dim1() && dim2() == rhs.dim2() && dim3() == rhs.dim3());
        Tensor3D ret(dim1(), dim2(), dim3());
        for (size_t i = 0; i < dim1(); ++i)
            for (size_t j = 0; j < dim2(); ++j)
                for (size_t k = 0; k < dim3(); ++k)
                    ret.data[i][j][k] = data[i][j][k] + rhs.data[i][j][k];
        return ret;
    }
    Tensor3D operator*(double scalar) const {
        Tensor3D ret(dim1(), dim2(), dim3());
        for (size_t i = 0; i < dim1(); ++i)
            for (size_t j = 0; j < dim2(); ++j)
                for (size_t k = 0; k < dim3(); ++k)
                    ret.data[i][j][k] = data[i][j][k] * scalar;
        return ret;
    }
    void fill(double val) {
        for (size_t i = 0; i < dim1(); ++i)
            for (size_t j = 0; j < dim2(); ++j)
                std::fill(data[i][j].begin(), data[i][j].end(), val);
    }
    void zero() { fill(0.0); }
};

// ================== 16. 统计学习判别 ================== //
inline double empirical_risk(const std::vector<Vector>& y_true, const std::vector<Vector>& y_pred, std::function<double(const Vector&, const Vector&)> loss_fn) {
    assert(y_true.size() == y_pred.size());
    double sum = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
        sum += loss_fn(y_true[i], y_pred[i]);
    return sum / y_true.size();
}
enum class FitStatus { UNDERFITTING, GOOD_FIT, OVERFITTING };
inline FitStatus fit_analysis(double train_loss, double val_loss, double tol = 0.1) {
    if (train_loss > val_loss + tol) return FitStatus::UNDERFITTING;
    if (val_loss > train_loss + tol) return FitStatus::OVERFITTING;
    return FitStatus::GOOD_FIT;
}

} // namespace llm_math
//```
//**说明：**
//- 此实现整合了泛型稠密/稀疏向量、矩阵、张量等数据结构，及其高阶、混合、函数式、微分、概率、优化等工具函数。
//- 提供了现代深度学习/统计/图模型/离散结构/信息论/泛型数值工具的高可读性和扩展性C++11接口，适合嵌入LLM底层数学核心。
//- 可直接包含此头文件使用（无需分cpp），支持单头文件风格。
