#ifndef DUCKT_AUTOGRAD_HPP
#define DUCKT_AUTOGRAD_HPP

#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iostream>

namespace duckt {

// Forward declaration
struct GradTensor;
using GradTensorPtr = std::shared_ptr<GradTensor>;

// ============ Gradient Tensor with Autograd ============

struct GradTensor : std::enable_shared_from_this<GradTensor> {
    std::vector<double> data;
    std::vector<int> shape;

    // Gradient tracking
    std::vector<double> grad;
    bool requires_grad = false;

    // Computation graph
    std::vector<GradTensorPtr> parents;
    std::function<void()> backward_fn;

    GradTensor() = default;

    GradTensor(std::vector<double> d, std::vector<int> s, bool req_grad = false)
        : data(std::move(d)), shape(std::move(s)), requires_grad(req_grad) {
        if (requires_grad) {
            grad.resize(data.size(), 0.0);
        }
    }

    static GradTensorPtr create(std::vector<double> d, std::vector<int> s, bool req_grad = false) {
        return std::make_shared<GradTensor>(std::move(d), std::move(s), req_grad);
    }

    static GradTensorPtr zeros(const std::vector<int>& shape, bool req_grad = false) {
        int size = 1;
        for (int dim : shape) size *= dim;
        return create(std::vector<double>(size, 0.0), shape, req_grad);
    }

    static GradTensorPtr ones(const std::vector<int>& shape, bool req_grad = false) {
        int size = 1;
        for (int dim : shape) size *= dim;
        return create(std::vector<double>(size, 1.0), shape, req_grad);
    }

    static GradTensorPtr rand(const std::vector<int>& shape, bool req_grad = false) {
        int size = 1;
        for (int dim : shape) size *= dim;
        std::vector<double> data(size);
        for (int i = 0; i < size; i++) {
            data[i] = static_cast<double>(std::rand()) / RAND_MAX;
        }
        return create(std::move(data), shape, req_grad);
    }

    static GradTensorPtr randn(const std::vector<int>& shape, bool req_grad = false) {
        // Box-Muller transform for normal distribution
        int size = 1;
        for (int dim : shape) size *= dim;
        std::vector<double> data(size);
        for (int i = 0; i < size; i += 2) {
            double u1 = static_cast<double>(std::rand()) / RAND_MAX;
            double u2 = static_cast<double>(std::rand()) / RAND_MAX;
            if (u1 < 1e-10) u1 = 1e-10;
            double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            double z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * M_PI * u2);
            data[i] = z0;
            if (i + 1 < size) data[i + 1] = z1;
        }
        return create(std::move(data), shape, req_grad);
    }

    int totalSize() const {
        int size = 1;
        for (int dim : shape) size *= dim;
        return size;
    }

    void zeroGrad() {
        std::fill(grad.begin(), grad.end(), 0.0);
    }

    // Backward pass - compute gradients through the graph
    void backward() {
        if (!requires_grad) return;

        // Initialize gradient to 1 if this is the loss (scalar output)
        if (grad.empty()) {
            grad.resize(data.size(), 1.0);
        } else if (std::all_of(grad.begin(), grad.end(), [](double v) { return v == 0; })) {
            // Grad exists but is all zeros - set to ones for the loss
            std::fill(grad.begin(), grad.end(), 1.0);
        }

        // Topological sort
        std::vector<GradTensor*> order;
        std::unordered_set<GradTensor*> visited;

        std::function<void(GradTensor*)> topo_sort = [&](GradTensor* t) {
            if (visited.count(t)) return;
            visited.insert(t);
            for (auto& parent : t->parents) {
                if (parent && parent->requires_grad) {
                    topo_sort(parent.get());
                }
            }
            order.push_back(t);
        };

        topo_sort(this);

        // Reverse topological order
        std::reverse(order.begin(), order.end());

        // Backpropagate
        for (GradTensor* t : order) {
            if (t->backward_fn) {
                t->backward_fn();
            }
        }
    }

    // ============ Operations with gradient tracking ============

    // Addition
    GradTensorPtr add(const GradTensorPtr& other) {
        if (shape != other->shape) {
            throw std::runtime_error("Shape mismatch for addition");
        }

        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = data[i] + other->data[i];
        }

        bool result_req_grad = requires_grad || other->requires_grad;
        auto result = create(std::move(result_data), shape, result_req_grad);

        if (result_req_grad) {
            auto self_ptr = shared_from_this();
            result->parents = {self_ptr, other};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, other]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient flows equally to both inputs
                if (self_ptr->requires_grad) {
                    for (size_t i = 0; i < res->grad.size(); i++) {
                        self_ptr->grad[i] += res->grad[i];
                    }
                }
                if (other->requires_grad) {
                    for (size_t i = 0; i < res->grad.size(); i++) {
                        other->grad[i] += res->grad[i];
                    }
                }
            };
        }

        return result;
    }

    // Subtraction
    GradTensorPtr sub(const GradTensorPtr& other) {
        if (shape != other->shape) {
            throw std::runtime_error("Shape mismatch for subtraction");
        }

        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = data[i] - other->data[i];
        }

        bool result_req_grad = requires_grad || other->requires_grad;
        auto result = create(std::move(result_data), shape, result_req_grad);

        if (result_req_grad) {
            auto self_ptr = shared_from_this();
            result->parents = {self_ptr, other};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, other]() {
                auto res = result_weak.lock();
                if (!res) return;

                if (self_ptr->requires_grad) {
                    for (size_t i = 0; i < res->grad.size(); i++) {
                        self_ptr->grad[i] += res->grad[i];
                    }
                }
                if (other->requires_grad) {
                    for (size_t i = 0; i < res->grad.size(); i++) {
                        other->grad[i] -= res->grad[i];
                    }
                }
            };
        }

        return result;
    }

    // Element-wise multiplication
    GradTensorPtr mul(const GradTensorPtr& other) {
        if (shape != other->shape) {
            throw std::runtime_error("Shape mismatch for multiplication");
        }

        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = data[i] * other->data[i];
        }

        bool result_req_grad = requires_grad || other->requires_grad;
        auto result = create(std::move(result_data), shape, result_req_grad);

        if (result_req_grad) {
            auto self_ptr = shared_from_this();
            // Save copies of data for backward
            std::vector<double> self_data = data;
            std::vector<double> other_data = other->data;

            result->parents = {self_ptr, other};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, other, self_data, other_data]() {
                auto res = result_weak.lock();
                if (!res) return;

                // d(a*b)/da = b, d(a*b)/db = a
                if (self_ptr->requires_grad) {
                    for (size_t i = 0; i < res->grad.size(); i++) {
                        self_ptr->grad[i] += res->grad[i] * other_data[i];
                    }
                }
                if (other->requires_grad) {
                    for (size_t i = 0; i < res->grad.size(); i++) {
                        other->grad[i] += res->grad[i] * self_data[i];
                    }
                }
            };
        }

        return result;
    }

    // Scalar multiplication
    GradTensorPtr mulScalar(double scalar) {
        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = data[i] * scalar;
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, scalar]() {
                auto res = result_weak.lock();
                if (!res) return;

                for (size_t i = 0; i < res->grad.size(); i++) {
                    self_ptr->grad[i] += res->grad[i] * scalar;
                }
            };
        }

        return result;
    }

    // Matrix multiplication
    GradTensorPtr matmul(const GradTensorPtr& other) {
        // Handle 1D vectors by promoting to 2D
        GradTensorPtr left = shared_from_this();
        GradTensorPtr right = other;
        bool leftWas1D = false;
        bool rightWas1D = false;

        std::vector<double> left_data = data;
        std::vector<int> left_shape = shape;
        std::vector<double> right_data = other->data;
        std::vector<int> right_shape = other->shape;

        if (left_shape.size() == 1) {
            leftWas1D = true;
            left_shape = {1, left_shape[0]};
        }
        if (right_shape.size() == 1) {
            rightWas1D = true;
            right_shape = {right_shape[0], 1};
        }

        if (left_shape.size() != 2 || right_shape.size() != 2) {
            throw std::runtime_error("Matrix multiplication requires 1D or 2D tensors");
        }
        if (left_shape[1] != right_shape[0]) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }

        int m = left_shape[0];
        int n = left_shape[1];
        int p = right_shape[1];

        std::vector<double> result_data(m * p, 0.0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    result_data[i * p + j] += left_data[i * n + k] * right_data[k * p + j];
                }
            }
        }

        // Determine output shape
        std::vector<int> result_shape;
        if (leftWas1D && rightWas1D) {
            result_shape = {1};
        } else if (leftWas1D) {
            result_shape = {p};
        } else if (rightWas1D) {
            result_shape = {m};
        } else {
            result_shape = {m, p};
        }

        bool result_req_grad = requires_grad || other->requires_grad;
        auto result = create(std::move(result_data), result_shape, result_req_grad);

        if (result_req_grad) {
            result->parents = {left, right};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   left, right, m, n, p, leftWas1D, rightWas1D,
                                   left_shape, right_shape]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Reshape grad if needed
                std::vector<double> grad_2d = res->grad;
                if (leftWas1D && rightWas1D) {
                    // grad is [1], but we need [1, 1] behavior
                } else if (leftWas1D) {
                    // grad is [p], reshape to [1, p]
                } else if (rightWas1D) {
                    // grad is [m], reshape to [m, 1]
                }

                // dL/dA = dL/dC @ B^T
                if (left->requires_grad) {
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            double sum = 0;
                            for (int k = 0; k < p; k++) {
                                int grad_idx = i * p + k;
                                if (grad_idx < (int)res->grad.size()) {
                                    sum += res->grad[grad_idx] * right->data[j * p + k];
                                }
                            }
                            int left_idx = i * n + j;
                            if (left_idx < (int)left->grad.size()) {
                                left->grad[left_idx] += sum;
                            }
                        }
                    }
                }

                // dL/dB = A^T @ dL/dC
                if (right->requires_grad) {
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < p; j++) {
                            double sum = 0;
                            for (int k = 0; k < m; k++) {
                                int grad_idx = k * p + j;
                                if (grad_idx < (int)res->grad.size()) {
                                    sum += left->data[k * n + i] * res->grad[grad_idx];
                                }
                            }
                            int right_idx = i * p + j;
                            if (right_idx < (int)right->grad.size()) {
                                right->grad[right_idx] += sum;
                            }
                        }
                    }
                }
            };
        }

        return result;
    }

    // Sum reduction
    GradTensorPtr sum() {
        double total = 0;
        for (double v : data) total += v;

        auto result = create({total}, {1}, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient of sum is 1 for all elements
                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    self_ptr->grad[i] += res->grad[0];
                }
            };
        }

        return result;
    }

    // Mean reduction
    GradTensorPtr mean() {
        double total = 0;
        for (double v : data) total += v;
        double avg = total / data.size();

        auto result = create({avg}, {1}, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            int n = data.size();
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, n]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient of mean is 1/n for all elements
                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    self_ptr->grad[i] += res->grad[0] / n;
                }
            };
        }

        return result;
    }

    // ReLU activation
    GradTensorPtr relu() {
        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = data[i] > 0 ? data[i] : 0;
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            std::vector<double> saved_data = data;
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, saved_data]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient of ReLU: 1 if x > 0, else 0
                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    if (saved_data[i] > 0) {
                        self_ptr->grad[i] += res->grad[i];
                    }
                }
            };
        }

        return result;
    }

    // Sigmoid activation
    GradTensorPtr sigmoid() {
        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = 1.0 / (1.0 + std::exp(-data[i]));
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            std::vector<double> sigmoid_output = result->data;
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, sigmoid_output]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    double s = sigmoid_output[i];
                    self_ptr->grad[i] += res->grad[i] * s * (1 - s);
                }
            };
        }

        return result;
    }

    // Tanh activation
    GradTensorPtr tanh_() {
        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = std::tanh(data[i]);
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            std::vector<double> tanh_output = result->data;
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, tanh_output]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient of tanh: 1 - tanh(x)^2
                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    double t = tanh_output[i];
                    self_ptr->grad[i] += res->grad[i] * (1 - t * t);
                }
            };
        }

        return result;
    }

    // Softmax activation
    GradTensorPtr softmax() {
        double max_val = *std::max_element(data.begin(), data.end());
        double sum_exp = 0;
        for (double v : data) {
            sum_exp += std::exp(v - max_val);
        }

        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = std::exp(data[i] - max_val) / sum_exp;
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            std::vector<double> softmax_output = result->data;
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, softmax_output]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Jacobian of softmax: diag(s) - s * s^T
                // dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
                int n = softmax_output.size();
                for (int i = 0; i < n; i++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) {
                        double jacobian = (i == j) ?
                            softmax_output[i] * (1 - softmax_output[j]) :
                            -softmax_output[i] * softmax_output[j];
                        sum += res->grad[j] * jacobian;
                    }
                    self_ptr->grad[i] += sum;
                }
            };
        }

        return result;
    }

    // Log (for cross-entropy)
    GradTensorPtr log_() {
        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = std::log(data[i] + 1e-10);  // Add small value for stability
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            std::vector<double> saved_data = data;
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, saved_data]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient of log: 1/x
                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    self_ptr->grad[i] += res->grad[i] / (saved_data[i] + 1e-10);
                }
            };
        }

        return result;
    }

    // Negative (for loss computation)
    GradTensorPtr neg() {
        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = -data[i];
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr]() {
                auto res = result_weak.lock();
                if (!res) return;

                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    self_ptr->grad[i] -= res->grad[i];
                }
            };
        }

        return result;
    }

    // Square (for MSE loss)
    GradTensorPtr square() {
        std::vector<double> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = data[i] * data[i];
        }

        auto result = create(std::move(result_data), shape, requires_grad);

        if (requires_grad) {
            auto self_ptr = shared_from_this();
            std::vector<double> saved_data = data;
            result->parents = {self_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   self_ptr, saved_data]() {
                auto res = result_weak.lock();
                if (!res) return;

                // Gradient of x^2: 2x
                for (size_t i = 0; i < self_ptr->grad.size(); i++) {
                    self_ptr->grad[i] += res->grad[i] * 2 * saved_data[i];
                }
            };
        }

        return result;
    }
};

// ============ Loss Functions ============

// Mean Squared Error Loss
inline GradTensorPtr mse_loss(const GradTensorPtr& pred, const GradTensorPtr& target) {
    auto diff = pred->sub(target);
    auto squared = diff->square();
    return squared->mean();
}

// Cross Entropy Loss (for classification)
// pred: output of softmax, target: one-hot encoded
inline GradTensorPtr cross_entropy_loss(const GradTensorPtr& pred, const GradTensorPtr& target) {
    auto log_pred = pred->log_();
    auto prod = log_pred->mul(target);
    auto sum = prod->sum();
    return sum->neg();
}

// ============ Optimizers ============

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zeroGrad() = 0;
};

// Stochastic Gradient Descent
class SGD : public Optimizer {
public:
    std::vector<GradTensorPtr> params;
    double lr;
    double momentum;
    std::vector<std::vector<double>> velocities;

    SGD(std::vector<GradTensorPtr> p, double learning_rate, double mom = 0.0)
        : params(std::move(p)), lr(learning_rate), momentum(mom) {
        if (momentum > 0) {
            for (const auto& param : params) {
                velocities.push_back(std::vector<double>(param->data.size(), 0.0));
            }
        }
    }

    void step() override {
        for (size_t i = 0; i < params.size(); i++) {
            auto& param = params[i];
            if (!param->requires_grad) continue;

            if (momentum > 0) {
                for (size_t j = 0; j < param->data.size(); j++) {
                    velocities[i][j] = momentum * velocities[i][j] + param->grad[j];
                    param->data[j] -= lr * velocities[i][j];
                }
            } else {
                for (size_t j = 0; j < param->data.size(); j++) {
                    param->data[j] -= lr * param->grad[j];
                }
            }
        }
    }

    void zeroGrad() override {
        for (auto& param : params) {
            param->zeroGrad();
        }
    }
};

// Adam Optimizer
class Adam : public Optimizer {
public:
    std::vector<GradTensorPtr> params;
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::vector<std::vector<double>> m;  // First moment
    std::vector<std::vector<double>> v;  // Second moment

    Adam(std::vector<GradTensorPtr> p, double learning_rate = 0.001,
         double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : params(std::move(p)), lr(learning_rate), beta1(b1), beta2(b2),
          epsilon(eps), t(0) {
        for (const auto& param : params) {
            m.push_back(std::vector<double>(param->data.size(), 0.0));
            v.push_back(std::vector<double>(param->data.size(), 0.0));
        }
    }

    void step() override {
        t++;
        for (size_t i = 0; i < params.size(); i++) {
            auto& param = params[i];
            if (!param->requires_grad) continue;

            for (size_t j = 0; j < param->data.size(); j++) {
                // Update biased first moment estimate
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * param->grad[j];
                // Update biased second raw moment estimate
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * param->grad[j] * param->grad[j];

                // Compute bias-corrected first moment estimate
                double m_hat = m[i][j] / (1 - std::pow(beta1, t));
                // Compute bias-corrected second raw moment estimate
                double v_hat = v[i][j] / (1 - std::pow(beta2, t));

                // Update parameters
                param->data[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }

    void zeroGrad() override {
        for (auto& param : params) {
            param->zeroGrad();
        }
    }
};

// ============ Neural Network Layer with Autograd ============

struct GradLayer {
    std::string name;
    GradTensorPtr weights;
    GradTensorPtr bias;

    GradLayer(const std::string& n, int in_features, int out_features) : name(n) {
        // Xavier initialization
        double scale = std::sqrt(2.0 / (in_features + out_features));
        weights = GradTensor::randn({in_features, out_features}, true);
        for (double& v : weights->data) v *= scale;

        bias = GradTensor::zeros({out_features}, true);
    }

    GradTensorPtr forward(const GradTensorPtr& input) {
        // For simplicity, work with 1D tensors directly
        // y = x @ W + b where x is [in_features], W is [in, out], b is [out]

        int in_features = weights->shape[0];
        int out_features = weights->shape[1];

        // Verify input size
        if (input->totalSize() != in_features) {
            throw std::runtime_error("Input size mismatch in layer " + name);
        }

        // Compute x @ W
        std::vector<double> result_data(out_features, 0.0);
        for (int j = 0; j < out_features; j++) {
            for (int i = 0; i < in_features; i++) {
                result_data[j] += input->data[i] * weights->data[i * out_features + j];
            }
            // Add bias
            result_data[j] += bias->data[j];
        }

        bool result_req_grad = input->requires_grad || weights->requires_grad || bias->requires_grad;
        auto result = GradTensor::create(std::move(result_data), {out_features}, result_req_grad);

        if (result_req_grad) {
            auto input_ptr = input;
            auto weights_ptr = weights;
            auto bias_ptr = bias;

            // Save input data for backward
            std::vector<double> saved_input = input->data;

            result->parents = {input_ptr, weights_ptr, bias_ptr};
            result->backward_fn = [result_weak = std::weak_ptr<GradTensor>(result),
                                   input_ptr, weights_ptr, bias_ptr,
                                   saved_input, in_features, out_features]() {
                auto res = result_weak.lock();
                if (!res) return;

                // dL/dx = dL/dy @ W^T
                if (input_ptr->requires_grad) {
                    for (int i = 0; i < in_features; i++) {
                        for (int j = 0; j < out_features; j++) {
                            input_ptr->grad[i] += res->grad[j] * weights_ptr->data[i * out_features + j];
                        }
                    }
                }

                // dL/dW = x^T @ dL/dy (outer product)
                if (weights_ptr->requires_grad) {
                    for (int i = 0; i < in_features; i++) {
                        for (int j = 0; j < out_features; j++) {
                            weights_ptr->grad[i * out_features + j] += saved_input[i] * res->grad[j];
                        }
                    }
                }

                // dL/db = dL/dy
                if (bias_ptr->requires_grad) {
                    for (int j = 0; j < out_features; j++) {
                        bias_ptr->grad[j] += res->grad[j];
                    }
                }
            };
        }

        return result;
    }

    std::vector<GradTensorPtr> parameters() {
        return {weights, bias};
    }
};

// ============ Simple Neural Network ============

struct GradNetwork {
    std::vector<std::shared_ptr<GradLayer>> layers;
    std::string activation;  // "relu", "sigmoid", "tanh"

    GradNetwork(const std::vector<int>& layer_sizes, const std::string& act = "relu")
        : activation(act) {
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            layers.push_back(std::make_shared<GradLayer>(
                "layer_" + std::to_string(i),
                layer_sizes[i],
                layer_sizes[i + 1]
            ));
        }
    }

    GradTensorPtr forward(const GradTensorPtr& input) {
        GradTensorPtr x = input;
        for (size_t i = 0; i < layers.size(); i++) {
            x = layers[i]->forward(x);
            // Apply activation (except for last layer)
            if (i < layers.size() - 1) {
                if (activation == "relu") {
                    x = x->relu();
                } else if (activation == "sigmoid") {
                    x = x->sigmoid();
                } else if (activation == "tanh") {
                    x = x->tanh_();
                }
            }
        }
        return x;
    }

    std::vector<GradTensorPtr> parameters() {
        std::vector<GradTensorPtr> params;
        for (auto& layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

} // namespace duckt

#endif // DUCKT_AUTOGRAD_HPP
