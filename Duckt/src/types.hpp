#ifndef DUCKT_TYPES_HPP
#define DUCKT_TYPES_HPP

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <functional>
#include <map>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <numeric>

namespace duckt {

// Forward declarations
struct Value;
struct Tensor;
struct Function;
struct Layer;
struct Model;
struct Agent;
class Environment;

using ValuePtr = std::shared_ptr<Value>;

// ============ Tensor ============

struct Tensor {
    std::vector<double> data;
    std::vector<int> shape;

    Tensor() = default;

    Tensor(std::vector<double> d, std::vector<int> s)
        : data(std::move(d)), shape(std::move(s)) {}

    // Create tensor of zeros
    static Tensor zeros(const std::vector<int>& shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        return Tensor(std::vector<double>(size, 0.0), shape);
    }

    // Create tensor of ones
    static Tensor ones(const std::vector<int>& shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        return Tensor(std::vector<double>(size, 1.0), shape);
    }

    // Create tensor with random values [0, 1)
    static Tensor rand(const std::vector<int>& shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        std::vector<double> data(size);
        for (int i = 0; i < size; i++) {
            data[i] = static_cast<double>(std::rand()) / RAND_MAX;
        }
        return Tensor(std::move(data), shape);
    }

    int totalSize() const {
        int size = 1;
        for (int dim : shape) size *= dim;
        return size;
    }

    // Element-wise operations
    Tensor add(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::runtime_error("Tensor shape mismatch for addition");
        }
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = data[i] + other.data[i];
        }
        return Tensor(std::move(result), shape);
    }

    Tensor sub(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::runtime_error("Tensor shape mismatch for subtraction");
        }
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = data[i] - other.data[i];
        }
        return Tensor(std::move(result), shape);
    }

    Tensor mul(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::runtime_error("Tensor shape mismatch for multiplication");
        }
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = data[i] * other.data[i];
        }
        return Tensor(std::move(result), shape);
    }

    Tensor div(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::runtime_error("Tensor shape mismatch for division");
        }
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = data[i] / other.data[i];
        }
        return Tensor(std::move(result), shape);
    }

    // Scalar operations
    Tensor addScalar(double scalar) const {
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = data[i] + scalar;
        }
        return Tensor(std::move(result), shape);
    }

    Tensor mulScalar(double scalar) const {
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = data[i] * scalar;
        }
        return Tensor(std::move(result), shape);
    }

    // Matrix multiplication (supports 1D and 2D tensors)
    Tensor matmul(const Tensor& other) const {
        // Handle 1D vectors by promoting to 2D
        Tensor left = *this;
        Tensor right = other;
        bool leftWas1D = false;
        bool rightWas1D = false;

        if (left.shape.size() == 1) {
            leftWas1D = true;
            left = Tensor(left.data, {1, left.shape[0]});
        }
        if (right.shape.size() == 1) {
            rightWas1D = true;
            right = Tensor(right.data, {right.shape[0], 1});
        }

        if (left.shape.size() != 2 || right.shape.size() != 2) {
            throw std::runtime_error("Matrix multiplication requires 1D or 2D tensors");
        }
        if (left.shape[1] != right.shape[0]) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }

        int m = left.shape[0];
        int n = left.shape[1];
        int p = right.shape[1];

        std::vector<double> result(m * p, 0.0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    result[i * p + j] += left.data[i * n + k] * right.data[k * p + j];
                }
            }
        }

        // Flatten result back to 1D if appropriate
        if (leftWas1D && rightWas1D) {
            // Both were 1D -> dot product, return scalar-like 1D tensor
            return Tensor(std::move(result), {1});
        } else if (leftWas1D) {
            // Left was 1D -> result is [1, p] -> flatten to [p]
            return Tensor(std::move(result), {p});
        } else if (rightWas1D) {
            // Right was 1D -> result is [m, 1] -> flatten to [m]
            return Tensor(std::move(result), {m});
        }
        return Tensor(std::move(result), {m, p});
    }

    // Transpose (2D only)
    Tensor transpose() const {
        if (shape.size() != 2) {
            throw std::runtime_error("Transpose requires 2D tensor");
        }

        int rows = shape[0];
        int cols = shape[1];
        std::vector<double> result(data.size());

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j * rows + i] = data[i * cols + j];
            }
        }

        return Tensor(std::move(result), {cols, rows});
    }

    // Reshape
    Tensor reshape(const std::vector<int>& newShape) const {
        int newSize = 1;
        for (int dim : newShape) newSize *= dim;
        if (newSize != totalSize()) {
            throw std::runtime_error("Cannot reshape: total size must remain the same");
        }
        return Tensor(data, newShape);
    }

    // Reductions
    double sum() const {
        double total = 0;
        for (double v : data) total += v;
        return total;
    }

    double mean() const {
        return sum() / data.size();
    }

    double max() const {
        double m = data[0];
        for (double v : data) if (v > m) m = v;
        return m;
    }

    double min() const {
        double m = data[0];
        for (double v : data) if (v < m) m = v;
        return m;
    }

    // Activation functions (element-wise)
    Tensor relu() const {
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = data[i] > 0 ? data[i] : 0;
        }
        return Tensor(std::move(result), shape);
    }

    Tensor sigmoid() const {
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = 1.0 / (1.0 + std::exp(-data[i]));
        }
        return Tensor(std::move(result), shape);
    }

    Tensor tanh_() const {
        std::vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = std::tanh(data[i]);
        }
        return Tensor(std::move(result), shape);
    }

    Tensor softmax() const {
        // Apply softmax along last dimension
        std::vector<double> result(data.size());
        double maxVal = max();
        double sumExp = 0;
        for (double v : data) {
            sumExp += std::exp(v - maxVal);
        }
        for (size_t i = 0; i < data.size(); i++) {
            result[i] = std::exp(data[i] - maxVal) / sumExp;
        }
        return Tensor(std::move(result), shape);
    }

    // String representation
    std::string toString() const {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < shape.size(); i++) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "], data=[";
        int maxShow = 10;
        for (size_t i = 0; i < std::min(data.size(), (size_t)maxShow); i++) {
            if (i > 0) oss << ", ";
            oss << data[i];
        }
        if (data.size() > (size_t)maxShow) {
            oss << ", ...";
        }
        oss << "])";
        return oss.str();
    }
};

// ============ Function ============

struct Function {
    std::string name;
    std::vector<std::string> params;
    std::vector<struct Stmt*> body;  // Non-owning pointers to AST
    std::shared_ptr<Environment> closure;

    Function(std::string n, std::vector<std::string> p,
             std::vector<struct Stmt*> b, std::shared_ptr<Environment> env)
        : name(std::move(n)), params(std::move(p)), body(std::move(b)), closure(std::move(env)) {}
};

// ============ Layer ============

enum class LayerType {
    Linear,
    Conv2D,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout,
    BatchNorm,
    MaxPool2D,
    Flatten
};

struct Layer {
    LayerType type;
    std::string name;
    std::map<std::string, double> params;

    // Weights for trainable layers
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;

    Layer(LayerType t, std::string n) : type(t), name(std::move(n)) {}

    void initWeights(int in, int out) {
        // Xavier initialization
        double scale = std::sqrt(2.0 / (in + out));
        auto w = Tensor::rand({in, out});
        for (double& v : w.data) {
            v = (v - 0.5) * 2 * scale;
        }
        weights = std::make_shared<Tensor>(std::move(w));
        bias = std::make_shared<Tensor>(Tensor::zeros({out}));
    }

    Tensor forward(const Tensor& input) const {
        switch (type) {
            case LayerType::Linear: {
                // input: [batch, in_features] or [in_features]
                // weights: [in_features, out_features]
                // output: [batch, out_features] or [out_features]
                if (!weights || !bias) {
                    throw std::runtime_error("Linear layer not initialized");
                }

                // Handle 1D input by reshaping to [1, in_features]
                Tensor inputReshaped = input;
                bool was1D = false;
                if (input.shape.size() == 1) {
                    was1D = true;
                    inputReshaped = Tensor(input.data, {1, input.shape[0]});
                }

                Tensor result = inputReshaped.matmul(*weights);
                // Add bias (broadcast)
                for (size_t i = 0; i < result.data.size(); i++) {
                    result.data[i] += bias->data[i % bias->data.size()];
                }
                // Reshape back to 1D if input was 1D
                if (was1D) {
                    result = result.reshape({result.shape[1]});
                }
                return result;
            }
            case LayerType::ReLU:
                return input.relu();
            case LayerType::Sigmoid:
                return input.sigmoid();
            case LayerType::Tanh:
                return input.tanh_();
            case LayerType::Softmax:
                return input.softmax();
            case LayerType::Dropout: {
                // During inference, dropout is identity
                return input;
            }
            case LayerType::Flatten: {
                int size = input.totalSize();
                return input.reshape({size});
            }
            default:
                throw std::runtime_error("Layer type not implemented for forward pass");
        }
    }
};

// ============ Model ============

struct Model {
    std::string name;
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<Function> forwardFn;

    Model(std::string n) : name(std::move(n)) {}

    Tensor forward(const Tensor& input) const {
        Tensor x = input;
        for (const auto& layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }
};

// ============ Agent ============

struct Agent {
    std::string name;
    std::map<std::string, Value> state;
    std::map<std::string, std::shared_ptr<Function>> tools;
    std::map<std::string, std::shared_ptr<Function>> methods;

    Agent(std::string n) : name(std::move(n)) {}
};

// ============ Native Function ============

struct NativeFunction {
    std::string name;
    int arity;  // -1 for variadic
    std::function<Value(const std::vector<Value>&)> fn;

    NativeFunction(std::string n, int a, std::function<Value(const std::vector<Value>&)> f)
        : name(std::move(n)), arity(a), fn(std::move(f)) {}
};

// ============ Value ============

enum class ValueType {
    None,
    Bool,
    Number,
    String,
    Tensor,
    List,
    Function,
    NativeFunction,
    Layer,
    Model,
    Agent
};

struct Value {
    ValueType type;
    std::variant<
        std::nullptr_t,
        bool,
        double,
        std::string,
        std::shared_ptr<Tensor>,
        std::vector<Value>,
        std::shared_ptr<Function>,
        std::shared_ptr<NativeFunction>,
        std::shared_ptr<Layer>,
        std::shared_ptr<Model>,
        std::shared_ptr<Agent>
    > data;

    // Constructors
    Value() : type(ValueType::None), data(nullptr) {}
    Value(std::nullptr_t) : type(ValueType::None), data(nullptr) {}
    Value(bool b) : type(ValueType::Bool), data(b) {}
    Value(int n) : type(ValueType::Number), data(static_cast<double>(n)) {}
    Value(long long n) : type(ValueType::Number), data(static_cast<double>(n)) {}
    Value(double n) : type(ValueType::Number), data(n) {}
    Value(const char* s) : type(ValueType::String), data(std::string(s)) {}
    Value(std::string s) : type(ValueType::String), data(std::move(s)) {}
    Value(Tensor t) : type(ValueType::Tensor), data(std::make_shared<Tensor>(std::move(t))) {}
    Value(std::shared_ptr<Tensor> t) : type(ValueType::Tensor), data(std::move(t)) {}
    Value(std::vector<Value> v) : type(ValueType::List), data(std::move(v)) {}
    Value(std::shared_ptr<Function> f) : type(ValueType::Function), data(std::move(f)) {}
    Value(std::shared_ptr<NativeFunction> f) : type(ValueType::NativeFunction), data(std::move(f)) {}
    Value(std::shared_ptr<Layer> l) : type(ValueType::Layer), data(std::move(l)) {}
    Value(std::shared_ptr<Model> m) : type(ValueType::Model), data(std::move(m)) {}
    Value(std::shared_ptr<Agent> a) : type(ValueType::Agent), data(std::move(a)) {}

    // Type checking
    bool isNone() const { return type == ValueType::None; }
    bool isBool() const { return type == ValueType::Bool; }
    bool isNumber() const { return type == ValueType::Number; }
    bool isString() const { return type == ValueType::String; }
    bool isTensor() const { return type == ValueType::Tensor; }
    bool isList() const { return type == ValueType::List; }
    bool isFunction() const { return type == ValueType::Function; }
    bool isNativeFunction() const { return type == ValueType::NativeFunction; }
    bool isLayer() const { return type == ValueType::Layer; }
    bool isModel() const { return type == ValueType::Model; }
    bool isAgent() const { return type == ValueType::Agent; }
    bool isCallable() const { return isFunction() || isNativeFunction() || isLayer() || isModel(); }

    // Value getters
    bool asBool() const { return std::get<bool>(data); }
    double asNumber() const { return std::get<double>(data); }
    const std::string& asString() const { return std::get<std::string>(data); }
    std::shared_ptr<Tensor> asTensor() const { return std::get<std::shared_ptr<Tensor>>(data); }
    const std::vector<Value>& asList() const { return std::get<std::vector<Value>>(data); }
    std::vector<Value>& asList() { return std::get<std::vector<Value>>(data); }
    std::shared_ptr<Function> asFunction() const { return std::get<std::shared_ptr<Function>>(data); }
    std::shared_ptr<NativeFunction> asNativeFunction() const { return std::get<std::shared_ptr<NativeFunction>>(data); }
    std::shared_ptr<Layer> asLayer() const { return std::get<std::shared_ptr<Layer>>(data); }
    std::shared_ptr<Model> asModel() const { return std::get<std::shared_ptr<Model>>(data); }
    std::shared_ptr<Agent> asAgent() const { return std::get<std::shared_ptr<Agent>>(data); }

    // Truthiness
    bool isTruthy() const {
        switch (type) {
            case ValueType::None: return false;
            case ValueType::Bool: return asBool();
            case ValueType::Number: return asNumber() != 0;
            case ValueType::String: return !asString().empty();
            case ValueType::List: return !asList().empty();
            default: return true;
        }
    }

    // String representation
    std::string toString() const {
        switch (type) {
            case ValueType::None: return "none";
            case ValueType::Bool: return asBool() ? "true" : "false";
            case ValueType::Number: {
                double n = asNumber();
                if (n == (long long)n) {
                    return std::to_string((long long)n);
                }
                return std::to_string(n);
            }
            case ValueType::String: return asString();
            case ValueType::Tensor: return asTensor()->toString();
            case ValueType::List: {
                std::ostringstream oss;
                oss << "[";
                const auto& list = asList();
                for (size_t i = 0; i < list.size(); i++) {
                    if (i > 0) oss << ", ";
                    if (list[i].isString()) {
                        oss << "\"" << list[i].asString() << "\"";
                    } else {
                        oss << list[i].toString();
                    }
                }
                oss << "]";
                return oss.str();
            }
            case ValueType::Function:
                return "<fn " + asFunction()->name + ">";
            case ValueType::NativeFunction:
                return "<native fn " + asNativeFunction()->name + ">";
            case ValueType::Layer:
                return "<layer " + asLayer()->name + ">";
            case ValueType::Model:
                return "<model " + asModel()->name + ">";
            case ValueType::Agent:
                return "<agent " + asAgent()->name + ">";
        }
        return "<unknown>";
    }

    // Type name
    std::string typeName() const {
        switch (type) {
            case ValueType::None: return "none";
            case ValueType::Bool: return "bool";
            case ValueType::Number: return "number";
            case ValueType::String: return "string";
            case ValueType::Tensor: return "tensor";
            case ValueType::List: return "list";
            case ValueType::Function: return "function";
            case ValueType::NativeFunction: return "function";
            case ValueType::Layer: return "layer";
            case ValueType::Model: return "model";
            case ValueType::Agent: return "agent";
        }
        return "unknown";
    }
};

// Value equality
inline bool operator==(const Value& a, const Value& b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ValueType::None: return true;
        case ValueType::Bool: return a.asBool() == b.asBool();
        case ValueType::Number: return a.asNumber() == b.asNumber();
        case ValueType::String: return a.asString() == b.asString();
        case ValueType::List: {
            const auto& la = a.asList();
            const auto& lb = b.asList();
            if (la.size() != lb.size()) return false;
            for (size_t i = 0; i < la.size(); i++) {
                if (!(la[i] == lb[i])) return false;
            }
            return true;
        }
        default: return false;  // Reference equality for complex types
    }
}

inline bool operator!=(const Value& a, const Value& b) {
    return !(a == b);
}

} // namespace duckt

#endif // DUCKT_TYPES_HPP
