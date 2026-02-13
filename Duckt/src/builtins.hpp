#ifndef DUCKT_BUILTINS_HPP
#define DUCKT_BUILTINS_HPP

#include "types.hpp"
#include "environment.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <array>
#include <memory>
#include <regex>
#include "autograd.hpp"

namespace duckt {

class Builtins {
public:
    static void registerAll(std::shared_ptr<Environment> env) {
        // I/O
        registerPrint(env);
        registerInput(env);
        registerLoad(env);

        // Tensor creation
        registerZeros(env);
        registerOnes(env);
        registerRand(env);
        registerArange(env);

        // Math functions
        registerSum(env);
        registerMean(env);
        registerMax(env);
        registerMin(env);
        registerSqrt(env);
        registerExp(env);
        registerLog(env);
        registerAbs(env);

        // Utilities
        registerLen(env);
        registerType(env);
        registerShape(env);
        registerRange(env);
        registerStr(env);
        registerInt(env);
        registerFloat(env);

        // Activation functions (also available as standalone)
        registerRelu(env);
        registerSigmoid(env);
        registerTanh(env);
        registerSoftmax(env);

        // Layer constructors
        registerLinear(env);
        registerConv2D(env);
        registerMaxPool2D(env);
        registerDropout(env);
        registerBatchNorm(env);
        registerFlatten(env);

        // Activation layer constructors
        registerReLULayer(env);
        registerSigmoidLayer(env);
        registerTanhLayer(env);
        registerSoftmaxLayer(env);

        // AI & Network functions
        registerHttpGet(env);
        registerHttpPost(env);
        registerJsonParse(env);
        registerJsonGet(env);
        registerEnv(env);
        registerAiChat(env);
        registerSplit(env);
        registerJoin(env);
        registerReplace(env);
        registerTrim(env);
        registerLower(env);
        registerUpper(env);
        registerContains(env);
        registerStartsWith(env);
        registerEndsWith(env);
        registerSlice(env);

        // Neural Network Training
        registerNNCreate(env);
        registerNNForward(env);
        registerNNTrain(env);
        registerNNPredict(env);
        registerNNSave(env);
        registerNNLoad(env);
        registerArgmax(env);
        registerTensorGet(env);
        registerDataPrepare(env);
    }

private:
    // ============ I/O ============

    static void registerPrint(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("print", -1, [](const std::vector<Value>& args) -> Value {
            for (size_t i = 0; i < args.size(); i++) {
                if (i > 0) std::cout << " ";
                std::cout << args[i].toString();
            }
            std::cout << std::endl;
            return Value();
        });
        env->define("print", Value(fn));
    }

    static void registerInput(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("input", 1, [](const std::vector<Value>& args) -> Value {
            if (args.size() >= 1 && args[0].isString()) {
                std::cout << args[0].asString();
            }
            std::string line;
            std::getline(std::cin, line);
            return Value(line);
        });
        env->define("input", Value(fn));
    }

    static void registerLoad(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("load", 1, [](const std::vector<Value>& args) -> Value {
            if (args.empty() || !args[0].isString()) {
                throw std::runtime_error("load() requires a string path argument");
            }
            std::string path = args[0].asString();
            std::ifstream file(path);
            if (!file) {
                throw std::runtime_error("Could not open file: " + path);
            }

            // Simple CSV loading -> returns list of lists
            std::vector<Value> rows;
            std::string line;
            while (std::getline(file, line)) {
                std::vector<Value> row;
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    try {
                        double num = std::stod(cell);
                        row.push_back(Value(num));
                    } catch (...) {
                        row.push_back(Value(cell));
                    }
                }
                rows.push_back(Value(std::move(row)));
            }
            return Value(std::move(rows));
        });
        env->define("load", Value(fn));
    }

    // ============ Tensor Creation ============

    static void registerZeros(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("zeros", 1, [](const std::vector<Value>& args) -> Value {
            std::vector<int> shapeVec = toShape(args[0]);
            return Value(Tensor::zeros(shapeVec));
        });
        env->define("zeros", Value(fn));
    }

    static void registerOnes(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("ones", 1, [](const std::vector<Value>& args) -> Value {
            std::vector<int> shapeVec = toShape(args[0]);
            return Value(Tensor::ones(shapeVec));
        });
        env->define("ones", Value(fn));
    }

    static void registerRand(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("rand", 1, [](const std::vector<Value>& args) -> Value {
            std::vector<int> shapeVec = toShape(args[0]);
            return Value(Tensor::rand(shapeVec));
        });
        env->define("rand", Value(fn));
    }

    static void registerArange(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("arange", -1, [](const std::vector<Value>& args) -> Value {
            double start = 0, end = 0, step = 1;

            if (args.size() == 1) {
                end = args[0].asNumber();
            } else if (args.size() == 2) {
                start = args[0].asNumber();
                end = args[1].asNumber();
            } else if (args.size() >= 3) {
                start = args[0].asNumber();
                end = args[1].asNumber();
                step = args[2].asNumber();
            }

            std::vector<double> data;
            for (double v = start; v < end; v += step) {
                data.push_back(v);
            }
            int dataSize = static_cast<int>(data.size());
            return Value(Tensor(std::move(data), {dataSize}));
        });
        env->define("arange", Value(fn));
    }

    // ============ Math Functions ============

    static void registerSum(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("sum", 1, [](const std::vector<Value>& args) -> Value {
            if (args[0].isTensor()) {
                return Value(args[0].asTensor()->sum());
            } else if (args[0].isList()) {
                double total = 0;
                for (const auto& v : args[0].asList()) {
                    if (v.isNumber()) total += v.asNumber();
                }
                return Value(total);
            }
            throw std::runtime_error("sum() requires tensor or list");
        });
        env->define("sum", Value(fn));
    }

    static void registerMean(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("mean", 1, [](const std::vector<Value>& args) -> Value {
            if (args[0].isTensor()) {
                return Value(args[0].asTensor()->mean());
            }
            throw std::runtime_error("mean() requires tensor");
        });
        env->define("mean", Value(fn));
    }

    static void registerMax(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("max", 1, [](const std::vector<Value>& args) -> Value {
            if (args[0].isTensor()) {
                return Value(args[0].asTensor()->max());
            } else if (args[0].isList()) {
                const auto& list = args[0].asList();
                if (list.empty()) throw std::runtime_error("max() on empty list");
                double m = list[0].asNumber();
                for (const auto& v : list) {
                    if (v.asNumber() > m) m = v.asNumber();
                }
                return Value(m);
            }
            throw std::runtime_error("max() requires tensor or list");
        });
        env->define("max", Value(fn));
    }

    static void registerMin(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("min", 1, [](const std::vector<Value>& args) -> Value {
            if (args[0].isTensor()) {
                return Value(args[0].asTensor()->min());
            } else if (args[0].isList()) {
                const auto& list = args[0].asList();
                if (list.empty()) throw std::runtime_error("min() on empty list");
                double m = list[0].asNumber();
                for (const auto& v : list) {
                    if (v.asNumber() < m) m = v.asNumber();
                }
                return Value(m);
            }
            throw std::runtime_error("min() requires tensor or list");
        });
        env->define("min", Value(fn));
    }

    static void registerSqrt(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("sqrt", 1, [](const std::vector<Value>& args) -> Value {
            return Value(std::sqrt(args[0].asNumber()));
        });
        env->define("sqrt", Value(fn));
    }

    static void registerExp(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("exp", 1, [](const std::vector<Value>& args) -> Value {
            return Value(std::exp(args[0].asNumber()));
        });
        env->define("exp", Value(fn));
    }

    static void registerLog(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("log", 1, [](const std::vector<Value>& args) -> Value {
            return Value(std::log(args[0].asNumber()));
        });
        env->define("log", Value(fn));
    }

    static void registerAbs(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("abs", 1, [](const std::vector<Value>& args) -> Value {
            return Value(std::abs(args[0].asNumber()));
        });
        env->define("abs", Value(fn));
    }

    // ============ Utilities ============

    static void registerLen(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("len", 1, [](const std::vector<Value>& args) -> Value {
            if (args[0].isString()) {
                return Value(static_cast<double>(args[0].asString().length()));
            } else if (args[0].isList()) {
                return Value(static_cast<double>(args[0].asList().size()));
            } else if (args[0].isTensor()) {
                return Value(static_cast<double>(args[0].asTensor()->shape[0]));
            }
            throw std::runtime_error("len() requires string, list, or tensor");
        });
        env->define("len", Value(fn));
    }

    static void registerType(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("type", 1, [](const std::vector<Value>& args) -> Value {
            return Value(args[0].typeName());
        });
        env->define("type", Value(fn));
    }

    static void registerShape(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("shape", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isTensor()) {
                throw std::runtime_error("shape() requires tensor");
            }
            std::vector<Value> result;
            for (int dim : args[0].asTensor()->shape) {
                result.push_back(Value(static_cast<double>(dim)));
            }
            return Value(std::move(result));
        });
        env->define("shape", Value(fn));
    }

    static void registerRange(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("range", -1, [](const std::vector<Value>& args) -> Value {
            int start = 0, end = 0, step = 1;

            if (args.size() == 1) {
                end = static_cast<int>(args[0].asNumber());
            } else if (args.size() == 2) {
                start = static_cast<int>(args[0].asNumber());
                end = static_cast<int>(args[1].asNumber());
            } else if (args.size() >= 3) {
                start = static_cast<int>(args[0].asNumber());
                end = static_cast<int>(args[1].asNumber());
                step = static_cast<int>(args[2].asNumber());
            }

            std::vector<Value> result;
            for (int i = start; i < end; i += step) {
                result.push_back(Value(static_cast<double>(i)));
            }
            return Value(std::move(result));
        });
        env->define("range", Value(fn));
    }

    static void registerStr(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("str", 1, [](const std::vector<Value>& args) -> Value {
            return Value(args[0].toString());
        });
        env->define("str", Value(fn));
    }

    static void registerInt(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("int", 1, [](const std::vector<Value>& args) -> Value {
            if (args[0].isNumber()) {
                return Value(static_cast<double>(static_cast<long long>(args[0].asNumber())));
            } else if (args[0].isString()) {
                return Value(static_cast<double>(std::stoll(args[0].asString())));
            }
            throw std::runtime_error("int() requires number or string");
        });
        env->define("int", Value(fn));
    }

    static void registerFloat(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("float", 1, [](const std::vector<Value>& args) -> Value {
            if (args[0].isNumber()) {
                return args[0];
            } else if (args[0].isString()) {
                return Value(std::stod(args[0].asString()));
            }
            throw std::runtime_error("float() requires number or string");
        });
        env->define("float", Value(fn));
    }

    // ============ Activation Functions ============

    static void registerRelu(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("relu", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isTensor()) {
                throw std::runtime_error("relu() requires tensor");
            }
            return Value(args[0].asTensor()->relu());
        });
        env->define("relu", Value(fn));
    }

    static void registerSigmoid(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("sigmoid", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isTensor()) {
                throw std::runtime_error("sigmoid() requires tensor");
            }
            return Value(args[0].asTensor()->sigmoid());
        });
        env->define("sigmoid", Value(fn));
    }

    static void registerTanh(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("tanh", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isTensor()) {
                throw std::runtime_error("tanh() requires tensor");
            }
            return Value(args[0].asTensor()->tanh_());
        });
        env->define("tanh", Value(fn));
    }

    static void registerSoftmax(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("softmax", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isTensor()) {
                throw std::runtime_error("softmax() requires tensor");
            }
            return Value(args[0].asTensor()->softmax());
        });
        env->define("softmax", Value(fn));
    }

    // ============ Layer Constructors ============

    static void registerLinear(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("Linear", 2, [](const std::vector<Value>& args) -> Value {
            int inFeatures = static_cast<int>(args[0].asNumber());
            int outFeatures = static_cast<int>(args[1].asNumber());

            auto layer = std::make_shared<Layer>(LayerType::Linear, "Linear");
            layer->params["in_features"] = inFeatures;
            layer->params["out_features"] = outFeatures;
            layer->initWeights(inFeatures, outFeatures);

            return Value(layer);
        });
        env->define("Linear", Value(fn));
    }

    static void registerConv2D(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("Conv2D", 3, [](const std::vector<Value>& args) -> Value {
            int inChannels = static_cast<int>(args[0].asNumber());
            int outChannels = static_cast<int>(args[1].asNumber());
            int kernel = static_cast<int>(args[2].asNumber());

            auto layer = std::make_shared<Layer>(LayerType::Conv2D, "Conv2D");
            layer->params["in_channels"] = inChannels;
            layer->params["out_channels"] = outChannels;
            layer->params["kernel_size"] = kernel;

            return Value(layer);
        });
        env->define("Conv2D", Value(fn));
    }

    static void registerMaxPool2D(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("MaxPool2D", 1, [](const std::vector<Value>& args) -> Value {
            int kernel = static_cast<int>(args[0].asNumber());

            auto layer = std::make_shared<Layer>(LayerType::MaxPool2D, "MaxPool2D");
            layer->params["kernel_size"] = kernel;

            return Value(layer);
        });
        env->define("MaxPool2D", Value(fn));
    }

    static void registerDropout(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("Dropout", 1, [](const std::vector<Value>& args) -> Value {
            double p = args[0].asNumber();

            auto layer = std::make_shared<Layer>(LayerType::Dropout, "Dropout");
            layer->params["p"] = p;

            return Value(layer);
        });
        env->define("Dropout", Value(fn));
    }

    static void registerBatchNorm(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("BatchNorm", 1, [](const std::vector<Value>& args) -> Value {
            int features = static_cast<int>(args[0].asNumber());

            auto layer = std::make_shared<Layer>(LayerType::BatchNorm, "BatchNorm");
            layer->params["features"] = features;

            return Value(layer);
        });
        env->define("BatchNorm", Value(fn));
    }

    static void registerFlatten(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("Flatten", 0, [](const std::vector<Value>&) -> Value {
            auto layer = std::make_shared<Layer>(LayerType::Flatten, "Flatten");
            return Value(layer);
        });
        env->define("Flatten", Value(fn));
    }

    // Activation layer constructors (return Layer objects, not functions)
    static void registerReLULayer(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("ReLU", 0, [](const std::vector<Value>&) -> Value {
            auto layer = std::make_shared<Layer>(LayerType::ReLU, "ReLU");
            return Value(layer);
        });
        env->define("ReLU", Value(fn));
    }

    static void registerSigmoidLayer(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("Sigmoid", 0, [](const std::vector<Value>&) -> Value {
            auto layer = std::make_shared<Layer>(LayerType::Sigmoid, "Sigmoid");
            return Value(layer);
        });
        env->define("Sigmoid", Value(fn));
    }

    static void registerTanhLayer(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("Tanh", 0, [](const std::vector<Value>&) -> Value {
            auto layer = std::make_shared<Layer>(LayerType::Tanh, "Tanh");
            return Value(layer);
        });
        env->define("Tanh", Value(fn));
    }

    static void registerSoftmaxLayer(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("Softmax", 0, [](const std::vector<Value>&) -> Value {
            auto layer = std::make_shared<Layer>(LayerType::Softmax, "Softmax");
            return Value(layer);
        });
        env->define("Softmax", Value(fn));
    }

    // ============ AI & Network Functions ============

    // Execute a shell command and return output
    static std::string execCommand(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("Failed to run command");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }

    // Escape string for shell
    static std::string shellEscape(const std::string& s) {
        std::string result = "'";
        for (char c : s) {
            if (c == '\'') {
                result += "'\\''";
            } else {
                result += c;
            }
        }
        result += "'";
        return result;
    }

    // Simple JSON parser helpers
    static void skipWhitespace(const std::string& json, size_t& pos) {
        while (pos < json.size() && std::isspace(json[pos])) pos++;
    }

    static std::string parseJsonString(const std::string& json, size_t& pos) {
        if (json[pos] != '"') throw std::runtime_error("Expected '\"' in JSON");
        pos++; // skip opening quote
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                switch (json[pos]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        // Unicode escape - simplified handling
                        if (pos + 4 < json.size()) {
                            pos += 4;
                        }
                        result += '?'; // Placeholder for unicode
                        break;
                    }
                    default: result += json[pos]; break;
                }
            } else {
                result += json[pos];
            }
            pos++;
        }
        if (pos < json.size()) pos++; // skip closing quote
        return result;
    }

    static Value parseJsonValue(const std::string& json, size_t& pos) {
        skipWhitespace(json, pos);
        if (pos >= json.size()) return Value();

        char c = json[pos];

        // String
        if (c == '"') {
            return Value(parseJsonString(json, pos));
        }

        // Number
        if (c == '-' || std::isdigit(c)) {
            size_t start = pos;
            if (json[pos] == '-') pos++;
            while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.' || json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+' || json[pos] == '-')) {
                if ((json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+') && pos > start + 1) {
                    pos++;
                } else if (json[pos] == '-' && pos > start) {
                    pos++;
                } else if (std::isdigit(json[pos]) || json[pos] == '.') {
                    pos++;
                } else {
                    break;
                }
            }
            return Value(std::stod(json.substr(start, pos - start)));
        }

        // Boolean true
        if (json.substr(pos, 4) == "true") {
            pos += 4;
            return Value(true);
        }

        // Boolean false
        if (json.substr(pos, 5) == "false") {
            pos += 5;
            return Value(false);
        }

        // Null
        if (json.substr(pos, 4) == "null") {
            pos += 4;
            return Value();
        }

        // Array
        if (c == '[') {
            pos++; // skip '['
            std::vector<Value> arr;
            skipWhitespace(json, pos);
            if (pos < json.size() && json[pos] == ']') {
                pos++;
                return Value(std::move(arr));
            }
            while (pos < json.size()) {
                arr.push_back(parseJsonValue(json, pos));
                skipWhitespace(json, pos);
                if (pos < json.size() && json[pos] == ',') {
                    pos++;
                } else {
                    break;
                }
            }
            skipWhitespace(json, pos);
            if (pos < json.size() && json[pos] == ']') pos++;
            return Value(std::move(arr));
        }

        // Object - store as list of [key, value] pairs
        if (c == '{') {
            pos++; // skip '{'
            std::vector<Value> obj;
            skipWhitespace(json, pos);
            if (pos < json.size() && json[pos] == '}') {
                pos++;
                return Value(std::move(obj));
            }
            while (pos < json.size()) {
                skipWhitespace(json, pos);
                std::string key = parseJsonString(json, pos);
                skipWhitespace(json, pos);
                if (pos < json.size() && json[pos] == ':') pos++;
                skipWhitespace(json, pos);
                Value val = parseJsonValue(json, pos);

                // Store as [key, value] pair
                std::vector<Value> pair;
                pair.push_back(Value(key));
                pair.push_back(val);
                obj.push_back(Value(std::move(pair)));

                skipWhitespace(json, pos);
                if (pos < json.size() && json[pos] == ',') {
                    pos++;
                } else {
                    break;
                }
            }
            skipWhitespace(json, pos);
            if (pos < json.size() && json[pos] == '}') pos++;
            return Value(std::move(obj));
        }

        return Value();
    }

    static void registerHttpGet(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("http_get", -1, [](const std::vector<Value>& args) -> Value {
            if (args.empty() || !args[0].isString()) {
                throw std::runtime_error("http_get() requires URL string");
            }
            std::string url = args[0].asString();

            // Build curl command
            std::string cmd = "curl -s -L " + shellEscape(url);

            // Add headers if provided
            if (args.size() > 1 && args[1].isList()) {
                for (const auto& header : args[1].asList()) {
                    if (header.isString()) {
                        cmd += " -H " + shellEscape(header.asString());
                    }
                }
            }

            cmd += " 2>/dev/null";

            std::string result = execCommand(cmd);
            return Value(result);
        });
        env->define("http_get", Value(fn));
    }

    static void registerHttpPost(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("http_post", -1, [](const std::vector<Value>& args) -> Value {
            if (args.size() < 2 || !args[0].isString() || !args[1].isString()) {
                throw std::runtime_error("http_post() requires URL and data strings");
            }
            std::string url = args[0].asString();
            std::string data = args[1].asString();

            // Build curl command
            std::string cmd = "curl -s -L -X POST " + shellEscape(url);
            cmd += " -d " + shellEscape(data);

            // Add headers if provided
            if (args.size() > 2 && args[2].isList()) {
                for (const auto& header : args[2].asList()) {
                    if (header.isString()) {
                        cmd += " -H " + shellEscape(header.asString());
                    }
                }
            }

            cmd += " 2>/dev/null";

            std::string result = execCommand(cmd);
            return Value(result);
        });
        env->define("http_post", Value(fn));
    }

    static void registerJsonParse(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("json_parse", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("json_parse() requires string");
            }
            std::string json = args[0].asString();
            size_t pos = 0;
            return parseJsonValue(json, pos);
        });
        env->define("json_parse", Value(fn));
    }

    static void registerJsonGet(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("json_get", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isList()) {
                throw std::runtime_error("json_get() requires parsed JSON object");
            }

            const auto& obj = args[0].asList();
            std::string key = args[1].asString();

            // Search for key in list of [key, value] pairs
            for (const auto& pair : obj) {
                if (pair.isList() && pair.asList().size() >= 2) {
                    if (pair.asList()[0].isString() && pair.asList()[0].asString() == key) {
                        return pair.asList()[1];
                    }
                }
            }
            return Value(); // Return none if not found
        });
        env->define("json_get", Value(fn));
    }

    static void registerEnv(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("env", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("env() requires string");
            }
            const char* val = std::getenv(args[0].asString().c_str());
            if (val) {
                return Value(std::string(val));
            }
            return Value(); // Return none if not set
        });
        env->define("env", Value(fn));
    }

    static void registerAiChat(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("ai_chat", -1, [](const std::vector<Value>& args) -> Value {
            if (args.empty() || !args[0].isString()) {
                throw std::runtime_error("ai_chat() requires message string");
            }
            std::string message = args[0].asString();

            // Get API key - from argument or environment
            std::string apiKey;
            if (args.size() > 1 && args[1].isString()) {
                apiKey = args[1].asString();
            } else {
                const char* envKey = std::getenv("ANTHROPIC_API_KEY");
                if (envKey) {
                    apiKey = envKey;
                } else {
                    throw std::runtime_error("ai_chat() requires API key (pass as second arg or set ANTHROPIC_API_KEY)");
                }
            }

            // Optional model parameter
            std::string model = "claude-sonnet-4-20250514";
            if (args.size() > 2 && args[2].isString()) {
                model = args[2].asString();
            }

            // Escape message for JSON
            std::string escapedMsg;
            for (char c : message) {
                switch (c) {
                    case '"': escapedMsg += "\\\""; break;
                    case '\\': escapedMsg += "\\\\"; break;
                    case '\n': escapedMsg += "\\n"; break;
                    case '\r': escapedMsg += "\\r"; break;
                    case '\t': escapedMsg += "\\t"; break;
                    default: escapedMsg += c; break;
                }
            }

            // Build JSON body
            std::string body = "{\"model\":\"" + model + "\",\"max_tokens\":1024,\"messages\":[{\"role\":\"user\",\"content\":\"" + escapedMsg + "\"}]}";

            // Build curl command
            std::string cmd = "curl -s -X POST https://api.anthropic.com/v1/messages";
            cmd += " -H 'Content-Type: application/json'";
            cmd += " -H 'x-api-key: " + apiKey + "'";
            cmd += " -H 'anthropic-version: 2023-06-01'";
            cmd += " -d " + shellEscape(body);
            cmd += " 2>/dev/null";

            std::string response = execCommand(cmd);

            // Parse response to extract content
            size_t pos = 0;
            Value parsed = parseJsonValue(response, pos);

            if (parsed.isList()) {
                // Look for "content" key
                for (const auto& pair : parsed.asList()) {
                    if (pair.isList() && pair.asList().size() >= 2) {
                        if (pair.asList()[0].isString() && pair.asList()[0].asString() == "content") {
                            // content is an array of content blocks
                            Value content = pair.asList()[1];
                            if (content.isList() && !content.asList().empty()) {
                                // Get first content block
                                Value block = content.asList()[0];
                                if (block.isList()) {
                                    // Look for "text" in the block
                                    for (const auto& bp : block.asList()) {
                                        if (bp.isList() && bp.asList().size() >= 2) {
                                            if (bp.asList()[0].isString() && bp.asList()[0].asString() == "text") {
                                                return bp.asList()[1];
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        // Check for error
                        if (pair.asList()[0].isString() && pair.asList()[0].asString() == "error") {
                            Value errObj = pair.asList()[1];
                            if (errObj.isList()) {
                                for (const auto& ep : errObj.asList()) {
                                    if (ep.isList() && ep.asList().size() >= 2) {
                                        if (ep.asList()[0].isString() && ep.asList()[0].asString() == "message") {
                                            throw std::runtime_error("AI API error: " + ep.asList()[1].asString());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // If parsing failed, return raw response
            return Value(response);
        });
        env->define("ai_chat", Value(fn));
    }

    // ============ String Functions ============

    static void registerSplit(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("split", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString() || !args[1].isString()) {
                throw std::runtime_error("split() requires two strings");
            }
            std::string str = args[0].asString();
            std::string delim = args[1].asString();
            std::vector<Value> result;

            size_t pos = 0;
            size_t found;
            while ((found = str.find(delim, pos)) != std::string::npos) {
                result.push_back(Value(str.substr(pos, found - pos)));
                pos = found + delim.length();
            }
            result.push_back(Value(str.substr(pos)));

            return Value(std::move(result));
        });
        env->define("split", Value(fn));
    }

    static void registerJoin(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("join", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isList() || !args[1].isString()) {
                throw std::runtime_error("join() requires list and delimiter string");
            }
            const auto& list = args[0].asList();
            std::string delim = args[1].asString();
            std::string result;

            for (size_t i = 0; i < list.size(); i++) {
                if (i > 0) result += delim;
                result += list[i].toString();
            }

            return Value(result);
        });
        env->define("join", Value(fn));
    }

    static void registerReplace(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("replace", 3, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString() || !args[1].isString() || !args[2].isString()) {
                throw std::runtime_error("replace() requires three strings");
            }
            std::string str = args[0].asString();
            std::string from = args[1].asString();
            std::string to = args[2].asString();

            size_t pos = 0;
            while ((pos = str.find(from, pos)) != std::string::npos) {
                str.replace(pos, from.length(), to);
                pos += to.length();
            }

            return Value(str);
        });
        env->define("replace", Value(fn));
    }

    static void registerTrim(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("trim", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("trim() requires string");
            }
            std::string str = args[0].asString();

            size_t start = str.find_first_not_of(" \t\n\r");
            if (start == std::string::npos) return Value(std::string(""));
            size_t end = str.find_last_not_of(" \t\n\r");

            return Value(str.substr(start, end - start + 1));
        });
        env->define("trim", Value(fn));
    }

    static void registerLower(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("lower", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("lower() requires string");
            }
            std::string str = args[0].asString();
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            return Value(str);
        });
        env->define("lower", Value(fn));
    }

    static void registerUpper(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("upper", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("upper() requires string");
            }
            std::string str = args[0].asString();
            std::transform(str.begin(), str.end(), str.begin(), ::toupper);
            return Value(str);
        });
        env->define("upper", Value(fn));
    }

    static void registerContains(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("contains", 2, [](const std::vector<Value>& args) -> Value {
            if (args[0].isString() && args[1].isString()) {
                return Value(args[0].asString().find(args[1].asString()) != std::string::npos);
            } else if (args[0].isList()) {
                for (const auto& item : args[0].asList()) {
                    if (item.toString() == args[1].toString()) return Value(true);
                }
                return Value(false);
            }
            throw std::runtime_error("contains() requires string/string or list/value");
        });
        env->define("contains", Value(fn));
    }

    static void registerStartsWith(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("starts_with", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString() || !args[1].isString()) {
                throw std::runtime_error("starts_with() requires two strings");
            }
            std::string str = args[0].asString();
            std::string prefix = args[1].asString();
            return Value(str.substr(0, prefix.length()) == prefix);
        });
        env->define("starts_with", Value(fn));
    }

    static void registerEndsWith(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("ends_with", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString() || !args[1].isString()) {
                throw std::runtime_error("ends_with() requires two strings");
            }
            std::string str = args[0].asString();
            std::string suffix = args[1].asString();
            if (suffix.length() > str.length()) return Value(false);
            return Value(str.substr(str.length() - suffix.length()) == suffix);
        });
        env->define("ends_with", Value(fn));
    }

    static void registerSlice(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("slice", -1, [](const std::vector<Value>& args) -> Value {
            if (args.size() < 2) {
                throw std::runtime_error("slice() requires at least 2 arguments");
            }

            int start = static_cast<int>(args[1].asNumber());

            if (args[0].isString()) {
                std::string str = args[0].asString();
                int len = static_cast<int>(str.length());
                if (start < 0) start = std::max(0, len + start);

                if (args.size() > 2) {
                    int end = static_cast<int>(args[2].asNumber());
                    if (end < 0) end = len + end;
                    return Value(str.substr(start, end - start));
                }
                return Value(str.substr(start));
            } else if (args[0].isList()) {
                const auto& list = args[0].asList();
                int len = static_cast<int>(list.size());
                if (start < 0) start = std::max(0, len + start);

                int end = len;
                if (args.size() > 2) {
                    end = static_cast<int>(args[2].asNumber());
                    if (end < 0) end = len + end;
                }

                std::vector<Value> result;
                for (int i = start; i < end && i < len; i++) {
                    result.push_back(list[i]);
                }
                return Value(std::move(result));
            }
            throw std::runtime_error("slice() requires string or list");
        });
        env->define("slice", Value(fn));
    }

    // ============ Neural Network Training ============

    // Store networks globally (simple approach)
    static std::map<std::string, std::shared_ptr<GradNetwork>>& getNetworks() {
        static std::map<std::string, std::shared_ptr<GradNetwork>> networks;
        return networks;
    }

    static void registerNNCreate(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("nn_create", -1, [](const std::vector<Value>& args) -> Value {
            if (args.empty()) {
                throw std::runtime_error("nn_create() requires list of layer sizes");
            }

            // Get layer sizes (handle both list and tensor)
            std::vector<int> sizes;
            if (args[0].isList()) {
                for (const auto& v : args[0].asList()) {
                    sizes.push_back(static_cast<int>(v.asNumber()));
                }
            } else if (args[0].isTensor()) {
                auto tensor = args[0].asTensor();
                for (double d : tensor->data) {
                    sizes.push_back(static_cast<int>(d));
                }
            } else {
                throw std::runtime_error("nn_create() requires list or tensor of layer sizes");
            }

            // Get activation (default: relu)
            std::string activation = "relu";
            if (args.size() > 1 && args[1].isString()) {
                activation = args[1].asString();
            }

            // Create network
            auto network = std::make_shared<GradNetwork>(sizes, activation);

            // Generate unique name
            static int counter = 0;
            std::string name = "network_" + std::to_string(counter++);
            getNetworks()[name] = network;

            return Value(name);
        });
        env->define("nn_create", Value(fn));
    }

    static void registerNNForward(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("nn_forward", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("nn_forward() requires network name");
            }

            std::string name = args[0].asString();
            auto& networks = getNetworks();
            if (networks.find(name) == networks.end()) {
                throw std::runtime_error("Network not found: " + name);
            }

            auto network = networks[name];

            // Convert input to GradTensor
            GradTensorPtr input;
            if (args[1].isTensor()) {
                auto t = args[1].asTensor();
                input = GradTensor::create(t->data, t->shape, false);
            } else if (args[1].isList()) {
                std::vector<double> data;
                for (const auto& v : args[1].asList()) {
                    data.push_back(v.asNumber());
                }
                int sz_tmp = static_cast<int>(data.size());
                input = GradTensor::create(std::move(data), {sz_tmp}, false);
            } else {
                throw std::runtime_error("nn_forward() requires tensor or list input");
            }

            // Forward pass
            auto output = network->forward(input);

            // Convert back to Duckt Tensor
            return Value(Tensor(output->data, output->shape));
        });
        env->define("nn_forward", Value(fn));
    }

    static void registerNNTrain(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("nn_train", -1, [](const std::vector<Value>& args) -> Value {
            if (args.size() < 3) {
                throw std::runtime_error("nn_train(network, X, y, [epochs], [lr], [batch_size], [verbose])");
            }

            // Get network
            std::string name = args[0].asString();
            auto& networks = getNetworks();
            if (networks.find(name) == networks.end()) {
                throw std::runtime_error("Network not found: " + name);
            }
            auto network = networks[name];

            // Get training data X (list of samples or 2D tensor)
            std::vector<Value> X_list;
            if (args[1].isList()) {
                X_list = args[1].asList();
            } else if (args[1].isTensor()) {
                // Convert 2D tensor to list of 1D tensors
                auto tensor = args[1].asTensor();
                if (tensor->shape.size() == 2) {
                    int rows = tensor->shape[0];
                    int cols = tensor->shape[1];
                    for (int r = 0; r < rows; r++) {
                        std::vector<double> row_data(cols);
                        for (int c = 0; c < cols; c++) {
                            row_data[c] = tensor->data[r * cols + c];
                        }
                        X_list.push_back(Value(Tensor(std::move(row_data), {cols})));
                    }
                } else {
                    throw std::runtime_error("X tensor must be 2D");
                }
            } else {
                throw std::runtime_error("X must be a list of samples or 2D tensor");
            }

            // Get labels y (list of labels or tensor)
            std::vector<Value> y_list;
            if (args[2].isList()) {
                y_list = args[2].asList();
            } else if (args[2].isTensor()) {
                auto tensor = args[2].asTensor();
                for (double d : tensor->data) {
                    y_list.push_back(Value(d));
                }
            } else {
                throw std::runtime_error("y must be a list of labels or tensor");
            }

            if (X_list.size() != y_list.size()) {
                throw std::runtime_error("X and y must have same length");
            }

            // Training parameters
            int epochs = args.size() > 3 ? static_cast<int>(args[3].asNumber()) : 100;
            double lr = args.size() > 4 ? args[4].asNumber() : 0.01;
            int batch_size = args.size() > 5 ? static_cast<int>(args[5].asNumber()) : 32;
            bool verbose = args.size() > 6 ? args[6].isTruthy() : true;

            // Create optimizer
            auto params = network->parameters();
            Adam optimizer(params, lr);

            // Determine if classification or regression
            // Classification if: y values are lists (one-hot) OR all y values are small non-negative integers
            bool is_classification = false;
            int num_classes = 0;
            if (y_list[0].isList()) {
                is_classification = true;
                num_classes = y_list[0].asList().size();
            } else if (y_list[0].isTensor()) {
                // y is a tensor - check shape
                auto t = y_list[0].asTensor();
                if (t->shape.size() == 1 && t->shape[0] > 1) {
                    // Multi-element tensor - probably one-hot for classification
                    is_classification = true;
                    num_classes = t->shape[0];
                }
                // Single element tensor - regression
            } else {
                // Scalar values - check if all are small non-negative integers
                bool all_int_labels = true;
                int max_label = 0;
                for (const auto& y : y_list) {
                    double val = y.asNumber();
                    int int_val = static_cast<int>(val);
                    // Only classify if value is exactly an integer and between 0-99
                    if (val != int_val || val < 0 || val > 99) {
                        all_int_labels = false;
                        break;
                    }
                    if (int_val > max_label) max_label = int_val;
                }
                if (all_int_labels && max_label < 100) {
                    is_classification = true;
                    num_classes = max_label + 1;
                }
            }

            int n_samples = X_list.size();
            std::vector<double> loss_history;

            for (int epoch = 0; epoch < epochs; epoch++) {
                double epoch_loss = 0.0;
                int n_batches = 0;

                // Shuffle indices
                std::vector<int> indices(n_samples);
                for (int i = 0; i < n_samples; i++) indices[i] = i;
                std::shuffle(indices.begin(), indices.end(), std::default_random_engine(std::rand()));

                // Mini-batch training
                for (int batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
                    int batch_end = std::min(batch_start + batch_size, n_samples);
                    double batch_loss = 0.0;

                    optimizer.zeroGrad();

                    for (int i = batch_start; i < batch_end; i++) {
                        int idx = indices[i];

                        // Convert input
                        GradTensorPtr x;
                        if (X_list[idx].isTensor()) {
                            auto t = X_list[idx].asTensor();
                            x = GradTensor::create(t->data, t->shape, true);
                        } else if (X_list[idx].isList()) {
                            std::vector<double> data;
                            for (const auto& v : X_list[idx].asList()) {
                                data.push_back(v.asNumber());
                            }
                            int size = data.size();
                            x = GradTensor::create(std::move(data), {size}, true);
                        }

                        // Forward pass
                        auto pred = network->forward(x);

                        // Convert target
                        GradTensorPtr target;
                        if (is_classification) {
                            // One-hot encode if needed
                            std::vector<double> one_hot(num_classes, 0.0);
                            if (y_list[idx].isList()) {
                                for (size_t j = 0; j < y_list[idx].asList().size(); j++) {
                                    one_hot[j] = y_list[idx].asList()[j].asNumber();
                                }
                            } else {
                                int label = static_cast<int>(y_list[idx].asNumber());
                                one_hot[label] = 1.0;
                            }
                            target = GradTensor::create(std::move(one_hot), {num_classes}, false);

                            // Apply softmax and cross-entropy
                            auto softmax_pred = pred->softmax();
                            auto loss = cross_entropy_loss(softmax_pred, target);
                            batch_loss += loss->data[0];
                            loss->backward();
                        } else {
                            // Regression - MSE loss
                            std::vector<double> target_data;
                            if (y_list[idx].isList()) {
                                for (const auto& v : y_list[idx].asList()) {
                                    target_data.push_back(v.asNumber());
                                }
                            } else if (y_list[idx].isTensor()) {
                                auto t = y_list[idx].asTensor();
                                for (double d : t->data) {
                                    target_data.push_back(d);
                                }
                            } else {
                                target_data.push_back(y_list[idx].asNumber());
                            }

                            int target_size = static_cast<int>(target_data.size());
                            target = GradTensor::create(std::move(target_data),
                                                        {target_size}, false);

                            auto loss = mse_loss(pred, target);
                            batch_loss += loss->data[0];
                            loss->backward();
                        }
                    }

                    // Update weights
                    optimizer.step();

                    epoch_loss += batch_loss;
                    n_batches++;
                }

                epoch_loss /= n_samples;
                loss_history.push_back(epoch_loss);

                if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
                    std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                              << " - Loss: " << epoch_loss << std::endl;
                }
            }

            // Return final loss
            return Value(loss_history.back());
        });
        env->define("nn_train", Value(fn));
    }

    static void registerNNPredict(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("nn_predict", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("nn_predict() requires network name");
            }

            std::string name = args[0].asString();
            auto& networks = getNetworks();
            if (networks.find(name) == networks.end()) {
                throw std::runtime_error("Network not found: " + name);
            }
            auto network = networks[name];

            // Handle single sample or batch
            if (args[1].isList() && !args[1].asList().empty() && args[1].asList()[0].isList()) {
                // Batch prediction
                std::vector<Value> predictions;
                for (const auto& sample : args[1].asList()) {
                    GradTensorPtr input;
                    if (sample.isTensor()) {
                        auto t = sample.asTensor();
                        input = GradTensor::create(t->data, t->shape, false);
                    } else if (sample.isList()) {
                        std::vector<double> data;
                        for (const auto& v : sample.asList()) {
                            data.push_back(v.asNumber());
                        }
                        int sz_tmp = static_cast<int>(data.size());
                        input = GradTensor::create(std::move(data), {sz_tmp}, false);
                    }

                    auto output = network->forward(input);
                    predictions.push_back(Value(Tensor(output->data, output->shape)));
                }
                return Value(std::move(predictions));
            } else {
                // Single sample
                GradTensorPtr input;
                if (args[1].isTensor()) {
                    auto t = args[1].asTensor();
                    input = GradTensor::create(t->data, t->shape, false);
                } else if (args[1].isList()) {
                    std::vector<double> data;
                    for (const auto& v : args[1].asList()) {
                        data.push_back(v.asNumber());
                    }
                    int sz = static_cast<int>(data.size());
                    input = GradTensor::create(std::move(data), {sz}, false);
                }

                auto output = network->forward(input);
                return Value(Tensor(output->data, output->shape));
            }
        });
        env->define("nn_predict", Value(fn));
    }

    static void registerNNSave(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("nn_save", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString() || !args[1].isString()) {
                throw std::runtime_error("nn_save(network, filename)");
            }

            std::string name = args[0].asString();
            std::string filename = args[1].asString();

            auto& networks = getNetworks();
            if (networks.find(name) == networks.end()) {
                throw std::runtime_error("Network not found: " + name);
            }
            auto network = networks[name];

            std::ofstream file(filename, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Could not open file for writing: " + filename);
            }

            // Write number of layers
            int num_layers = network->layers.size();
            file.write(reinterpret_cast<char*>(&num_layers), sizeof(int));

            // Write activation
            int act_len = network->activation.length();
            file.write(reinterpret_cast<char*>(&act_len), sizeof(int));
            file.write(network->activation.c_str(), act_len);

            // Write each layer's weights and biases
            for (const auto& layer : network->layers) {
                // Write weights shape and data
                int w_dims = layer->weights->shape.size();
                file.write(reinterpret_cast<char*>(&w_dims), sizeof(int));
                for (int dim : layer->weights->shape) {
                    file.write(reinterpret_cast<char*>(&dim), sizeof(int));
                }
                int w_size = layer->weights->data.size();
                file.write(reinterpret_cast<char*>(&w_size), sizeof(int));
                file.write(reinterpret_cast<char*>(layer->weights->data.data()),
                           w_size * sizeof(double));

                // Write bias shape and data
                int b_dims = layer->bias->shape.size();
                file.write(reinterpret_cast<char*>(&b_dims), sizeof(int));
                for (int dim : layer->bias->shape) {
                    file.write(reinterpret_cast<char*>(&dim), sizeof(int));
                }
                int b_size = layer->bias->data.size();
                file.write(reinterpret_cast<char*>(&b_size), sizeof(int));
                file.write(reinterpret_cast<char*>(layer->bias->data.data()),
                           b_size * sizeof(double));
            }

            return Value(true);
        });
        env->define("nn_save", Value(fn));
    }

    static void registerNNLoad(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("nn_load", 1, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isString()) {
                throw std::runtime_error("nn_load(filename)");
            }

            std::string filename = args[0].asString();

            std::ifstream file(filename, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Could not open file for reading: " + filename);
            }

            // Read number of layers
            int num_layers;
            file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));

            // Read activation
            int act_len;
            file.read(reinterpret_cast<char*>(&act_len), sizeof(int));
            std::string activation(act_len, ' ');
            file.read(&activation[0], act_len);

            // Read layer sizes from weights
            std::vector<int> layer_sizes;
            std::vector<std::pair<GradTensorPtr, GradTensorPtr>> weights_biases;

            for (int l = 0; l < num_layers; l++) {
                // Read weights
                int w_dims;
                file.read(reinterpret_cast<char*>(&w_dims), sizeof(int));
                std::vector<int> w_shape(w_dims);
                for (int i = 0; i < w_dims; i++) {
                    file.read(reinterpret_cast<char*>(&w_shape[i]), sizeof(int));
                }
                int w_size;
                file.read(reinterpret_cast<char*>(&w_size), sizeof(int));
                std::vector<double> w_data(w_size);
                file.read(reinterpret_cast<char*>(w_data.data()), w_size * sizeof(double));

                // Read bias
                int b_dims;
                file.read(reinterpret_cast<char*>(&b_dims), sizeof(int));
                std::vector<int> b_shape(b_dims);
                for (int i = 0; i < b_dims; i++) {
                    file.read(reinterpret_cast<char*>(&b_shape[i]), sizeof(int));
                }
                int b_size;
                file.read(reinterpret_cast<char*>(&b_size), sizeof(int));
                std::vector<double> b_data(b_size);
                file.read(reinterpret_cast<char*>(b_data.data()), b_size * sizeof(double));

                auto weights = GradTensor::create(std::move(w_data), w_shape, true);
                auto bias = GradTensor::create(std::move(b_data), b_shape, true);
                weights_biases.push_back({weights, bias});

                if (l == 0) {
                    layer_sizes.push_back(w_shape[0]);
                }
                layer_sizes.push_back(w_shape[1]);
            }

            // Create network
            auto network = std::make_shared<GradNetwork>(layer_sizes, activation);

            // Copy weights
            for (int l = 0; l < num_layers; l++) {
                network->layers[l]->weights = weights_biases[l].first;
                network->layers[l]->bias = weights_biases[l].second;
            }

            // Generate unique name
            static int counter = 0;
            std::string name = "loaded_network_" + std::to_string(counter++);
            getNetworks()[name] = network;

            return Value(name);
        });
        env->define("nn_load", Value(fn));
    }

    static void registerArgmax(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("argmax", 1, [](const std::vector<Value>& args) -> Value {
            std::vector<double> data;
            if (args[0].isTensor()) {
                data = args[0].asTensor()->data;
            } else if (args[0].isList()) {
                for (const auto& v : args[0].asList()) {
                    data.push_back(v.asNumber());
                }
            } else {
                throw std::runtime_error("argmax() requires tensor or list");
            }

            int max_idx = 0;
            double max_val = data[0];
            for (size_t i = 1; i < data.size(); i++) {
                if (data[i] > max_val) {
                    max_val = data[i];
                    max_idx = i;
                }
            }
            return Value(static_cast<double>(max_idx));
        });
        env->define("argmax", Value(fn));
    }

    static void registerTensorGet(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("tensor_get", 2, [](const std::vector<Value>& args) -> Value {
            if (!args[0].isTensor()) {
                throw std::runtime_error("tensor_get() requires tensor");
            }
            auto tensor = args[0].asTensor();
            int idx = static_cast<int>(args[1].asNumber());
            if (idx < 0 || idx >= (int)tensor->data.size()) {
                throw std::runtime_error("Index out of bounds");
            }
            return Value(tensor->data[idx]);
        });
        env->define("tensor_get", Value(fn));
    }

    // data_prepare(csv_data, label_column, max_samples, scale_factor)
    // Efficiently splits CSV data into [X, y] with optional normalization
    static void registerDataPrepare(std::shared_ptr<Environment> env) {
        auto fn = std::make_shared<NativeFunction>("data_prepare", -1, [](const std::vector<Value>& args) -> Value {
            if (args.size() < 1 || !args[0].isList()) {
                throw std::runtime_error("data_prepare(csv_data, [label_col=0], [max_samples=-1], [scale=1.0])");
            }

            const auto& rows = args[0].asList();
            int label_col = args.size() > 1 ? static_cast<int>(args[1].asNumber()) : 0;
            int max_samples = args.size() > 2 ? static_cast<int>(args[2].asNumber()) : -1;
            double scale = args.size() > 3 ? args[3].asNumber() : 1.0;

            int n = static_cast<int>(rows.size());
            if (max_samples > 0 && max_samples < n) n = max_samples;

            std::vector<Value> X_list;
            std::vector<Value> y_list;
            X_list.reserve(n);
            y_list.reserve(n);

            for (int i = 0; i < n; i++) {
                const auto& row = rows[i].asList();
                int ncols = static_cast<int>(row.size());

                // Extract label
                y_list.push_back(Value(static_cast<double>(static_cast<int>(row[label_col].asNumber()))));

                // Extract features (all columns except label)
                std::vector<Value> features;
                features.reserve(ncols - 1);
                for (int j = 0; j < ncols; j++) {
                    if (j != label_col) {
                        features.push_back(Value(row[j].asNumber() * scale));
                    }
                }
                X_list.push_back(Value(std::move(features)));
            }

            // Return [X, y] as a list of two elements
            std::vector<Value> result;
            result.push_back(Value(std::move(X_list)));
            result.push_back(Value(std::move(y_list)));
            return Value(std::move(result));
        });
        env->define("data_prepare", Value(fn));
    }

    // Helper function
    static std::vector<int> toShape(const Value& v) {
        std::vector<int> shapeVec;
        if (v.isList()) {
            for (const auto& dim : v.asList()) {
                shapeVec.push_back(static_cast<int>(dim.asNumber()));
            }
        } else if (v.isTensor()) {
            // Handle 1D tensor as shape specification
            auto tensor = v.asTensor();
            for (double d : tensor->data) {
                shapeVec.push_back(static_cast<int>(d));
            }
        } else if (v.isNumber()) {
            shapeVec.push_back(static_cast<int>(v.asNumber()));
        }
        return shapeVec;
    }
};

} // namespace duckt

#endif // DUCKT_BUILTINS_HPP
