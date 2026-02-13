#include "interpreter.hpp"
#include "builtins.hpp"
#include <iostream>
#include <algorithm>

namespace duckt {

Interpreter::Interpreter() {
    globals_ = std::make_shared<Environment>();
    environment_ = globals_;

    // Register built-in functions
    Builtins::registerAll(globals_);
}

void Interpreter::interpret(const Program& program) {
    for (const auto& stmt : program.statements) {
        execute(stmt.get());
    }
}

Value Interpreter::evaluate(Expr* expr) {
    if (auto* e = dynamic_cast<NumberExpr*>(expr)) return evalNumber(e);
    if (auto* e = dynamic_cast<StringExpr*>(expr)) return evalString(e);
    if (auto* e = dynamic_cast<BoolExpr*>(expr)) return evalBool(e);
    if (auto* e = dynamic_cast<NoneExpr*>(expr)) return evalNone(e);
    if (auto* e = dynamic_cast<IdentifierExpr*>(expr)) return evalIdentifier(e);
    if (auto* e = dynamic_cast<BinaryExpr*>(expr)) return evalBinary(e);
    if (auto* e = dynamic_cast<UnaryExpr*>(expr)) return evalUnary(e);
    if (auto* e = dynamic_cast<CallExpr*>(expr)) return evalCall(e);
    if (auto* e = dynamic_cast<IndexExpr*>(expr)) return evalIndex(e);
    if (auto* e = dynamic_cast<MemberExpr*>(expr)) return evalMember(e);
    if (auto* e = dynamic_cast<ListExpr*>(expr)) return evalList(e);
    if (auto* e = dynamic_cast<TensorExpr*>(expr)) return evalTensor(e);

    throw std::runtime_error("Unknown expression type");
}

void Interpreter::execute(Stmt* stmt) {
    if (auto* s = dynamic_cast<ExprStmt*>(stmt)) return execExprStmt(s);
    if (auto* s = dynamic_cast<LetStmt*>(stmt)) return execLet(s);
    if (auto* s = dynamic_cast<AssignStmt*>(stmt)) return execAssign(s);
    if (auto* s = dynamic_cast<IfStmt*>(stmt)) return execIf(s);
    if (auto* s = dynamic_cast<WhileStmt*>(stmt)) return execWhile(s);
    if (auto* s = dynamic_cast<ForStmt*>(stmt)) return execFor(s);
    if (auto* s = dynamic_cast<FnStmt*>(stmt)) return execFn(s);
    if (auto* s = dynamic_cast<ReturnStmt*>(stmt)) return execReturn(s);
    if (auto* s = dynamic_cast<ModelStmt*>(stmt)) return execModel(s);
    if (auto* s = dynamic_cast<AgentStmt*>(stmt)) return execAgent(s);
    if (auto* s = dynamic_cast<TrainStmt*>(stmt)) return execTrain(s);
    if (auto* s = dynamic_cast<LayerStmt*>(stmt)) return execLayer(s);
    if (auto* s = dynamic_cast<StateStmt*>(stmt)) return execState(s);
    if (auto* s = dynamic_cast<ToolStmt*>(stmt)) return execTool(s);

    throw std::runtime_error("Unknown statement type");
}

// ============ Expression Evaluation ============

Value Interpreter::evalNumber(NumberExpr* expr) {
    return Value(expr->value);
}

Value Interpreter::evalString(StringExpr* expr) {
    return Value(expr->value);
}

Value Interpreter::evalBool(BoolExpr* expr) {
    return Value(expr->value);
}

Value Interpreter::evalNone(NoneExpr*) {
    return Value();
}

Value Interpreter::evalIdentifier(IdentifierExpr* expr) {
    return environment_->get(expr->name);
}

Value Interpreter::evalBinary(BinaryExpr* expr) {
    Value left = evaluate(expr->left.get());
    Value right = evaluate(expr->right.get());

    switch (expr->op) {
        case BinaryOp::ADD:
            if (left.isNumber() && right.isNumber()) {
                return Value(left.asNumber() + right.asNumber());
            }
            if (left.isString() && right.isString()) {
                return Value(left.asString() + right.asString());
            }
            if (left.isString()) {
                return Value(left.asString() + right.toString());
            }
            if (left.isTensor() && right.isTensor()) {
                return Value(left.asTensor()->add(*right.asTensor()));
            }
            if (left.isTensor() && right.isNumber()) {
                return Value(left.asTensor()->addScalar(right.asNumber()));
            }
            if (left.isList() && right.isList()) {
                std::vector<Value> result = left.asList();
                const auto& rightList = right.asList();
                result.insert(result.end(), rightList.begin(), rightList.end());
                return Value(std::move(result));
            }
            // Handle List + Tensor: flatten tensor elements into the list
            if (left.isList() && right.isTensor()) {
                std::vector<Value> result = left.asList();
                auto tensor = right.asTensor();
                for (double d : tensor->data) {
                    result.push_back(Value(d));
                }
                return Value(std::move(result));
            }
            if (left.isTensor() && right.isList()) {
                std::vector<Value> result;
                auto tensor = left.asTensor();
                for (double d : tensor->data) {
                    result.push_back(Value(d));
                }
                const auto& rightList = right.asList();
                result.insert(result.end(), rightList.begin(), rightList.end());
                return Value(std::move(result));
            }
            throw std::runtime_error("Invalid operands for +");

        case BinaryOp::SUB:
            if (left.isNumber() && right.isNumber()) {
                return Value(left.asNumber() - right.asNumber());
            }
            if (left.isTensor() && right.isTensor()) {
                return Value(left.asTensor()->sub(*right.asTensor()));
            }
            throw std::runtime_error("Invalid operands for -");

        case BinaryOp::MUL:
            if (left.isNumber() && right.isNumber()) {
                return Value(left.asNumber() * right.asNumber());
            }
            if (left.isTensor() && right.isTensor()) {
                return Value(left.asTensor()->mul(*right.asTensor()));
            }
            if (left.isTensor() && right.isNumber()) {
                return Value(left.asTensor()->mulScalar(right.asNumber()));
            }
            if (left.isNumber() && right.isTensor()) {
                return Value(right.asTensor()->mulScalar(left.asNumber()));
            }
            throw std::runtime_error("Invalid operands for *");

        case BinaryOp::DIV:
            if (left.isNumber() && right.isNumber()) {
                if (right.asNumber() == 0) {
                    throw std::runtime_error("Division by zero");
                }
                return Value(left.asNumber() / right.asNumber());
            }
            if (left.isTensor() && right.isTensor()) {
                return Value(left.asTensor()->div(*right.asTensor()));
            }
            if (left.isTensor() && right.isNumber()) {
                if (right.asNumber() == 0) {
                    throw std::runtime_error("Division by zero");
                }
                return Value(left.asTensor()->mulScalar(1.0 / right.asNumber()));
            }
            if (left.isList() && right.isNumber()) {
                if (right.asNumber() == 0) {
                    throw std::runtime_error("Division by zero");
                }
                double divisor = right.asNumber();
                std::vector<Value> result;
                for (const auto& v : left.asList()) {
                    if (v.isNumber()) {
                        result.push_back(Value(v.asNumber() / divisor));
                    } else {
                        result.push_back(v);
                    }
                }
                return Value(std::move(result));
            }
            throw std::runtime_error("Invalid operands for /");

        case BinaryOp::MATMUL:
            if (left.isTensor() && right.isTensor()) {
                return Value(left.asTensor()->matmul(*right.asTensor()));
            }
            throw std::runtime_error("Matrix multiplication requires tensors");

        case BinaryOp::EQ:
            return Value(left == right);

        case BinaryOp::NE:
            return Value(left != right);

        case BinaryOp::LT:
            if (left.isNumber() && right.isNumber()) {
                return Value(left.asNumber() < right.asNumber());
            }
            throw std::runtime_error("Invalid operands for <");

        case BinaryOp::GT:
            if (left.isNumber() && right.isNumber()) {
                return Value(left.asNumber() > right.asNumber());
            }
            throw std::runtime_error("Invalid operands for >");

        case BinaryOp::LE:
            if (left.isNumber() && right.isNumber()) {
                return Value(left.asNumber() <= right.asNumber());
            }
            throw std::runtime_error("Invalid operands for <=");

        case BinaryOp::GE:
            if (left.isNumber() && right.isNumber()) {
                return Value(left.asNumber() >= right.asNumber());
            }
            throw std::runtime_error("Invalid operands for >=");

        case BinaryOp::AND:
            return Value(left.isTruthy() && right.isTruthy());

        case BinaryOp::OR:
            return Value(left.isTruthy() || right.isTruthy());
    }

    throw std::runtime_error("Unknown binary operator");
}

Value Interpreter::evalUnary(UnaryExpr* expr) {
    Value operand = evaluate(expr->operand.get());

    switch (expr->op) {
        case UnaryOp::NEG:
            if (operand.isNumber()) {
                return Value(-operand.asNumber());
            }
            if (operand.isTensor()) {
                return Value(operand.asTensor()->mulScalar(-1));
            }
            throw std::runtime_error("Invalid operand for negation");

        case UnaryOp::NOT:
            return Value(!operand.isTruthy());
    }

    throw std::runtime_error("Unknown unary operator");
}

Value Interpreter::evalCall(CallExpr* expr) {
    Value callee = evaluate(expr->callee.get());

    std::vector<Value> args;
    for (const auto& arg : expr->args) {
        args.push_back(evaluate(arg.get()));
    }

    return call(callee, args);
}

Value Interpreter::call(const Value& callee, const std::vector<Value>& args) {
    if (callee.isNativeFunction()) {
        auto fn = callee.asNativeFunction();
        if (fn->arity >= 0 && static_cast<int>(args.size()) != fn->arity) {
            throw std::runtime_error("Expected " + std::to_string(fn->arity) +
                                     " arguments but got " + std::to_string(args.size()));
        }
        return fn->fn(args);
    }

    if (callee.isFunction()) {
        auto fn = callee.asFunction();

        if (args.size() != fn->params.size()) {
            throw std::runtime_error("Expected " + std::to_string(fn->params.size()) +
                                     " arguments but got " + std::to_string(args.size()));
        }

        auto fnEnv = std::make_shared<Environment>(fn->closure);

        for (size_t i = 0; i < fn->params.size(); i++) {
            fnEnv->define(fn->params[i], args[i]);
        }

        auto prevEnv = environment_;
        environment_ = fnEnv;
        try {
            // Execute function body
            for (auto* stmt : fn->body) {
                execute(stmt);
            }
            environment_ = prevEnv;
            return Value();  // No return statement
        } catch (const ReturnException& ret) {
            environment_ = prevEnv;  // Restore environment before returning
            return ret.value;
        }
    }

    if (callee.isLayer()) {
        auto layer = callee.asLayer();
        if (args.empty() || !args[0].isTensor()) {
            throw std::runtime_error("Layer expects tensor input");
        }
        return Value(layer->forward(*args[0].asTensor()));
    }

    if (callee.isModel()) {
        auto model = callee.asModel();
        if (args.empty() || !args[0].isTensor()) {
            throw std::runtime_error("Model expects tensor input");
        }
        return Value(model->forward(*args[0].asTensor()));
    }

    throw std::runtime_error("Can only call functions, layers, and models");
}

Value Interpreter::evalIndex(IndexExpr* expr) {
    Value object = evaluate(expr->object.get());
    Value index = evaluate(expr->index.get());

    if (object.isList()) {
        int idx = static_cast<int>(index.asNumber());
        const auto& list = object.asList();
        if (idx < 0) idx = list.size() + idx;
        if (idx < 0 || idx >= static_cast<int>(list.size())) {
            throw std::runtime_error("List index out of bounds");
        }
        return list[idx];
    }

    if (object.isString()) {
        int idx = static_cast<int>(index.asNumber());
        const auto& str = object.asString();
        if (idx < 0) idx = str.length() + idx;
        if (idx < 0 || idx >= static_cast<int>(str.length())) {
            throw std::runtime_error("String index out of bounds");
        }
        return Value(std::string(1, str[idx]));
    }

    if (object.isTensor()) {
        int idx = static_cast<int>(index.asNumber());
        const auto& tensor = object.asTensor();
        if (idx < 0 || idx >= static_cast<int>(tensor->data.size())) {
            throw std::runtime_error("Tensor index out of bounds");
        }
        return Value(tensor->data[idx]);
    }

    throw std::runtime_error("Cannot index " + object.typeName());
}

Value Interpreter::evalMember(MemberExpr* expr) {
    Value object = evaluate(expr->object.get());

    if (object.isAgent()) {
        auto agent = object.asAgent();

        // Check state
        auto stateIt = agent->state.find(expr->member);
        if (stateIt != agent->state.end()) {
            return stateIt->second;
        }

        // Check tools
        auto toolIt = agent->tools.find(expr->member);
        if (toolIt != agent->tools.end()) {
            return Value(toolIt->second);
        }

        // Check methods
        auto methodIt = agent->methods.find(expr->member);
        if (methodIt != agent->methods.end()) {
            return Value(methodIt->second);
        }

        throw std::runtime_error("Agent has no member '" + expr->member + "'");
    }

    if (object.isModel()) {
        auto model = object.asModel();

        // Check for forward function
        if (expr->member == "forward" && model->forwardFn) {
            return Value(model->forwardFn);
        }

        // Check layers by name
        for (const auto& layer : model->layers) {
            if (layer->name == expr->member) {
                return Value(layer);
            }
        }

        throw std::runtime_error("Model has no member '" + expr->member + "'");
    }

    if (object.isTensor()) {
        auto tensor = object.asTensor();

        if (expr->member == "shape") {
            std::vector<Value> shape;
            for (int dim : tensor->shape) {
                shape.push_back(Value(static_cast<double>(dim)));
            }
            return Value(std::move(shape));
        }

        if (expr->member == "T") {
            return Value(tensor->transpose());
        }

        throw std::runtime_error("Tensor has no member '" + expr->member + "'");
    }

    throw std::runtime_error("Cannot access member on " + object.typeName());
}

Value Interpreter::evalList(ListExpr* expr) {
    std::vector<Value> elements;
    for (const auto& elem : expr->elements) {
        elements.push_back(evaluate(elem.get()));
    }

    // Check if this should be a tensor
    if (isTensorLike(elements)) {
        return Value(listToTensor(elements));
    }

    return Value(std::move(elements));
}

Value Interpreter::evalTensor(TensorExpr* expr) {
    std::vector<Value> elements;
    for (const auto& elem : expr->elements) {
        elements.push_back(evaluate(elem.get()));
    }
    return Value(listToTensor(elements));
}

bool Interpreter::isTensorLike(const std::vector<Value>& elements) {
    if (elements.empty()) return false;

    // All numbers
    bool allNumbers = std::all_of(elements.begin(), elements.end(),
        [](const Value& v) { return v.isNumber(); });
    if (allNumbers) return true;

    // All tensors of same shape
    bool allTensors = std::all_of(elements.begin(), elements.end(),
        [](const Value& v) { return v.isTensor(); });
    if (allTensors) return true;

    // All lists that are tensor-like
    bool allLists = std::all_of(elements.begin(), elements.end(),
        [this](const Value& v) {
            return v.isList() && isTensorLike(v.asList());
        });
    return allLists;
}

Tensor Interpreter::listToTensor(const std::vector<Value>& elements) {
    if (elements.empty()) {
        return Tensor({}, {0});
    }

    // All numbers - 1D tensor
    if (elements[0].isNumber()) {
        std::vector<double> data;
        for (const auto& e : elements) {
            data.push_back(e.asNumber());
        }
        return Tensor(std::move(data), {static_cast<int>(elements.size())});
    }

    // All tensors - stack along new axis
    if (elements[0].isTensor()) {
        std::vector<double> data;
        std::vector<int> innerShape = elements[0].asTensor()->shape;

        for (const auto& e : elements) {
            const auto& t = e.asTensor();
            data.insert(data.end(), t->data.begin(), t->data.end());
        }

        std::vector<int> shape = {static_cast<int>(elements.size())};
        shape.insert(shape.end(), innerShape.begin(), innerShape.end());

        return Tensor(std::move(data), std::move(shape));
    }

    // Nested lists
    if (elements[0].isList()) {
        std::vector<Tensor> tensors;
        for (const auto& e : elements) {
            tensors.push_back(listToTensor(e.asList()));
        }

        std::vector<double> data;
        for (const auto& t : tensors) {
            data.insert(data.end(), t.data.begin(), t.data.end());
        }

        std::vector<int> shape = {static_cast<int>(elements.size())};
        if (!tensors.empty()) {
            shape.insert(shape.end(), tensors[0].shape.begin(), tensors[0].shape.end());
        }

        return Tensor(std::move(data), std::move(shape));
    }

    throw std::runtime_error("Cannot convert list to tensor");
}

// ============ Statement Execution ============

void Interpreter::execExprStmt(ExprStmt* stmt) {
    evaluate(stmt->expr.get());
}

void Interpreter::execLet(LetStmt* stmt) {
    Value value = evaluate(stmt->value.get());
    environment_->define(stmt->name, std::move(value));
}

void Interpreter::execAssign(AssignStmt* stmt) {
    Value value = evaluate(stmt->value.get());

    if (auto* ident = dynamic_cast<IdentifierExpr*>(stmt->target.get())) {
        environment_->assign(ident->name, std::move(value));
    } else if (auto* index = dynamic_cast<IndexExpr*>(stmt->target.get())) {
        // Handle list/tensor element assignment
        Value object = evaluate(index->object.get());
        Value idx = evaluate(index->index.get());

        if (object.isList()) {
            // Need to get reference and modify
            if (auto* objIdent = dynamic_cast<IdentifierExpr*>(index->object.get())) {
                Value& listRef = environment_->getRef(objIdent->name);
                int i = static_cast<int>(idx.asNumber());
                auto& list = listRef.asList();
                if (i < 0) i = list.size() + i;
                if (i < 0 || i >= static_cast<int>(list.size())) {
                    throw std::runtime_error("List index out of bounds");
                }
                list[i] = std::move(value);
            }
        } else if (object.isTensor()) {
            if (auto* objIdent = dynamic_cast<IdentifierExpr*>(index->object.get())) {
                Value& tensorRef = environment_->getRef(objIdent->name);
                int i = static_cast<int>(idx.asNumber());
                auto tensor = tensorRef.asTensor();
                if (i < 0 || i >= static_cast<int>(tensor->data.size())) {
                    throw std::runtime_error("Tensor index out of bounds");
                }
                tensor->data[i] = value.asNumber();
            }
        }
    } else if (auto* member = dynamic_cast<MemberExpr*>(stmt->target.get())) {
        // Handle agent state assignment
        Value object = evaluate(member->object.get());
        if (object.isAgent()) {
            auto agent = object.asAgent();
            agent->state[member->member] = std::move(value);
        } else {
            throw std::runtime_error("Cannot assign to member of " + object.typeName());
        }
    } else {
        throw std::runtime_error("Invalid assignment target");
    }
}

void Interpreter::execIf(IfStmt* stmt) {
    Value condition = evaluate(stmt->condition.get());

    if (condition.isTruthy()) {
        auto ifEnv = std::make_shared<Environment>(environment_);
        auto prevEnv = environment_;
        environment_ = ifEnv;
        for (const auto& s : stmt->thenBranch) {
            execute(s.get());
        }
        environment_ = prevEnv;
    } else if (!stmt->elseBranch.empty()) {
        auto elseEnv = std::make_shared<Environment>(environment_);
        auto prevEnv = environment_;
        environment_ = elseEnv;
        for (const auto& s : stmt->elseBranch) {
            execute(s.get());
        }
        environment_ = prevEnv;
    }
}

void Interpreter::execWhile(WhileStmt* stmt) {
    while (evaluate(stmt->condition.get()).isTruthy()) {
        auto loopEnv = std::make_shared<Environment>(environment_);
        auto prevEnv = environment_;
        environment_ = loopEnv;
        for (const auto& s : stmt->body) {
            execute(s.get());
        }
        environment_ = prevEnv;
    }
}

void Interpreter::execFor(ForStmt* stmt) {
    Value iterable = evaluate(stmt->iterable.get());

    std::vector<Value> items;

    if (iterable.isList()) {
        items = iterable.asList();
    } else if (iterable.isTensor()) {
        // Iterate over first dimension
        const auto& tensor = iterable.asTensor();
        if (tensor->shape.empty()) {
            throw std::runtime_error("Cannot iterate over 0-dimensional tensor");
        }

        int firstDim = tensor->shape[0];
        int stride = tensor->totalSize() / firstDim;

        for (int i = 0; i < firstDim; i++) {
            if (stride == 1) {
                items.push_back(Value(tensor->data[i]));
            } else {
                std::vector<double> subData(tensor->data.begin() + i * stride,
                                            tensor->data.begin() + (i + 1) * stride);
                std::vector<int> subShape(tensor->shape.begin() + 1, tensor->shape.end());
                items.push_back(Value(Tensor(std::move(subData), std::move(subShape))));
            }
        }
    } else {
        throw std::runtime_error("Cannot iterate over " + iterable.typeName());
    }

    for (const auto& item : items) {
        auto loopEnv = std::make_shared<Environment>(environment_);
        loopEnv->define(stmt->variable, item);

        auto prevEnv = environment_;
        environment_ = loopEnv;
        for (const auto& s : stmt->body) {
            execute(s.get());
        }
        environment_ = prevEnv;
    }
}

void Interpreter::execFn(FnStmt* stmt) {
    std::vector<Stmt*> bodyPtrs;
    for (const auto& s : stmt->body) {
        bodyPtrs.push_back(s.get());
    }

    auto fn = std::make_shared<Function>(
        stmt->name,
        stmt->params,
        bodyPtrs,
        environment_
    );

    environment_->define(stmt->name, Value(fn));
}

void Interpreter::execReturn(ReturnStmt* stmt) {
    Value value;
    if (stmt->value) {
        value = evaluate(stmt->value.get());
    }
    throw ReturnException(std::move(value));
}

void Interpreter::execModel(ModelStmt* stmt) {
    auto model = std::make_shared<Model>(stmt->name);

    // Create model environment
    auto modelEnv = std::make_shared<Environment>(environment_);

    auto prevEnv = environment_;
    environment_ = modelEnv;

    // Process body: layers and forward function
    for (const auto& s : stmt->body) {
        if (auto* layer = dynamic_cast<LayerStmt*>(s.get())) {
            execLayer(layer);
            Value layerVal = environment_->get(layer->name);
            if (layerVal.isLayer()) {
                layerVal.asLayer()->name = layer->name;
                model->layers.push_back(layerVal.asLayer());
            }
        } else if (auto* fn = dynamic_cast<FnStmt*>(s.get())) {
            if (fn->name == "forward") {
                std::vector<Stmt*> bodyPtrs;
                for (const auto& bodyStmt : fn->body) {
                    bodyPtrs.push_back(bodyStmt.get());
                }
                model->forwardFn = std::make_shared<Function>(
                    fn->name, fn->params, bodyPtrs, modelEnv
                );
            } else {
                execFn(fn);
            }
        }
    }

    environment_ = prevEnv;

    // Define model in outer environment
    environment_->define(stmt->name, Value(model));
}

void Interpreter::execAgent(AgentStmt* stmt) {
    auto agent = std::make_shared<Agent>(stmt->name);

    // Create agent environment with reference to agent for state access
    auto agentEnv = std::make_shared<Environment>(environment_);

    auto prevEnv = environment_;
    environment_ = agentEnv;

    // Process body: state, tools, functions
    for (const auto& s : stmt->body) {
        if (auto* state = dynamic_cast<StateStmt*>(s.get())) {
            Value initial = evaluate(state->initial.get());
            agent->state[state->name] = initial;
            agentEnv->define(state->name, initial);
        } else if (auto* tool = dynamic_cast<ToolStmt*>(s.get())) {
            std::vector<Stmt*> bodyPtrs;
            for (const auto& bodyStmt : tool->body) {
                bodyPtrs.push_back(bodyStmt.get());
            }
            auto fn = std::make_shared<Function>(
                tool->name, tool->params, bodyPtrs, agentEnv
            );
            agent->tools[tool->name] = fn;
            agentEnv->define(tool->name, Value(fn));
        } else if (auto* fn = dynamic_cast<FnStmt*>(s.get())) {
            std::vector<Stmt*> bodyPtrs;
            for (const auto& bodyStmt : fn->body) {
                bodyPtrs.push_back(bodyStmt.get());
            }
            auto func = std::make_shared<Function>(
                fn->name, fn->params, bodyPtrs, agentEnv
            );
            agent->methods[fn->name] = func;
            agentEnv->define(fn->name, Value(func));
        }
    }

    environment_ = prevEnv;

    // Define agent in outer environment
    environment_->define(stmt->name, Value(agent));
}

void Interpreter::execTrain(TrainStmt* stmt) {
    Value modelVal = environment_->get(stmt->modelName);
    if (!modelVal.isModel()) {
        throw std::runtime_error("Can only train models");
    }

    auto model = modelVal.asModel();

    Value dataVal = stmt->data ? evaluate(stmt->data.get()) : Value();
    int epochs = stmt->epochs ? static_cast<int>(evaluate(stmt->epochs.get()).asNumber()) : 1;
    double lr = stmt->learningRate ? evaluate(stmt->learningRate.get()).asNumber() : 0.001;

    std::cout << "Training " << model->name << " for " << epochs << " epochs with lr=" << lr << std::endl;

    // Simple training simulation (actual backprop would require autograd)
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;

        // In a real implementation, we would:
        // 1. Iterate over data batches
        // 2. Forward pass
        // 3. Compute loss
        // 4. Backward pass (autograd)
        // 5. Update weights
    }

    std::cout << "Training complete." << std::endl;
}

void Interpreter::execLayer(LayerStmt* stmt) {
    // Get the layer constructor
    Value constructor = environment_->get(stmt->layerType);

    if (!constructor.isNativeFunction()) {
        throw std::runtime_error("Unknown layer type: " + stmt->layerType);
    }

    // Evaluate arguments
    std::vector<Value> args;
    for (const auto& arg : stmt->args) {
        args.push_back(evaluate(arg.get()));
    }

    // Call constructor
    Value layer = call(constructor, args);

    // Set the name
    if (layer.isLayer()) {
        layer.asLayer()->name = stmt->name;
    }

    environment_->define(stmt->name, std::move(layer));
}

void Interpreter::execState(StateStmt* stmt) {
    Value initial = evaluate(stmt->initial.get());
    environment_->define(stmt->name, std::move(initial));
}

void Interpreter::execTool(ToolStmt* stmt) {
    std::vector<Stmt*> bodyPtrs;
    for (const auto& s : stmt->body) {
        bodyPtrs.push_back(s.get());
    }

    auto fn = std::make_shared<Function>(
        stmt->name,
        stmt->params,
        bodyPtrs,
        environment_
    );

    environment_->define(stmt->name, Value(fn));
}

void Interpreter::executeBlock(const std::vector<StmtPtr>& statements,
                               std::shared_ptr<Environment> env) {
    auto prevEnv = environment_;
    environment_ = env;

    for (const auto& stmt : statements) {
        execute(stmt.get());
    }

    environment_ = prevEnv;
}

} // namespace duckt
