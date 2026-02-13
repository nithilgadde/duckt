#ifndef DUCKT_ENVIRONMENT_HPP
#define DUCKT_ENVIRONMENT_HPP

#include "types.hpp"
#include <unordered_map>
#include <memory>
#include <stdexcept>

namespace duckt {

class Environment : public std::enable_shared_from_this<Environment> {
public:
    std::shared_ptr<Environment> parent;

    Environment() : parent(nullptr) {}
    explicit Environment(std::shared_ptr<Environment> enclosing)
        : parent(std::move(enclosing)) {}

    // Define a new variable in current scope
    void define(const std::string& name, Value value) {
        variables_[name] = std::move(value);
    }

    // Get a variable, searching up the scope chain
    Value get(const std::string& name) const {
        auto it = variables_.find(name);
        if (it != variables_.end()) {
            return it->second;
        }

        if (parent) {
            return parent->get(name);
        }

        throw std::runtime_error("Undefined variable: " + name);
    }

    // Check if a variable exists
    bool has(const std::string& name) const {
        auto it = variables_.find(name);
        if (it != variables_.end()) {
            return true;
        }

        if (parent) {
            return parent->has(name);
        }

        return false;
    }

    // Assign to an existing variable
    void assign(const std::string& name, Value value) {
        auto it = variables_.find(name);
        if (it != variables_.end()) {
            it->second = std::move(value);
            return;
        }

        if (parent) {
            parent->assign(name, std::move(value));
            return;
        }

        throw std::runtime_error("Undefined variable: " + name);
    }

    // Get a reference to modify in place (for list/tensor mutation)
    Value& getRef(const std::string& name) {
        auto it = variables_.find(name);
        if (it != variables_.end()) {
            return it->second;
        }

        if (parent) {
            return parent->getRef(name);
        }

        throw std::runtime_error("Undefined variable: " + name);
    }

    // Create a child environment
    std::shared_ptr<Environment> createChild() {
        return std::make_shared<Environment>(shared_from_this());
    }

private:
    std::unordered_map<std::string, Value> variables_;
};

} // namespace duckt

#endif // DUCKT_ENVIRONMENT_HPP
