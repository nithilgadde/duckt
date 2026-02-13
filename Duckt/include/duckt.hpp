#ifndef DUCKT_HPP
#define DUCKT_HPP

// Public API for embedding Duckt in other applications

#include "../src/token.hpp"
#include "../src/lexer.hpp"
#include "../src/ast.hpp"
#include "../src/parser.hpp"
#include "../src/types.hpp"
#include "../src/environment.hpp"
#include "../src/interpreter.hpp"
#include "../src/builtins.hpp"

namespace duckt {

/**
 * Run a Duckt program from source code
 *
 * @param source The Duckt source code
 * @return true if execution succeeded, false if there were errors
 */
inline bool run(const std::string& source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    if (lexer.hasError()) {
        return false;
    }

    Parser parser(std::move(tokens));
    Program program = parser.parse();

    if (parser.hasError()) {
        return false;
    }

    try {
        Interpreter interpreter;
        interpreter.interpret(program);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * Create an interpreter with all built-ins registered
 */
inline Interpreter createInterpreter() {
    return Interpreter();
}

/**
 * Evaluate an expression and return the result
 */
inline Value eval(const std::string& source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    if (lexer.hasError()) {
        throw std::runtime_error("Lexer error");
    }

    Parser parser(std::move(tokens));
    Program program = parser.parse();

    if (parser.hasError()) {
        throw std::runtime_error("Parser error");
    }

    Interpreter interpreter;

    if (program.statements.empty()) {
        return Value();
    }

    // If single expression statement, return its value
    if (program.statements.size() == 1) {
        if (auto* exprStmt = dynamic_cast<ExprStmt*>(program.statements[0].get())) {
            return interpreter.evaluate(exprStmt->expr.get());
        }
    }

    interpreter.interpret(program);
    return Value();
}

} // namespace duckt

#endif // DUCKT_HPP
