#ifndef DUCKT_PARSER_HPP
#define DUCKT_PARSER_HPP

#include "token.hpp"
#include "ast.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace duckt {

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);

    Program parse();
    bool hasError() const { return hadError_; }
    const std::vector<std::string>& errors() const { return errors_; }

private:
    std::vector<Token> tokens_;
    std::vector<std::string> errors_;
    size_t current_ = 0;
    bool hadError_ = false;

    // Token navigation
    const Token& peek() const;
    const Token& previous() const;
    const Token& advance();
    bool isAtEnd() const;
    bool check(TokenType type) const;
    bool match(TokenType type);
    bool match(std::initializer_list<TokenType> types);
    Token consume(TokenType type, const std::string& message);

    // Error handling
    void error(const Token& token, const std::string& message);
    void synchronize();

    // Skip newlines helper
    void skipNewlines();

    // Parsing methods
    StmtPtr parseStatement();
    StmtPtr parseLetStatement();
    StmtPtr parseFnStatement();
    StmtPtr parseIfStatement();
    StmtPtr parseWhileStatement();
    StmtPtr parseForStatement();
    StmtPtr parseReturnStatement();
    StmtPtr parseModelStatement();
    StmtPtr parseAgentStatement();
    StmtPtr parseTrainStatement();
    StmtPtr parseLayerStatement();
    StmtPtr parseStateStatement();
    StmtPtr parseToolStatement();
    StmtPtr parseExpressionStatement();

    std::vector<StmtPtr> parseBlock();

    // Expression parsing with precedence
    ExprPtr parseExpression();
    ExprPtr parseOr();
    ExprPtr parseAnd();
    ExprPtr parseEquality();
    ExprPtr parseComparison();
    ExprPtr parseTerm();
    ExprPtr parseFactor();
    ExprPtr parseMatmul();
    ExprPtr parseUnary();
    ExprPtr parseCall();
    ExprPtr parsePrimary();
    ExprPtr parseListOrTensor();

    // Helpers
    std::vector<std::string> parseParameters();
    std::vector<ExprPtr> parseArguments();
};

} // namespace duckt

#endif // DUCKT_PARSER_HPP
