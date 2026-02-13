#ifndef DUCKT_LEXER_HPP
#define DUCKT_LEXER_HPP

#include "token.hpp"
#include <string>
#include <vector>
#include <stack>
#include <unordered_map>

namespace duckt {

class Lexer {
public:
    explicit Lexer(std::string source);

    std::vector<Token> tokenize();
    bool hasError() const { return hadError_; }
    const std::vector<std::string>& errors() const { return errors_; }

private:
    std::string source_;
    std::vector<Token> tokens_;
    std::vector<std::string> errors_;

    size_t start_ = 0;
    size_t current_ = 0;
    int line_ = 1;
    int column_ = 1;
    int startColumn_ = 1;

    std::stack<int> indentStack_;
    bool atLineStart_ = true;
    bool hadError_ = false;
    int bracketDepth_ = 0;  // Track [], (), {} nesting

    static const std::unordered_map<std::string, TokenType> keywords_;

    // Scanning methods
    bool isAtEnd() const;
    char peek() const;
    char peekNext() const;
    char advance();
    bool match(char expected);

    // Token creation
    void addToken(TokenType type);
    void addToken(TokenType type, long long value);
    void addToken(TokenType type, double value);

    // Lexing methods
    void scanToken();
    void handleIndentation();
    void skipWhitespace();
    void skipComment();
    void scanString();
    void scanNumber();
    void scanIdentifier();

    // Error handling
    void error(const std::string& message);

    // Helpers
    bool isDigit(char c) const;
    bool isAlpha(char c) const;
    bool isAlphaNumeric(char c) const;
};

} // namespace duckt

#endif // DUCKT_LEXER_HPP
