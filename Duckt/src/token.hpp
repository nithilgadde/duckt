#ifndef DUCKT_TOKEN_HPP
#define DUCKT_TOKEN_HPP

#include <string>
#include <variant>

namespace duckt {

enum class TokenType {
    // Literals
    INTEGER,
    FLOAT,
    STRING,
    IDENTIFIER,

    // Keywords
    LET,
    FN,
    IF,
    ELSE,
    FOR,
    WHILE,
    RETURN,
    LAYER,
    MODEL,
    AGENT,
    TOOL,
    TRAIN,
    TENSOR,
    TRUE,
    FALSE,
    NONE,
    IN,
    ON,
    WITH,
    EPOCHS,
    LR,
    STATE,
    AND,
    OR,
    NOT,

    // Operators
    PLUS,           // +
    MINUS,          // -
    STAR,           // *
    SLASH,          // /
    AT,             // @ (matrix multiplication)
    EQUAL,          // =
    EQUAL_EQUAL,    // ==
    BANG_EQUAL,     // !=
    LESS,           // <
    GREATER,        // >
    LESS_EQUAL,     // <=
    GREATER_EQUAL,  // >=

    // Delimiters
    LPAREN,         // (
    RPAREN,         // )
    LBRACKET,       // [
    RBRACKET,       // ]
    LBRACE,         // {
    RBRACE,         // }
    COLON,          // :
    COMMA,          // ,
    ARROW,          // ->
    DOT,            // .

    // Indentation
    INDENT,
    DEDENT,
    NEWLINE,

    // Special
    END_OF_FILE,
    ERROR
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;

    // For numeric literals
    std::variant<std::monostate, long long, double> literal;

    Token(TokenType t, std::string lex, int ln, int col)
        : type(t), lexeme(std::move(lex)), line(ln), column(col) {}

    Token(TokenType t, std::string lex, int ln, int col, long long val)
        : type(t), lexeme(std::move(lex)), line(ln), column(col), literal(val) {}

    Token(TokenType t, std::string lex, int ln, int col, double val)
        : type(t), lexeme(std::move(lex)), line(ln), column(col), literal(val) {}
};

inline std::string tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::INTEGER: return "INTEGER";
        case TokenType::FLOAT: return "FLOAT";
        case TokenType::STRING: return "STRING";
        case TokenType::IDENTIFIER: return "IDENTIFIER";
        case TokenType::LET: return "LET";
        case TokenType::FN: return "FN";
        case TokenType::IF: return "IF";
        case TokenType::ELSE: return "ELSE";
        case TokenType::FOR: return "FOR";
        case TokenType::WHILE: return "WHILE";
        case TokenType::RETURN: return "RETURN";
        case TokenType::LAYER: return "LAYER";
        case TokenType::MODEL: return "MODEL";
        case TokenType::AGENT: return "AGENT";
        case TokenType::TOOL: return "TOOL";
        case TokenType::TRAIN: return "TRAIN";
        case TokenType::TENSOR: return "TENSOR";
        case TokenType::TRUE: return "TRUE";
        case TokenType::FALSE: return "FALSE";
        case TokenType::NONE: return "NONE";
        case TokenType::IN: return "IN";
        case TokenType::ON: return "ON";
        case TokenType::WITH: return "WITH";
        case TokenType::EPOCHS: return "EPOCHS";
        case TokenType::LR: return "LR";
        case TokenType::STATE: return "STATE";
        case TokenType::AND: return "AND";
        case TokenType::OR: return "OR";
        case TokenType::NOT: return "NOT";
        case TokenType::PLUS: return "PLUS";
        case TokenType::MINUS: return "MINUS";
        case TokenType::STAR: return "STAR";
        case TokenType::SLASH: return "SLASH";
        case TokenType::AT: return "AT";
        case TokenType::EQUAL: return "EQUAL";
        case TokenType::EQUAL_EQUAL: return "EQUAL_EQUAL";
        case TokenType::BANG_EQUAL: return "BANG_EQUAL";
        case TokenType::LESS: return "LESS";
        case TokenType::GREATER: return "GREATER";
        case TokenType::LESS_EQUAL: return "LESS_EQUAL";
        case TokenType::GREATER_EQUAL: return "GREATER_EQUAL";
        case TokenType::LPAREN: return "LPAREN";
        case TokenType::RPAREN: return "RPAREN";
        case TokenType::LBRACKET: return "LBRACKET";
        case TokenType::RBRACKET: return "RBRACKET";
        case TokenType::LBRACE: return "LBRACE";
        case TokenType::RBRACE: return "RBRACE";
        case TokenType::COLON: return "COLON";
        case TokenType::COMMA: return "COMMA";
        case TokenType::ARROW: return "ARROW";
        case TokenType::DOT: return "DOT";
        case TokenType::INDENT: return "INDENT";
        case TokenType::DEDENT: return "DEDENT";
        case TokenType::NEWLINE: return "NEWLINE";
        case TokenType::END_OF_FILE: return "EOF";
        case TokenType::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

} // namespace duckt

#endif // DUCKT_TOKEN_HPP
