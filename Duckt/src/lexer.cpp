#include "lexer.hpp"
#include <stdexcept>
#include <sstream>

namespace duckt {

const std::unordered_map<std::string, TokenType> Lexer::keywords_ = {
    {"let", TokenType::LET},
    {"fn", TokenType::FN},
    {"if", TokenType::IF},
    {"else", TokenType::ELSE},
    {"for", TokenType::FOR},
    {"while", TokenType::WHILE},
    {"return", TokenType::RETURN},
    {"layer", TokenType::LAYER},
    {"model", TokenType::MODEL},
    {"agent", TokenType::AGENT},
    {"tool", TokenType::TOOL},
    {"train", TokenType::TRAIN},
    {"tensor", TokenType::TENSOR},
    {"true", TokenType::TRUE},
    {"false", TokenType::FALSE},
    {"none", TokenType::NONE},
    {"in", TokenType::IN},
    {"on", TokenType::ON},
    {"with", TokenType::WITH},
    {"epochs", TokenType::EPOCHS},
    {"lr", TokenType::LR},
    {"state", TokenType::STATE},
    {"and", TokenType::AND},
    {"or", TokenType::OR},
    {"not", TokenType::NOT}
};

Lexer::Lexer(std::string source) : source_(std::move(source)) {
    indentStack_.push(0);  // Start with zero indentation
}

std::vector<Token> Lexer::tokenize() {
    while (!isAtEnd()) {
        start_ = current_;
        startColumn_ = column_;
        scanToken();
    }

    // Emit any remaining DEDENTs at end of file
    while (indentStack_.size() > 1) {
        indentStack_.pop();
        tokens_.emplace_back(TokenType::DEDENT, "", line_, column_);
    }

    tokens_.emplace_back(TokenType::END_OF_FILE, "", line_, column_);
    return tokens_;
}

bool Lexer::isAtEnd() const {
    return current_ >= source_.length();
}

char Lexer::peek() const {
    if (isAtEnd()) return '\0';
    return source_[current_];
}

char Lexer::peekNext() const {
    if (current_ + 1 >= source_.length()) return '\0';
    return source_[current_ + 1];
}

char Lexer::advance() {
    char c = source_[current_++];
    column_++;
    return c;
}

bool Lexer::match(char expected) {
    if (isAtEnd()) return false;
    if (source_[current_] != expected) return false;
    current_++;
    column_++;
    return true;
}

void Lexer::addToken(TokenType type) {
    std::string text = source_.substr(start_, current_ - start_);
    tokens_.emplace_back(type, text, line_, startColumn_);
}

void Lexer::addToken(TokenType type, long long value) {
    std::string text = source_.substr(start_, current_ - start_);
    tokens_.emplace_back(type, text, line_, startColumn_, value);
}

void Lexer::addToken(TokenType type, double value) {
    std::string text = source_.substr(start_, current_ - start_);
    tokens_.emplace_back(type, text, line_, startColumn_, value);
}

void Lexer::scanToken() {
    // Handle indentation at start of line
    if (atLineStart_) {
        handleIndentation();
        atLineStart_ = false;
        if (isAtEnd()) return;
        start_ = current_;
        startColumn_ = column_;
    }

    char c = advance();

    switch (c) {
        // Single character tokens (with bracket tracking)
        case '(': bracketDepth_++; addToken(TokenType::LPAREN); break;
        case ')': bracketDepth_--; addToken(TokenType::RPAREN); break;
        case '[': bracketDepth_++; addToken(TokenType::LBRACKET); break;
        case ']': bracketDepth_--; addToken(TokenType::RBRACKET); break;
        case '{': bracketDepth_++; addToken(TokenType::LBRACE); break;
        case '}': bracketDepth_--; addToken(TokenType::RBRACE); break;
        case ':': addToken(TokenType::COLON); break;
        case ',': addToken(TokenType::COMMA); break;
        case '+': addToken(TokenType::PLUS); break;
        case '*': addToken(TokenType::STAR); break;
        case '/': addToken(TokenType::SLASH); break;
        case '@': addToken(TokenType::AT); break;
        case '.': addToken(TokenType::DOT); break;

        // Two character tokens
        case '-':
            if (match('>')) {
                addToken(TokenType::ARROW);
            } else {
                addToken(TokenType::MINUS);
            }
            break;

        case '=':
            if (match('=')) {
                addToken(TokenType::EQUAL_EQUAL);
            } else {
                addToken(TokenType::EQUAL);
            }
            break;

        case '!':
            if (match('=')) {
                addToken(TokenType::BANG_EQUAL);
            } else {
                error("Unexpected character '!'");
            }
            break;

        case '<':
            if (match('=')) {
                addToken(TokenType::LESS_EQUAL);
            } else {
                addToken(TokenType::LESS);
            }
            break;

        case '>':
            if (match('=')) {
                addToken(TokenType::GREATER_EQUAL);
            } else {
                addToken(TokenType::GREATER);
            }
            break;

        // Comments
        case '#':
            skipComment();
            break;

        // Whitespace
        case ' ':
        case '\t':
        case '\r':
            // Ignore whitespace (not at line start)
            break;

        case '\n':
            // Only emit NEWLINE if we have meaningful tokens on this line
            // and we're not inside brackets (for multi-line lists/expressions)
            if (bracketDepth_ == 0 && !tokens_.empty() &&
                tokens_.back().type != TokenType::NEWLINE &&
                tokens_.back().type != TokenType::INDENT) {
                addToken(TokenType::NEWLINE);
            }
            line_++;
            column_ = 1;
            // Only process indentation if not inside brackets
            atLineStart_ = (bracketDepth_ == 0);
            break;

        // String literals
        case '"':
        case '\'':
            current_--;  // Back up to include the quote
            column_--;
            scanString();
            break;

        default:
            if (isDigit(c)) {
                current_--;  // Back up
                column_--;
                scanNumber();
            } else if (isAlpha(c)) {
                current_--;  // Back up
                column_--;
                scanIdentifier();
            } else {
                error("Unexpected character '" + std::string(1, c) + "'");
            }
            break;
    }
}

void Lexer::handleIndentation() {
    int indent = 0;

    while (!isAtEnd() && (peek() == ' ' || peek() == '\t')) {
        if (peek() == ' ') {
            indent++;
        } else {
            indent += 4;  // Tab = 4 spaces
        }
        advance();
    }

    // Skip blank lines and comment-only lines
    if (isAtEnd() || peek() == '\n' || peek() == '#') {
        return;
    }

    int currentIndent = indentStack_.top();

    if (indent > currentIndent) {
        indentStack_.push(indent);
        tokens_.emplace_back(TokenType::INDENT, "", line_, 1);
    } else if (indent < currentIndent) {
        while (!indentStack_.empty() && indentStack_.top() > indent) {
            indentStack_.pop();
            tokens_.emplace_back(TokenType::DEDENT, "", line_, 1);
        }

        if (indentStack_.empty() || indentStack_.top() != indent) {
            error("Inconsistent indentation");
        }
    }
}

void Lexer::skipComment() {
    while (!isAtEnd() && peek() != '\n') {
        advance();
    }
}

void Lexer::scanString() {
    char quote = advance();  // Consume opening quote
    start_ = current_;

    std::string value;
    while (!isAtEnd() && peek() != quote) {
        if (peek() == '\n') {
            error("Unterminated string");
            return;
        }
        if (peek() == '\\') {
            advance();  // Skip backslash
            if (!isAtEnd()) {
                char escaped = advance();
                switch (escaped) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    case '\'': value += '\''; break;
                    default: value += escaped; break;
                }
            }
        } else {
            value += advance();
        }
    }

    if (isAtEnd()) {
        error("Unterminated string");
        return;
    }

    advance();  // Consume closing quote

    tokens_.emplace_back(TokenType::STRING, value, line_, startColumn_);
}

void Lexer::scanNumber() {
    start_ = current_;

    while (isDigit(peek())) {
        advance();
    }

    bool isFloat = false;
    if (peek() == '.' && isDigit(peekNext())) {
        isFloat = true;
        advance();  // Consume '.'
        while (isDigit(peek())) {
            advance();
        }
    }

    // Scientific notation
    if (peek() == 'e' || peek() == 'E') {
        isFloat = true;
        advance();
        if (peek() == '+' || peek() == '-') {
            advance();
        }
        while (isDigit(peek())) {
            advance();
        }
    }

    std::string text = source_.substr(start_, current_ - start_);

    if (isFloat) {
        double value = std::stod(text);
        addToken(TokenType::FLOAT, value);
    } else {
        long long value = std::stoll(text);
        addToken(TokenType::INTEGER, value);
    }
}

void Lexer::scanIdentifier() {
    start_ = current_;

    while (isAlphaNumeric(peek())) {
        advance();
    }

    std::string text = source_.substr(start_, current_ - start_);

    auto it = keywords_.find(text);
    if (it != keywords_.end()) {
        addToken(it->second);
    } else {
        addToken(TokenType::IDENTIFIER);
    }
}

void Lexer::error(const std::string& message) {
    hadError_ = true;
    std::ostringstream oss;
    oss << "[Line " << line_ << ", Column " << column_ << "] Error: " << message;
    errors_.push_back(oss.str());
}

bool Lexer::isDigit(char c) const {
    return c >= '0' && c <= '9';
}

bool Lexer::isAlpha(char c) const {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           c == '_';
}

bool Lexer::isAlphaNumeric(char c) const {
    return isAlpha(c) || isDigit(c);
}

} // namespace duckt
