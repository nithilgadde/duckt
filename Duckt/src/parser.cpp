#include "parser.hpp"
#include <sstream>

namespace duckt {

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

Program Parser::parse() {
    Program program;

    skipNewlines();

    while (!isAtEnd()) {
        try {
            auto stmt = parseStatement();
            if (stmt) {
                program.statements.push_back(std::move(stmt));
            }
        } catch (const ParseError&) {
            synchronize();
        }
        skipNewlines();
    }

    return program;
}

// ============ Token Navigation ============

const Token& Parser::peek() const {
    return tokens_[current_];
}

const Token& Parser::previous() const {
    return tokens_[current_ - 1];
}

const Token& Parser::advance() {
    if (!isAtEnd()) current_++;
    return previous();
}

bool Parser::isAtEnd() const {
    return peek().type == TokenType::END_OF_FILE;
}

bool Parser::check(TokenType type) const {
    if (isAtEnd()) return false;
    return peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::match(std::initializer_list<TokenType> types) {
    for (TokenType type : types) {
        if (check(type)) {
            advance();
            return true;
        }
    }
    return false;
}

Token Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) return advance();
    error(peek(), message);
    throw ParseError(message);
}

void Parser::skipNewlines() {
    while (match(TokenType::NEWLINE)) {
        // Skip
    }
}

// ============ Error Handling ============

void Parser::error(const Token& token, const std::string& message) {
    hadError_ = true;
    std::ostringstream oss;
    oss << "[Line " << token.line << "] Error at '"
        << token.lexeme << "': " << message;
    errors_.push_back(oss.str());
}

void Parser::synchronize() {
    advance();

    while (!isAtEnd()) {
        if (previous().type == TokenType::NEWLINE) return;

        switch (peek().type) {
            case TokenType::LET:
            case TokenType::FN:
            case TokenType::IF:
            case TokenType::WHILE:
            case TokenType::FOR:
            case TokenType::RETURN:
            case TokenType::MODEL:
            case TokenType::AGENT:
            case TokenType::LAYER:
            case TokenType::TRAIN:
                return;
            default:
                break;
        }

        advance();
    }
}

// ============ Statement Parsing ============

StmtPtr Parser::parseStatement() {
    if (match(TokenType::LET)) return parseLetStatement();
    if (match(TokenType::FN)) return parseFnStatement();
    if (match(TokenType::IF)) return parseIfStatement();
    if (match(TokenType::WHILE)) return parseWhileStatement();
    if (match(TokenType::FOR)) return parseForStatement();
    if (match(TokenType::RETURN)) return parseReturnStatement();
    if (match(TokenType::MODEL)) return parseModelStatement();
    if (match(TokenType::AGENT)) return parseAgentStatement();
    if (match(TokenType::TRAIN)) return parseTrainStatement();
    if (match(TokenType::LAYER)) return parseLayerStatement();
    if (match(TokenType::STATE)) return parseStateStatement();
    if (match(TokenType::TOOL)) return parseToolStatement();

    return parseExpressionStatement();
}

StmtPtr Parser::parseLetStatement() {
    int line = previous().line;
    Token name = consume(TokenType::IDENTIFIER, "Expected variable name after 'let'");
    consume(TokenType::EQUAL, "Expected '=' after variable name");
    ExprPtr value = parseExpression();

    if (!isAtEnd() && !check(TokenType::DEDENT)) {
        match(TokenType::NEWLINE);
    }

    return std::make_unique<LetStmt>(name.lexeme, std::move(value), line);
}

StmtPtr Parser::parseFnStatement() {
    int line = previous().line;
    Token name = consume(TokenType::IDENTIFIER, "Expected function name");
    consume(TokenType::LPAREN, "Expected '(' after function name");

    std::vector<std::string> params = parseParameters();

    consume(TokenType::RPAREN, "Expected ')' after parameters");
    consume(TokenType::COLON, "Expected ':' after function declaration");

    std::vector<StmtPtr> body = parseBlock();

    return std::make_unique<FnStmt>(name.lexeme, std::move(params), std::move(body), line);
}

StmtPtr Parser::parseIfStatement() {
    int line = previous().line;
    ExprPtr condition = parseExpression();
    consume(TokenType::COLON, "Expected ':' after if condition");

    std::vector<StmtPtr> thenBranch = parseBlock();
    std::vector<StmtPtr> elseBranch;

    skipNewlines();

    if (match(TokenType::ELSE)) {
        consume(TokenType::COLON, "Expected ':' after 'else'");
        elseBranch = parseBlock();
    }

    return std::make_unique<IfStmt>(std::move(condition), std::move(thenBranch),
                                     std::move(elseBranch), line);
}

StmtPtr Parser::parseWhileStatement() {
    int line = previous().line;
    ExprPtr condition = parseExpression();
    consume(TokenType::COLON, "Expected ':' after while condition");

    std::vector<StmtPtr> body = parseBlock();

    return std::make_unique<WhileStmt>(std::move(condition), std::move(body), line);
}

StmtPtr Parser::parseForStatement() {
    int line = previous().line;
    Token var = consume(TokenType::IDENTIFIER, "Expected variable name in for loop");
    consume(TokenType::IN, "Expected 'in' after variable in for loop");
    ExprPtr iterable = parseExpression();
    consume(TokenType::COLON, "Expected ':' after for loop declaration");

    std::vector<StmtPtr> body = parseBlock();

    return std::make_unique<ForStmt>(var.lexeme, std::move(iterable), std::move(body), line);
}

StmtPtr Parser::parseReturnStatement() {
    int line = previous().line;
    ExprPtr value = nullptr;

    if (!check(TokenType::NEWLINE) && !check(TokenType::DEDENT) && !isAtEnd()) {
        value = parseExpression();
    }

    if (!isAtEnd() && !check(TokenType::DEDENT)) {
        match(TokenType::NEWLINE);
    }

    return std::make_unique<ReturnStmt>(std::move(value), line);
}

StmtPtr Parser::parseModelStatement() {
    int line = previous().line;
    Token name = consume(TokenType::IDENTIFIER, "Expected model name");
    consume(TokenType::COLON, "Expected ':' after model name");

    std::vector<StmtPtr> body = parseBlock();

    return std::make_unique<ModelStmt>(name.lexeme, std::move(body), line);
}

StmtPtr Parser::parseAgentStatement() {
    int line = previous().line;
    Token name = consume(TokenType::IDENTIFIER, "Expected agent name");
    consume(TokenType::COLON, "Expected ':' after agent name");

    std::vector<StmtPtr> body = parseBlock();

    return std::make_unique<AgentStmt>(name.lexeme, std::move(body), line);
}

StmtPtr Parser::parseTrainStatement() {
    int line = previous().line;
    Token model = consume(TokenType::IDENTIFIER, "Expected model name after 'train'");

    ExprPtr data = nullptr;
    ExprPtr epochs = nullptr;
    ExprPtr lr = nullptr;
    ExprPtr loss = nullptr;

    // Parse optional clauses: on data, for N epochs, with lr=X
    while (!check(TokenType::NEWLINE) && !check(TokenType::DEDENT) && !isAtEnd()) {
        if (match(TokenType::ON)) {
            data = parseExpression();
        } else if (match(TokenType::FOR)) {
            epochs = parseExpression();
            match(TokenType::EPOCHS);
        } else if (match(TokenType::WITH)) {
            if (match(TokenType::LR)) {
                consume(TokenType::EQUAL, "Expected '=' after 'lr'");
                lr = parseExpression();
            } else {
                // Generic expression
                lr = parseExpression();
            }
        } else {
            break;
        }
    }

    if (!isAtEnd() && !check(TokenType::DEDENT)) {
        match(TokenType::NEWLINE);
    }

    return std::make_unique<TrainStmt>(model.lexeme, std::move(data),
                                        std::move(epochs), std::move(lr), std::move(loss), line);
}

StmtPtr Parser::parseLayerStatement() {
    int line = previous().line;
    Token name = consume(TokenType::IDENTIFIER, "Expected layer name");
    consume(TokenType::EQUAL, "Expected '=' after layer name");

    Token layerType = consume(TokenType::IDENTIFIER, "Expected layer type");
    consume(TokenType::LPAREN, "Expected '(' after layer type");

    std::vector<ExprPtr> args = parseArguments();

    consume(TokenType::RPAREN, "Expected ')' after layer arguments");

    if (!isAtEnd() && !check(TokenType::DEDENT)) {
        match(TokenType::NEWLINE);
    }

    return std::make_unique<LayerStmt>(name.lexeme, layerType.lexeme, std::move(args), line);
}

StmtPtr Parser::parseStateStatement() {
    int line = previous().line;
    Token name = consume(TokenType::IDENTIFIER, "Expected state name");
    consume(TokenType::EQUAL, "Expected '=' after state name");
    ExprPtr initial = parseExpression();

    if (!isAtEnd() && !check(TokenType::DEDENT)) {
        match(TokenType::NEWLINE);
    }

    return std::make_unique<StateStmt>(name.lexeme, std::move(initial), line);
}

StmtPtr Parser::parseToolStatement() {
    int line = previous().line;
    Token name = consume(TokenType::IDENTIFIER, "Expected tool name");
    consume(TokenType::LPAREN, "Expected '(' after tool name");

    std::vector<std::string> params = parseParameters();

    consume(TokenType::RPAREN, "Expected ')' after parameters");
    consume(TokenType::COLON, "Expected ':' after tool declaration");

    std::vector<StmtPtr> body = parseBlock();

    return std::make_unique<ToolStmt>(name.lexeme, std::move(params), std::move(body), line);
}

StmtPtr Parser::parseExpressionStatement() {
    int line = peek().line;
    ExprPtr expr = parseExpression();

    // Check for assignment
    if (match(TokenType::EQUAL)) {
        ExprPtr value = parseExpression();
        if (!isAtEnd() && !check(TokenType::DEDENT)) {
            match(TokenType::NEWLINE);
        }
        return std::make_unique<AssignStmt>(std::move(expr), std::move(value), line);
    }

    if (!isAtEnd() && !check(TokenType::DEDENT)) {
        match(TokenType::NEWLINE);
    }

    return std::make_unique<ExprStmt>(std::move(expr), line);
}

std::vector<StmtPtr> Parser::parseBlock() {
    std::vector<StmtPtr> statements;

    // Expect NEWLINE then INDENT
    if (!match(TokenType::NEWLINE)) {
        // Single-line block not supported
        error(peek(), "Expected newline before indented block");
        return statements;
    }

    skipNewlines();

    if (!match(TokenType::INDENT)) {
        error(peek(), "Expected indented block");
        return statements;
    }

    while (!check(TokenType::DEDENT) && !isAtEnd()) {
        skipNewlines();
        if (check(TokenType::DEDENT) || isAtEnd()) break;

        auto stmt = parseStatement();
        if (stmt) {
            statements.push_back(std::move(stmt));
        }
        skipNewlines();
    }

    if (!match(TokenType::DEDENT) && !isAtEnd()) {
        // It's okay if we're at EOF
    }

    return statements;
}

// ============ Expression Parsing ============

ExprPtr Parser::parseExpression() {
    return parseOr();
}

ExprPtr Parser::parseOr() {
    ExprPtr left = parseAnd();

    while (match(TokenType::OR)) {
        int line = previous().line;
        ExprPtr right = parseAnd();
        left = std::make_unique<BinaryExpr>(BinaryOp::OR, std::move(left), std::move(right), line);
    }

    return left;
}

ExprPtr Parser::parseAnd() {
    ExprPtr left = parseEquality();

    while (match(TokenType::AND)) {
        int line = previous().line;
        ExprPtr right = parseEquality();
        left = std::make_unique<BinaryExpr>(BinaryOp::AND, std::move(left), std::move(right), line);
    }

    return left;
}

ExprPtr Parser::parseEquality() {
    ExprPtr left = parseComparison();

    while (match({TokenType::EQUAL_EQUAL, TokenType::BANG_EQUAL})) {
        int line = previous().line;
        BinaryOp op = previous().type == TokenType::EQUAL_EQUAL ? BinaryOp::EQ : BinaryOp::NE;
        ExprPtr right = parseComparison();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right), line);
    }

    return left;
}

ExprPtr Parser::parseComparison() {
    ExprPtr left = parseTerm();

    while (match({TokenType::LESS, TokenType::GREATER,
                  TokenType::LESS_EQUAL, TokenType::GREATER_EQUAL})) {
        int line = previous().line;
        BinaryOp op;
        switch (previous().type) {
            case TokenType::LESS: op = BinaryOp::LT; break;
            case TokenType::GREATER: op = BinaryOp::GT; break;
            case TokenType::LESS_EQUAL: op = BinaryOp::LE; break;
            case TokenType::GREATER_EQUAL: op = BinaryOp::GE; break;
            default: op = BinaryOp::LT; break;  // Unreachable
        }
        ExprPtr right = parseTerm();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right), line);
    }

    return left;
}

ExprPtr Parser::parseTerm() {
    ExprPtr left = parseFactor();

    while (match({TokenType::PLUS, TokenType::MINUS})) {
        int line = previous().line;
        BinaryOp op = previous().type == TokenType::PLUS ? BinaryOp::ADD : BinaryOp::SUB;
        ExprPtr right = parseFactor();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right), line);
    }

    return left;
}

ExprPtr Parser::parseFactor() {
    ExprPtr left = parseMatmul();

    while (match({TokenType::STAR, TokenType::SLASH})) {
        int line = previous().line;
        BinaryOp op = previous().type == TokenType::STAR ? BinaryOp::MUL : BinaryOp::DIV;
        ExprPtr right = parseMatmul();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right), line);
    }

    return left;
}

ExprPtr Parser::parseMatmul() {
    ExprPtr left = parseUnary();

    while (match(TokenType::AT)) {
        int line = previous().line;
        ExprPtr right = parseUnary();
        left = std::make_unique<BinaryExpr>(BinaryOp::MATMUL, std::move(left), std::move(right), line);
    }

    return left;
}

ExprPtr Parser::parseUnary() {
    if (match(TokenType::MINUS)) {
        int line = previous().line;
        ExprPtr operand = parseUnary();
        return std::make_unique<UnaryExpr>(UnaryOp::NEG, std::move(operand), line);
    }

    if (match(TokenType::NOT)) {
        int line = previous().line;
        ExprPtr operand = parseUnary();
        return std::make_unique<UnaryExpr>(UnaryOp::NOT, std::move(operand), line);
    }

    return parseCall();
}

ExprPtr Parser::parseCall() {
    ExprPtr expr = parsePrimary();

    while (true) {
        if (match(TokenType::LPAREN)) {
            int line = previous().line;
            std::vector<ExprPtr> args = parseArguments();
            consume(TokenType::RPAREN, "Expected ')' after arguments");
            expr = std::make_unique<CallExpr>(std::move(expr), std::move(args), line);
        } else if (match(TokenType::LBRACKET)) {
            int line = previous().line;
            ExprPtr index = parseExpression();
            consume(TokenType::RBRACKET, "Expected ']' after index");
            expr = std::make_unique<IndexExpr>(std::move(expr), std::move(index), line);
        } else if (match(TokenType::DOT)) {
            int line = previous().line;
            Token member = consume(TokenType::IDENTIFIER, "Expected member name after '.'");
            expr = std::make_unique<MemberExpr>(std::move(expr), member.lexeme, line);
        } else {
            break;
        }
    }

    return expr;
}

ExprPtr Parser::parsePrimary() {
    int line = peek().line;

    if (match(TokenType::TRUE)) {
        return std::make_unique<BoolExpr>(true, line);
    }

    if (match(TokenType::FALSE)) {
        return std::make_unique<BoolExpr>(false, line);
    }

    if (match(TokenType::NONE)) {
        return std::make_unique<NoneExpr>(line);
    }

    if (match(TokenType::INTEGER)) {
        long long val = std::get<long long>(previous().literal);
        return std::make_unique<NumberExpr>(static_cast<double>(val), true, line);
    }

    if (match(TokenType::FLOAT)) {
        double val = std::get<double>(previous().literal);
        return std::make_unique<NumberExpr>(val, false, line);
    }

    if (match(TokenType::STRING)) {
        return std::make_unique<StringExpr>(previous().lexeme, line);
    }

    if (match(TokenType::IDENTIFIER)) {
        return std::make_unique<IdentifierExpr>(previous().lexeme, line);
    }

    if (match(TokenType::LBRACKET)) {
        return parseListOrTensor();
    }

    if (match(TokenType::LPAREN)) {
        ExprPtr expr = parseExpression();
        consume(TokenType::RPAREN, "Expected ')' after expression");
        return expr;
    }

    error(peek(), "Expected expression");
    throw ParseError("Expected expression");
}

ExprPtr Parser::parseListOrTensor() {
    int line = previous().line;
    std::vector<ExprPtr> elements;

    if (!check(TokenType::RBRACKET)) {
        do {
            elements.push_back(parseExpression());
        } while (match(TokenType::COMMA));
    }

    consume(TokenType::RBRACKET, "Expected ']' after list elements");

    // Check if it's a tensor (nested lists of numbers)
    // For simplicity, treat all bracket expressions as ListExpr
    // The interpreter can decide if it's tensor-like
    return std::make_unique<ListExpr>(std::move(elements), line);
}

// ============ Helpers ============

std::vector<std::string> Parser::parseParameters() {
    std::vector<std::string> params;

    if (!check(TokenType::RPAREN)) {
        do {
            Token param = consume(TokenType::IDENTIFIER, "Expected parameter name");
            params.push_back(param.lexeme);
        } while (match(TokenType::COMMA));
    }

    return params;
}

std::vector<ExprPtr> Parser::parseArguments() {
    std::vector<ExprPtr> args;

    if (!check(TokenType::RPAREN)) {
        do {
            args.push_back(parseExpression());
        } while (match(TokenType::COMMA));
    }

    return args;
}

} // namespace duckt
