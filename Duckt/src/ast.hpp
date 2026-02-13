#ifndef DUCKT_AST_HPP
#define DUCKT_AST_HPP

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "token.hpp"

namespace duckt {

// Forward declarations
struct Expr;
struct Stmt;

using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

// Binary operators
enum class BinaryOp {
    ADD, SUB, MUL, DIV, MATMUL,
    EQ, NE, LT, GT, LE, GE,
    AND, OR
};

// Unary operators
enum class UnaryOp {
    NEG, NOT
};

// ============ Expressions ============

struct Expr {
    int line;
    virtual ~Expr() = default;
    Expr(int ln = 0) : line(ln) {}
};

struct NumberExpr : Expr {
    double value;
    bool isInteger;

    NumberExpr(double val, bool isInt = false, int ln = 0)
        : Expr(ln), value(val), isInteger(isInt) {}
};

struct StringExpr : Expr {
    std::string value;

    explicit StringExpr(std::string val, int ln = 0)
        : Expr(ln), value(std::move(val)) {}
};

struct BoolExpr : Expr {
    bool value;

    explicit BoolExpr(bool val, int ln = 0)
        : Expr(ln), value(val) {}
};

struct NoneExpr : Expr {
    explicit NoneExpr(int ln = 0) : Expr(ln) {}
};

struct IdentifierExpr : Expr {
    std::string name;

    explicit IdentifierExpr(std::string n, int ln = 0)
        : Expr(ln), name(std::move(n)) {}
};

struct BinaryExpr : Expr {
    BinaryOp op;
    ExprPtr left;
    ExprPtr right;

    BinaryExpr(BinaryOp o, ExprPtr l, ExprPtr r, int ln = 0)
        : Expr(ln), op(o), left(std::move(l)), right(std::move(r)) {}
};

struct UnaryExpr : Expr {
    UnaryOp op;
    ExprPtr operand;

    UnaryExpr(UnaryOp o, ExprPtr e, int ln = 0)
        : Expr(ln), op(o), operand(std::move(e)) {}
};

struct CallExpr : Expr {
    ExprPtr callee;
    std::vector<ExprPtr> args;

    CallExpr(ExprPtr c, std::vector<ExprPtr> a, int ln = 0)
        : Expr(ln), callee(std::move(c)), args(std::move(a)) {}
};

struct IndexExpr : Expr {
    ExprPtr object;
    ExprPtr index;

    IndexExpr(ExprPtr obj, ExprPtr idx, int ln = 0)
        : Expr(ln), object(std::move(obj)), index(std::move(idx)) {}
};

struct MemberExpr : Expr {
    ExprPtr object;
    std::string member;

    MemberExpr(ExprPtr obj, std::string mem, int ln = 0)
        : Expr(ln), object(std::move(obj)), member(std::move(mem)) {}
};

struct ListExpr : Expr {
    std::vector<ExprPtr> elements;

    explicit ListExpr(std::vector<ExprPtr> elems, int ln = 0)
        : Expr(ln), elements(std::move(elems)) {}
};

struct TensorExpr : Expr {
    std::vector<ExprPtr> elements;

    explicit TensorExpr(std::vector<ExprPtr> elems, int ln = 0)
        : Expr(ln), elements(std::move(elems)) {}
};

struct LambdaExpr : Expr {
    std::vector<std::string> params;
    ExprPtr body;

    LambdaExpr(std::vector<std::string> p, ExprPtr b, int ln = 0)
        : Expr(ln), params(std::move(p)), body(std::move(b)) {}
};

// ============ Statements ============

struct Stmt {
    int line;
    virtual ~Stmt() = default;
    Stmt(int ln = 0) : line(ln) {}
};

struct ExprStmt : Stmt {
    ExprPtr expr;

    explicit ExprStmt(ExprPtr e, int ln = 0)
        : Stmt(ln), expr(std::move(e)) {}
};

struct LetStmt : Stmt {
    std::string name;
    ExprPtr value;

    LetStmt(std::string n, ExprPtr v, int ln = 0)
        : Stmt(ln), name(std::move(n)), value(std::move(v)) {}
};

struct AssignStmt : Stmt {
    ExprPtr target;  // Can be IdentifierExpr, IndexExpr, or MemberExpr
    ExprPtr value;

    AssignStmt(ExprPtr t, ExprPtr v, int ln = 0)
        : Stmt(ln), target(std::move(t)), value(std::move(v)) {}
};

struct BlockStmt : Stmt {
    std::vector<StmtPtr> statements;

    explicit BlockStmt(std::vector<StmtPtr> stmts, int ln = 0)
        : Stmt(ln), statements(std::move(stmts)) {}
};

struct IfStmt : Stmt {
    ExprPtr condition;
    std::vector<StmtPtr> thenBranch;
    std::vector<StmtPtr> elseBranch;

    IfStmt(ExprPtr cond, std::vector<StmtPtr> then_, std::vector<StmtPtr> else_, int ln = 0)
        : Stmt(ln), condition(std::move(cond)),
          thenBranch(std::move(then_)), elseBranch(std::move(else_)) {}
};

struct WhileStmt : Stmt {
    ExprPtr condition;
    std::vector<StmtPtr> body;

    WhileStmt(ExprPtr cond, std::vector<StmtPtr> b, int ln = 0)
        : Stmt(ln), condition(std::move(cond)), body(std::move(b)) {}
};

struct ForStmt : Stmt {
    std::string variable;
    ExprPtr iterable;
    std::vector<StmtPtr> body;

    ForStmt(std::string var, ExprPtr iter, std::vector<StmtPtr> b, int ln = 0)
        : Stmt(ln), variable(std::move(var)), iterable(std::move(iter)), body(std::move(b)) {}
};

struct FnStmt : Stmt {
    std::string name;
    std::vector<std::string> params;
    std::vector<StmtPtr> body;

    FnStmt(std::string n, std::vector<std::string> p, std::vector<StmtPtr> b, int ln = 0)
        : Stmt(ln), name(std::move(n)), params(std::move(p)), body(std::move(b)) {}
};

struct ReturnStmt : Stmt {
    ExprPtr value;  // Can be nullptr

    explicit ReturnStmt(ExprPtr v = nullptr, int ln = 0)
        : Stmt(ln), value(std::move(v)) {}
};

// ============ ML-Specific Statements ============

struct LayerStmt : Stmt {
    std::string name;
    std::string layerType;
    std::vector<ExprPtr> args;

    LayerStmt(std::string n, std::string type, std::vector<ExprPtr> a, int ln = 0)
        : Stmt(ln), name(std::move(n)), layerType(std::move(type)), args(std::move(a)) {}
};

struct ModelStmt : Stmt {
    std::string name;
    std::vector<StmtPtr> body;

    ModelStmt(std::string n, std::vector<StmtPtr> b, int ln = 0)
        : Stmt(ln), name(std::move(n)), body(std::move(b)) {}
};

struct TrainStmt : Stmt {
    std::string modelName;
    ExprPtr data;
    ExprPtr epochs;
    ExprPtr learningRate;
    ExprPtr loss;

    TrainStmt(std::string model, ExprPtr d, ExprPtr e, ExprPtr lr, ExprPtr l = nullptr, int ln = 0)
        : Stmt(ln), modelName(std::move(model)), data(std::move(d)),
          epochs(std::move(e)), learningRate(std::move(lr)), loss(std::move(l)) {}
};

// ============ Agent-Specific Statements ============

struct StateStmt : Stmt {
    std::string name;
    ExprPtr initial;

    StateStmt(std::string n, ExprPtr init, int ln = 0)
        : Stmt(ln), name(std::move(n)), initial(std::move(init)) {}
};

struct ToolStmt : Stmt {
    std::string name;
    std::vector<std::string> params;
    std::vector<StmtPtr> body;

    ToolStmt(std::string n, std::vector<std::string> p, std::vector<StmtPtr> b, int ln = 0)
        : Stmt(ln), name(std::move(n)), params(std::move(p)), body(std::move(b)) {}
};

struct AgentStmt : Stmt {
    std::string name;
    std::vector<StmtPtr> body;  // Contains StateStmt, ToolStmt, FnStmt

    AgentStmt(std::string n, std::vector<StmtPtr> b, int ln = 0)
        : Stmt(ln), name(std::move(n)), body(std::move(b)) {}
};

// ============ Program ============

struct Program {
    std::vector<StmtPtr> statements;
};

} // namespace duckt

#endif // DUCKT_AST_HPP
