#ifndef DUCKT_INTERPRETER_HPP
#define DUCKT_INTERPRETER_HPP

#include "ast.hpp"
#include "types.hpp"
#include "environment.hpp"
#include <memory>
#include <stdexcept>

namespace duckt {

// Exception for return statement control flow
class ReturnException : public std::exception {
public:
    Value value;
    explicit ReturnException(Value v) : value(std::move(v)) {}
};

class Interpreter {
public:
    Interpreter();

    void interpret(const Program& program);
    Value evaluate(Expr* expr);
    void execute(Stmt* stmt);

    std::shared_ptr<Environment> globals() const { return globals_; }
    std::shared_ptr<Environment> environment() const { return environment_; }

private:
    std::shared_ptr<Environment> globals_;
    std::shared_ptr<Environment> environment_;

    // Expression evaluation
    Value evalNumber(NumberExpr* expr);
    Value evalString(StringExpr* expr);
    Value evalBool(BoolExpr* expr);
    Value evalNone(NoneExpr* expr);
    Value evalIdentifier(IdentifierExpr* expr);
    Value evalBinary(BinaryExpr* expr);
    Value evalUnary(UnaryExpr* expr);
    Value evalCall(CallExpr* expr);
    Value evalIndex(IndexExpr* expr);
    Value evalMember(MemberExpr* expr);
    Value evalList(ListExpr* expr);
    Value evalTensor(TensorExpr* expr);

    // Statement execution
    void execExprStmt(ExprStmt* stmt);
    void execLet(LetStmt* stmt);
    void execAssign(AssignStmt* stmt);
    void execIf(IfStmt* stmt);
    void execWhile(WhileStmt* stmt);
    void execFor(ForStmt* stmt);
    void execFn(FnStmt* stmt);
    void execReturn(ReturnStmt* stmt);
    void execModel(ModelStmt* stmt);
    void execAgent(AgentStmt* stmt);
    void execTrain(TrainStmt* stmt);
    void execLayer(LayerStmt* stmt);
    void execState(StateStmt* stmt);
    void execTool(ToolStmt* stmt);

    // Helpers
    Value call(const Value& callee, const std::vector<Value>& args);
    void executeBlock(const std::vector<StmtPtr>& statements, std::shared_ptr<Environment> env);
    bool isTensorLike(const std::vector<Value>& elements);
    Tensor listToTensor(const std::vector<Value>& elements);
};

} // namespace duckt

#endif // DUCKT_INTERPRETER_HPP
