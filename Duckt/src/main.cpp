#include "lexer.hpp"
#include "parser.hpp"
#include "interpreter.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace duckt;

void printUsage(const char* programName) {
    std::cout << "Duckt - A language for ML networks and AI agents\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << "              Start REPL\n";
    std::cout << "  " << programName << " <file.dt>    Run a Duckt program\n";
    std::cout << "  " << programName << " --help       Show this help\n";
    std::cout << "  " << programName << " --version    Show version\n";
}

void printVersion() {
    std::cout << "Duckt 0.1.0\n";
    std::cout << "A Python-like language for ML networks and AI agents\n";
}

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void runSource(const std::string& source, Interpreter& interpreter) {
    // Lexer
    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    if (lexer.hasError()) {
        for (const auto& err : lexer.errors()) {
            std::cerr << err << std::endl;
        }
        return;
    }

    // Parser
    Parser parser(std::move(tokens));
    Program program = parser.parse();

    if (parser.hasError()) {
        for (const auto& err : parser.errors()) {
            std::cerr << err << std::endl;
        }
        return;
    }

    // Interpreter
    try {
        interpreter.interpret(program);
    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
    }
}

void runFile(const std::string& path) {
    try {
        std::string source = readFile(path);
        Interpreter interpreter;
        runSource(source, interpreter);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void runRepl() {
    std::cout << "Duckt 0.1.0 - Interactive Mode\n";
    std::cout << "Type 'exit' or Ctrl+D to quit, 'help' for commands\n\n";

    Interpreter interpreter;
    std::string line;
    std::string buffer;
    int openBrackets = 0;
    int openParens = 0;
    bool inBlock = false;

    auto countDelimiters = [](const std::string& s, int& brackets, int& parens, bool& block) {
        for (size_t i = 0; i < s.length(); i++) {
            char c = s[i];
            if (c == '[') brackets++;
            else if (c == ']') brackets--;
            else if (c == '(') parens++;
            else if (c == ')') parens--;
            else if (c == ':' && i == s.length() - 1) block = true;
        }
    };

    while (true) {
        if (buffer.empty()) {
            std::cout << ">>> ";
        } else {
            std::cout << "... ";
        }

        if (!std::getline(std::cin, line)) {
            std::cout << std::endl;
            break;
        }

        // Handle special commands
        if (buffer.empty()) {
            if (line == "exit" || line == "quit") {
                break;
            }
            if (line == "help") {
                std::cout << "\nCommands:\n";
                std::cout << "  exit, quit   Exit the REPL\n";
                std::cout << "  help         Show this help\n";
                std::cout << "\nExamples:\n";
                std::cout << "  let x = 42\n";
                std::cout << "  print(x + 8)\n";
                std::cout << "  let t = [[1, 2], [3, 4]]\n";
                std::cout << "  print(t @ t.T)\n\n";
                continue;
            }
        }

        // Handle multi-line input
        bool wasInBlock = inBlock;
        countDelimiters(line, openBrackets, openParens, inBlock);

        buffer += line + "\n";

        // Check if input is complete
        bool isComplete = openBrackets <= 0 && openParens <= 0;

        // For blocks (after :), continue until we get an empty line
        if (wasInBlock && !line.empty() && line[0] != ' ' && line[0] != '\t') {
            // End of block - previous non-indented line
        } else if (inBlock && line.empty()) {
            inBlock = false;
        } else if (inBlock) {
            continue;  // Wait for more input
        }

        if (!isComplete) {
            continue;  // Wait for closing brackets/parens
        }

        // Execute the buffered input
        runSource(buffer, interpreter);

        // Reset for next input
        buffer.clear();
        openBrackets = 0;
        openParens = 0;
        inBlock = false;
    }
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        runRepl();
        return 0;
    }

    std::string arg = argv[1];

    if (arg == "--help" || arg == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (arg == "--version" || arg == "-v") {
        printVersion();
        return 0;
    }

    // Run file
    runFile(arg);
    return 0;
}
