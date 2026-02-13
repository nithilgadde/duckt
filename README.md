# Duckt

![duckt](https://github.com/user-attachments/assets/6676efe3-3792-4cdb-963f-0042db8fbe47)


a dumb Python-like programming language i made for ML networks and AI agents, implemented in C++17.

Duckt combines the simplicity of Python's syntax with specialized constructs for machine learning and AI agent development, all while delivering the performance of a C++ implementation.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Language Guide](#language-guide)
  - [Variables and Types](#variables-and-types)
  - [Operators](#operators)
  - [Control Flow](#control-flow)
  - [Functions](#functions)
  - [Tensors](#tensors)
  - [Neural Networks](#neural-networks)
  - [AI Agents](#ai-agents)
- [Built-in Functions Reference](#built-in-functions-reference)
- [Layer Types Reference](#layer-types-reference)
- [Examples](#examples)
- [Building from Source](#building-from-source)
- [Project Structure](#project-structure)
- [Embedding Duckt](#embedding-duckt)
- [License](#license)

## Features

- **Python-like Syntax** - Indentation-based blocks, minimal punctuation, familiar keywords
- **First-class Tensors** - Native multi-dimensional arrays with broadcasting and linear algebra
- **Neural Network Primitives** - Built-in layers, models, and training constructs
- **AI Agent System** - State management, tools, and method definitions for building agents
- **Real AI Integration** - Built-in functions for HTTP requests, JSON parsing
- **Interactive REPL** - Explore and test code interactively
- **Fast Execution** - Tree-walking interpreter implemented in modern C++17

## Quick Start

### Build

```bash
git clone <repository>
cd Duckt
make
```

### Run the REPL

```bash
./bin/duckt
```

```
Duckt 0.1.0 - Interactive Mode
Type 'exit' or Ctrl+D to quit, 'help' for commands

>>> let x = 42
>>> print(x * 2)
84
>>> let t = [[1, 2], [3, 4]]
>>> print(t @ t.T)
Tensor(shape=[2, 2], data=[5, 11, 11, 25])
```

### Run a Program

```bash
./bin/duckt examples/hello.dt
```

## Language Guide

### Variables and Types

Duckt is dynamically typed. Variables are declared with `let`:

```python
# Numbers (integers and floats)
let age = 25
let pi = 3.14159
let scientific = 1.5e-10

# Strings
let name = "Duckt"
let greeting = 'Hello, World!'

# Booleans
let is_ready = true
let is_done = false

# None value
let empty = none

# Lists
let numbers = [1, 2, 3, 4, 5]
let mixed = [1, "two", 3.0, true]

# Tensors (numeric lists become tensors automatically)
let vector = [1, 2, 3]           # 1D tensor, shape [3]
let matrix = [[1, 2], [3, 4]]    # 2D tensor, shape [2, 2]
```

### Operators

#### Arithmetic
| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition / Concatenation | `3 + 4` → `7`, `"a" + "b"` → `"ab"` |
| `-` | Subtraction | `10 - 3` → `7` |
| `*` | Multiplication | `4 * 5` → `20` |
| `/` | Division | `15 / 4` → `3.75` |
| `@` | Matrix multiplication | `A @ B` |

#### Comparison
| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `x == 5` |
| `!=` | Not equal | `x != 0` |
| `<` | Less than | `x < 10` |
| `>` | Greater than | `x > 0` |
| `<=` | Less or equal | `x <= 100` |
| `>=` | Greater or equal | `x >= 1` |

#### Logical
| Operator | Description | Example |
|----------|-------------|---------|
| `and` | Logical AND | `x > 0 and x < 10` |
| `or` | Logical OR | `x < 0 or x > 100` |
| `not` | Logical NOT | `not is_done` |

### Control Flow

#### If/Else Statements

```python
let score = 85

if score >= 90:
    print("Grade: A")
else:
    if score >= 80:
        print("Grade: B")
    else:
        print("Grade: C or below")
```

#### For Loops

```python
# Iterate over a range
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# Iterate over a list
let fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Iterate over a tensor
let data = [10, 20, 30]
for value in data:
    print(value)
```

#### While Loops

```python
let count = 0
while count < 5:
    print("Count:", count)
    count = count + 1
```

### Functions

Functions are defined with `fn`:

```python
# Basic function
fn greet(name):
    return "Hello, " + name + "!"

print(greet("Alice"))  # Hello, Alice!

# Function with multiple parameters
fn add(a, b):
    return a + b

# Recursive function
fn factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120

# Functions are first-class values
let my_func = add
print(my_func(3, 4))  # 7
```

### Tensors

Tensors are the core data structure for numerical computation:

```python
# Creation
let v = [1, 2, 3, 4, 5]              # From list
let z = zeros([3, 3])                 # 3x3 zeros
let o = ones([2, 4])                  # 2x4 ones
let r = rand([10, 10])                # 10x10 random [0,1)
let seq = arange(0, 10, 2)            # [0, 2, 4, 6, 8]

# Properties
print(shape(v))                       # [5]
print(len(v))                         # 5

# Element-wise operations
let a = [1, 2, 3]
let b = [4, 5, 6]
print(a + b)                          # [5, 7, 9]
print(a * b)                          # [4, 10, 18]
print(a * 2)                          # [2, 4, 6]

# Matrix operations
let m1 = [[1, 2], [3, 4]]
let m2 = [[5, 6], [7, 8]]
print(m1 @ m2)                        # Matrix multiplication
print(m1.T)                           # Transpose

# Reductions
let t = [1, 2, 3, 4, 5]
print(sum(t))                         # 15
print(mean(t))                        # 3.0
print(max(t))                         # 5
print(min(t))                         # 1

# Activation functions
let x = [-2, -1, 0, 1, 2]
print(relu(x))                        # [0, 0, 0, 1, 2]
print(sigmoid(x))                     # [0.12, 0.27, 0.5, 0.73, 0.88]
print(softmax(x))                     # Normalized probabilities
```

### Neural Networks

Duckt provides first-class support for defining neural network models:

```python
# Define a model
model MLP:
    layer fc1 = Linear(784, 256)
    layer relu1 = ReLU()
    layer fc2 = Linear(256, 64)
    layer relu2 = ReLU()
    layer fc3 = Linear(64, 10)
    layer softmax = Softmax()

    fn forward(x):
        x = fc1(x)
        x = relu(x)
        x = fc2(x)
        x = relu(x)
        x = fc3(x)
        return softmax(x)

# Use the model (models are defined directly)
let net = MLP

# Create input tensor
let input = rand([784])

# Forward pass
let output = net.forward(input)
print("Output shape:", shape(output))

# Individual layers can also be used standalone
let linear = Linear(10, 5)
let x = rand([10])
let y = linear(x)
print("Layer output:", y)
```

#### Training (Syntax demonstration)

```python
# Training syntax
train MLP on data for 10 epochs with lr=0.001

# Or use a custom training loop
fn train_model(model_name, data, num_epochs, learning_rate):
    print("Training", model_name)
    for epoch in range(num_epochs):
        # Forward pass, compute loss, backward pass, update weights
        let loss = 1.0 / (epoch + 1)  # Simulated decreasing loss
        print("Epoch", epoch + 1, "- Loss:", loss)
    print("Training complete!")
```

### AI Agents

Agents encapsulate state, tools, and behavior:

```python
agent Researcher:
    # State persists across method calls
    state notes = []
    state query_count = 0

    # Tools are special functions the agent can use
    tool search(query):
        print("Searching for:", query)
        return "Results for: " + query

    tool summarize(text):
        return "Summary: " + text

    # Methods define agent behavior
    fn process(question):
        query_count = query_count + 1
        let results = search(question)
        let summary = summarize(results)
        notes = notes + [summary]
        return summary

    fn get_notes():
        return notes

# Use the agent (agents are defined directly, no instantiation needed)
let researcher = Researcher

# Call methods
let answer = researcher.process("What is machine learning?")
print(answer)

# Access state
print("Total queries:", researcher.query_count)
print("Notes:", researcher.get_notes())
```

## Built-in Functions Reference

### I/O Functions

| Function | Description | Example |
|----------|-------------|---------|
| `print(...)` | Print values to console (variadic) | `print("x =", x)` |
| `input(prompt)` | Read a line from stdin | `let name = input("Name: ")` |
| `load(path)` | Load CSV file as list of lists | `let data = load("data.csv")` |

### Tensor Creation

| Function | Description | Example |
|----------|-------------|---------|
| `zeros(shape)` | Create tensor filled with zeros | `zeros([3, 3])` |
| `ones(shape)` | Create tensor filled with ones | `ones([2, 4])` |
| `rand(shape)` | Create tensor with random values [0, 1) | `rand([10, 10])` |
| `arange(start, end, step)` | Create range tensor | `arange(0, 10, 2)` |

### Math Functions

| Function | Description | Example |
|----------|-------------|---------|
| `sum(x)` | Sum of all elements | `sum([1, 2, 3])` → `6` |
| `mean(x)` | Mean of all elements | `mean([1, 2, 3])` → `2.0` |
| `max(x)` | Maximum element | `max([1, 5, 3])` → `5` |
| `min(x)` | Minimum element | `min([1, 5, 3])` → `1` |
| `sqrt(x)` | Square root | `sqrt(16)` → `4` |
| `exp(x)` | Exponential (e^x) | `exp(1)` → `2.718...` |
| `log(x)` | Natural logarithm | `log(2.718)` → `1` |
| `abs(x)` | Absolute value | `abs(-5)` → `5` |

### Activation Functions

| Function | Description |
|----------|-------------|
| `relu(x)` | Rectified Linear Unit: max(0, x) |
| `sigmoid(x)` | Sigmoid: 1 / (1 + e^-x) |
| `tanh(x)` | Hyperbolic tangent |
| `softmax(x)` | Softmax normalization |

### Utility Functions

| Function | Description | Example |
|----------|-------------|---------|
| `len(x)` | Length of list/string/tensor | `len([1, 2, 3])` → `3` |
| `type(x)` | Type name as string | `type(42)` → `"number"` |
| `shape(x)` | Tensor shape as list | `shape([[1,2],[3,4]])` → `[2, 2]` |
| `range(start, end, step)` | Create list of integers | `range(0, 5)` → `[0, 1, 2, 3, 4]` |
| `str(x)` | Convert to string | `str(42)` → `"42"` |
| `int(x)` | Convert to integer | `int("42")` → `42` |
| `float(x)` | Convert to float | `float("3.14")` → `3.14` |

### AI & Network Functions

| Function | Description | Example |
|----------|-------------|---------|
| `ai_chat(message, [api_key], [model])` | Chat with Claude AI | `ai_chat("Hello!")` |
| `http_get(url, [headers])` | HTTP GET request | `http_get("https://api.example.com")` |
| `http_post(url, data, [headers])` | HTTP POST request | `http_post(url, body, headers)` |
| `json_parse(str)` | Parse JSON string | `json_parse('{"a": 1}')` |
| `json_get(obj, key)` | Get value from parsed JSON | `json_get(parsed, "name")` |
| `env(name)` | Get environment variable | `env("ANTHROPIC_API_KEY")` |

**Note:** `ai_chat()` requires an Anthropic API key. Set it via `ANTHROPIC_API_KEY` environment variable or pass as second argument.

### String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `split(str, delim)` | Split string by delimiter | `split("a,b,c", ",")` → `["a", "b", "c"]` |
| `join(list, delim)` | Join list with delimiter | `join(["a", "b"], "-")` → `"a-b"` |
| `replace(str, old, new)` | Replace occurrences | `replace("hello", "l", "L")` → `"heLLo"` |
| `trim(str)` | Remove leading/trailing whitespace | `trim("  hi  ")` → `"hi"` |
| `lower(str)` | Convert to lowercase | `lower("Hello")` → `"hello"` |
| `upper(str)` | Convert to uppercase | `upper("hello")` → `"HELLO"` |
| `contains(str, sub)` | Check if string contains substring | `contains("hello", "ell")` → `true` |
| `starts_with(str, prefix)` | Check prefix | `starts_with("hello", "he")` → `true` |
| `ends_with(str, suffix)` | Check suffix | `ends_with("hello", "lo")` → `true` |
| `slice(str, start, [end])` | Extract substring | `slice("hello", 1, 4)` → `"ell"` |

### Neural Network Training

| Function | Description | Example |
|----------|-------------|---------|
| `nn_create(sizes, [activation])` | Create neural network | `nn_create([784, 128, 10], "relu")` |
| `nn_train(net, X, y, epochs, lr, batch_size, verbose)` | Train network | `nn_train(net, X, y, 100, 0.01, 32, true)` |
| `nn_predict(net, input)` | Make prediction | `nn_predict(net, [1.0, 2.0])` |
| `nn_save(net, path)` | Save trained model | `nn_save(net, "model.bin")` |
| `nn_load(path)` | Load saved model | `nn_load("model.bin")` |
| `argmax(tensor)` | Index of maximum value | `argmax([0.1, 0.7, 0.2])` → `1` |
| `tensor_get(tensor, idx)` | Get tensor element | `tensor_get(t, 0)` → first element |

**Training supports:**
- **Classification**: Integer labels or one-hot encoded targets
- **Regression**: Scalar or vector outputs
- **Activations**: `"relu"`, `"tanh"`, `"sigmoid"`
- **Optimizer**: Adam with configurable learning rate
- **Automatic differentiation**: Full backpropagation through computation graph

## Layer Types Reference

### Trainable Layers

| Layer | Description | Parameters |
|-------|-------------|------------|
| `Linear(in, out)` | Fully connected layer | Input features, output features |
| `Conv2D(in, out, k)` | 2D convolution | Input channels, output channels, kernel size |

### Activation Layers

| Layer | Description |
|-------|-------------|
| `ReLU()` | Rectified Linear Unit |
| `Sigmoid()` | Sigmoid activation |
| `Tanh()` | Hyperbolic tangent |
| `Softmax()` | Softmax normalization |

### Regularization & Normalization

| Layer | Description | Parameters |
|-------|-------------|------------|
| `Dropout(p)` | Dropout regularization | Drop probability |
| `BatchNorm(features)` | Batch normalization | Number of features |

### Pooling & Reshaping

| Layer | Description | Parameters |
|-------|-------------|------------|
| `MaxPool2D(k)` | Max pooling | Kernel size |
| `Flatten()` | Flatten to 1D | None |

## Examples

### Hello World

```python
# examples/hello.dt
print("Hello, Duckt!")

let name = "World"
print("Hello, " + name + "!")

fn greet(person):
    return "Welcome, " + person + "!"

print(greet("User"))
```

### Tensor Operations

```python
# examples/tensors.dt
let a = [[1, 2], [3, 4]]
let b = [[5, 6], [7, 8]]

print("A:", a)
print("B:", b)
print("A + B:", a + b)
print("A @ B:", a @ b)
print("A^T:", a.T)
```

### Simple Neural Network

```python
# examples/neural_net.dt
model SimpleNet:
    layer fc1 = Linear(4, 8)
    layer fc2 = Linear(8, 2)

    fn forward(x):
        x = relu(fc1(x))
        return softmax(fc2(x))

let net = SimpleNet
let input = [1.0, 2.0, 3.0, 4.0]
let output = net.forward(input)
print("Output:", output)
```

### AI Agent

```python
# examples/agent.dt
agent Calculator:
    state history = []

    tool add(a, b):
        return a + b

    tool multiply(a, b):
        return a * b

    fn calculate(op, x, y):
        let result = 0
        if op == "add":
            result = add(x, y)
        if op == "multiply":
            result = multiply(x, y)
        let record = op + "(" + str(x) + ", " + str(y) + ") = " + str(result)
        history = history + [record]
        return result

let calc = Calculator
print(calc.calculate("add", 5, 3))       # 8
print(calc.calculate("multiply", 4, 7))  # 28
print("History:", calc.history)
```

### Train XOR Network

```python
# examples/train_xor.dt
# Train a network to learn the XOR function

let X = [[0, 0], [0, 1], [1, 0], [1, 1]]
let y = [0, 1, 1, 0]

let net = nn_create([2, 16, 2], "tanh")
nn_train(net, X, y, 1000, 0.1, 4, true)

# Test
let pred = nn_predict(net, [0, 1])
print("Predicted class:", argmax(pred))  # Output: 1
```

### Regression Training

```python
# examples/train_regression.dt
# Train a network to learn y = x^2

let X = [[-1.0], [-0.5], [0.0], [0.5], [1.0]]
let y = [1.0, 0.25, 0.0, 0.25, 1.0]

let net = nn_create([1, 8, 1], "tanh")
nn_train(net, X, y, 1000, 0.05, 5, true)

# Test
let pred = nn_predict(net, [0.5])
print("f(0.5) =", tensor_get(pred, 0))  # Output: ~0.25
```

### Real AI Chatbot

```python
# examples/ai_chatbot.dt
# Set your API key first: export ANTHROPIC_API_KEY="your-key"

agent AIBot:
    state history = []

    fn chat(question):
        history = history + [question]
        return ai_chat(question)

let api_key = env("ANTHROPIC_API_KEY")
if api_key == none:
    print("Please set ANTHROPIC_API_KEY")
else:
    let bot = AIBot
    let running = true

    while running:
        let user_input = input("You: ")
        if lower(user_input) == "quit":
            running = false
        else:
            print("Bot:", bot.chat(user_input))
```

## Building from Source

### Requirements

- C++17 compatible compiler (clang++ or g++)
- Make

### Build Commands

```bash
# Build release version
make

# Build with debug symbols
make debug

# Run all examples
make examples

# Run a specific file
make run-file FILE=examples/hello.dt

# Start the REPL
make run

# Clean build artifacts
make clean

# Install to /usr/local/bin
sudo make install

# Uninstall
sudo make uninstall
```

## Project Structure

```
Duckt/
├── src/
│   ├── main.cpp           # Entry point, REPL, file execution
│   ├── token.hpp          # Token types and structures
│   ├── lexer.hpp/cpp      # Tokenizer with indentation handling
│   ├── ast.hpp            # AST node definitions
│   ├── parser.hpp/cpp     # Recursive descent parser
│   ├── types.hpp          # Runtime types (Value, Tensor, Layer, etc.)
│   ├── environment.hpp    # Lexical scoping and variable storage
│   ├── interpreter.hpp/cpp # Tree-walking interpreter
│   └── builtins.hpp       # Built-in functions implementation
├── include/
│   └── duckt.hpp          # Public API for embedding
├── examples/
│   ├── hello.dt           # Basic syntax demonstration
│   ├── tensors.dt         # Tensor operations
│   ├── neural_net.dt      # Neural network example
│   ├── agent.dt           # AI agent example
│   ├── chatbot.dt         # Simple rule-based chatbot
│   ├── ai_chatbot.dt      # Real AI chatbot using Claude API
│   ├── train_xor.dt       # Train network on XOR (classification)
│   └── train_regression.dt # Train network on y=x^2 (regression)
├── Makefile               # Build configuration
└── README.md              # This file
```

## Embedding Duckt

Duckt can be embedded in C++ applications:

```cpp
#include "duckt.hpp"

int main() {
    // Run a program from string
    duckt::run(R"(
        let x = 42
        print("x =", x)
    )");

    // Or use the interpreter directly
    duckt::Interpreter interp;

    // Define variables
    interp.globals()->define("my_var", duckt::Value(100));

    // Run code
    duckt::run("print(my_var)");

    return 0;
}
```

## Keywords

Reserved words in Duckt:

```
let     fn      if      else    for     while   return  in
and     or      not     true    false   none    model   layer
agent   tool    state   train   tensor  on      with    epochs  lr
```

## File Extension

Duckt source files use the `.dt` extension by convention.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
