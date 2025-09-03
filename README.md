# cppcon2025_autodiff
Notes, resources, and slides for my cppcon 2025 talk, "From Bayesian Inference to LLMs: Modern C++ Optimizations for Reverse‑Mode Automatic Differentiation"


## Abstract:
Reverse-mode automatic differentiation (AD) powers everything from back-propagation that trains trillion-parameter large-language models to the Stan programming language’s Bayesian-inference engines. Performance tricks like arena allocators, expression templates, SIMD-friendly data structures, GPU-kernel fusion, and template metaprogramming find their way inside C++ AD libraries. Milliseconds saved per gradient can compound into hours of wall-time wins.

This session dissects the engineering behind those performance wins, showcasing improvements across different C++ AD libraries over time. Attendees will see how contemporary C++ AD techniques can make AD so fast that there is rarely a need to add handwritten derivatives to your program. These techniques extend cleanly to many-core and GPU back-ends. The talk assumes familiarity with modern C++ but **no prior exposure to automatic differentiation.**

Current Time-Boxed Outline:
0 – 2 min Opening hook
Split-screen benchmark slide: UK Covid and ChatGPT Use autodiff

1 – 10 min AD Fundamentals
- What is Reverse Mode AutoDiff
- Show graph building out reverse mode AD
- Minimal reverse-mode toy example

10 – 20 min Basic Patterns
Go over basic patterns used in AD
AD scalar design:
 - From shared pointer impl to arena allocator to lambda functions
Operator Overloading:
CPU + dynamic tape baseline times (e.g., 2.3 s / 3.1 s)
Show a baseline for something like sin(x) * log(y * y)
Progressive Benchmark #1 (arena win)

20:30 min AoS vs. SoA Benchmark
Show SoA vs AoS
Show limitations of SoA approach for things like aliasing

30:45 min Static vs. Dynamic Tapes
Flexibility vs. speed trade-offs
Benchmark difference between static and dynamic tape

45:55 min Operator Fusion & Expression Templates
SIMD-friendly fusion, template-metaprogramming tricks
FastAD
Progressive Benchmark #3 (fusion win)

55:60 min Parallel & GPU Back-Ends
Divide-and-conquer reverse pass, GPU kernel fusion & async exec
Progressive Benchmark #4 (GPU win)


54 – 60 min Q & A / Buffer
