# cppcon2025_autodiff

Notes, resources, and slides for my cppcon 2025 talk, "From Bayesian Inference to LLMs: Modern C++ Optimizations for Reverse‑Mode Automatic Differentiation"

## Repository Structure

-   code

    \- C++ code examples for each of the AD designs in the talk along with benchmarks. You can build this with cmake as

```         
  cmake -S . -B "build" 
  ./run_all.sh
```

-   papers

    \- Useful papers and resources for AD research └── slides - Latex and images for the slides

-   slides

    \- Latex and images used in the talk
    \- The slides use `xelatexmk -shell-escape` to build. In vscode you can use latex workshop and add the following to `settings.json` to compile.

``` json
    "latex-workshop.latex.recipes": [
      {
        "name": "latexmk (xelatex)",
        "tools": [
          "xelatexmk",
        ],
        "args": [
          "-shell-escape"
        ]
      }, # Rest of your recipe
    
```