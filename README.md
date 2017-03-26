## SYMON: A tensor-oriented symbolic computation framework

Symon is a tensor-oriented symbolic computation framework, currently using arrayfire as the backend. It aims to make full use of CPUs and GPUs, with a longer term goal of supporting both data and model parallelism.
It is currently in its infancy, and any help is appreciated. Questions, comments, and suggestions should be provided as github issues. PRs welcome.

# Intent
Symon is meant to be used as a building block for more user-friendly libraries. Currently, the likes of input validation are expected to be handled by such libraries. This may change in the future. It is primarily aimed at machine learning research and should mostly fill a role similar to theano, while hopefully providing a viable system for deployment in production.
Symon also aims to support both nvidia and non-nvidia hardware (so while the backend might change eventually, opencl support is important to symon).

# Usage
```
use arrayfire::{Array, Dim4, randn};

// Create a graph object
let graph = Graph::new();

// Add variables to the graph
let w = graph.add(Var::new_shared(randn(Dim4::new(&[4, 10, 1, 1]))));
let b = graph.add(Var::new_shared(randn(Dim4::new(&[1, 10, 1, 1]))));
let x = graph.add(Var::new());
let y = graph.add(Var::new());
let lambda = graph.add(Var::new_shared(Array::new(&[1e-3], Dim4::new(&[1, 1, 1, 1]))));

// Forward pass (linear regression)
let wx = graph.add(MatMul::new(w, x));
let wxb = graph.add(Add::new(wx, b));
let diff = graph.add(Sub::new(wxb, y));
let loss = graph.add(Mul::new(diff, diff));

// Not shown: set values for the x and y variables for this epoch

// Gradient descent
let grads = graph.grad(loss, vec![w, b]);
let ldw = graph.add(Mul::new(lambda, grads[0]));
let ldb = graph.add(Mul::new(lambda, grads[1]));

// Weight updates
let new_w = graph.add(Sub::new(w, ldw));
let new_b = graph.add(Sub::new(b, ldb));

// Example: get back the value of wx+b given the current variables' values
let prediction_value = graph.eval(wxb);
```

Most functions keep the same name as in arrayfire, with some exceptions such as
- locate -> Nonzero
- assign -> SetIndex
- select -> If
- i{max, min} -> Arg{max, min}

(This list is not exhaustive)

# Limitations
- Gradients must be taken against scalar values, i.e. the gradient of `SumAll::new(fx)` is valid but not the gradient of `fx`, in general
- Some (less useful) ops aren't defined
- There is no extra processing (e.g. optimizations) on the graph

# TODO
- !! Waiting on arrayfire convolution being updated -- or not -- to finish convolution support
- For/Scan with arbitrary inner graph
- Cleanup
- More thorough/complete tests
- Serialization for the graph
- Optimzation passes (common subexpression elimination, identity, negation-negation, etc.)
- Proper docs

If you would like to help on any of these, please open discussions as github issues and/or make pull requests addressing these. All help is greatly appreciated.
