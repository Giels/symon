## SYMON: A tensor-oriented symbolic computation framework

Symon is a tensor-oriented symbolic computation framework, currently using arrayfire as the backend. It aims to make full use of CPUs and GPUs, with a longer term goal of supporting both data and model parallelism.
It is currently in its infancy and can barely be considered to be in an alpha state. In other words, DO NOT USE IT.

# TODO
- Finish basic arrayfire ops (waiting on arrayfire convolution being updated -- or not -- to finish convolution support)
- Cleanup
- More/better tests
- Serialization
- Optimzation passes (common subexpression elimination, identity, negation-negation, etc.)
- Etc.
