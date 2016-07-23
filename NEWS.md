# bmlm 1.1.0

## Major update

bmlm now uses pre-compiled C++ code for the Stan models, which eliminates the need to compile a model each time `mlm()` is run. This significantly speeds up model estimation.

## Minor update

The Stan code used by `mlm()` is now built from separate chunks, allowing more flexible and robust model development.

# bmlm 1.0.0

Initial release to CRAN.