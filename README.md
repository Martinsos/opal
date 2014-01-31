# Swimd

Swimd is C/C++ library for fast query vs database sequence alignment using SSE.
SSE4.1 or higher version is required.
Swimd is implemented mainly by Rognes's "Faster Smith-Waterman database searches with inter-sequence SIMD parallelisation".
Main difference is that Swimd offers 4 alignment modes instead of just Smith-Waterman.

### Alignment modes
Swimd offers 4 different modes of alignment: 1 local and 3 global modes, explained below.

