# Testing

Testing is split into the following categories:

  * unit test of single functions
  * integration tests of whole mml calls, specifically config compilation and full mode runs
  * performance tests of time critical core functionality
  * manual tests for special purposes, these are not run by default


Performance benchmarks are disabled by default. To do benchmarking, call 
`pytest --benchmark-enable --benchmark-only --benchmark-autosave` for all commits you want to 
benchmark. Afterwards you may call `pytest-benchmark compare` as described in 
the [docs](https://pytest-benchmark.readthedocs.io/en/stable/usage.html#comparison-cli).
