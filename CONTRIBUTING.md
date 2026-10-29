# Contributing

Bug reports and reproducibility questions are welcome through GitHub Issues.
For code changes, create a focused branch, add tests for changed behavior, and
run `PYTHONPATH=src pytest -q` before opening a pull request. Do not commit
OneStop data, participant-level exports, checkpoints, or local absolute paths.

Scientific changes must state whether they were specified before or after the
reported test results were inspected. Existing result artifacts and frozen
protocols must not be overwritten.
