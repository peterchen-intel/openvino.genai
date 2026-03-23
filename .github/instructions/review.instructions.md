---
applyTo: "**"
excludeAgent: "coding-agent"
---

# Code Review Instructions for PRs

When performing a code review on a Pull Request, additionally follow this protocol:

1. PR description must be aligned with [../pull_request_template.md](../pull_request_template.md) and its checklist must be filled out. If not, request the author to update the description and checklist before proceeding with the review.
2. If the documentation is updated, PR description must include a link to the corresponding documentation deployed on the fork.
3. PR description must be up to date and include all information about the changes.
4. Include C++ Core Guidelines references in review comments.
5. Python Bindings: if C++ APIs are changed, check if the corresponding Python pybind11 wrappers in src/python need updates.
6. Documentation: ensure that any new public APIs have docstrings in C++ headers and Python bindings. Ensure that new public APIs have documentation updated in /site.
7. Test Coverage: ensure that new features or changes have corresponding tests.
8. Verify that the result of every newly introduced function is used in at least one call site except for `void` functions.
