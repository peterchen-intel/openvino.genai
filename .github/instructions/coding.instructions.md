---
applyTo: "**"
excludeAgent: "code-review"
---

# General Coding Guidelines

Follow these rules when writing or modifying code in this repository:

1. Follow C++ Core Guidelines strictly.
2. Performance: avoid `dynamic_cast` in hot paths (inference loops). Use `static_cast` or redesign if the type is known.
3. Avoid copies: large data structures (like tensors) must be passed by reference or moved, not copied.
4. Pass non-fundamental values by `const` reference wherever possible.
5. Exceptions: use `OPENVINO_ASSERT(condition, ...)` for checks instead of `if` + `OPENVINO_THROW(...)` or `throw`.
6. Formatting & Safety:
   - No `using namespace std;`.
   - No `auto` for primitive types where it obscures readability.
   - Use `const` and `constexpr` wherever possible.
7. Follow constructors and member initializer lists style instead of direct assignments in the constructor body.
8. When initial container values are known upfront, prefer initializer-list / brace-initialization over constructing an empty container and immediately inserting values.
9. Make sure the function names are descriptive.
10. Check for variables with different names but similar meaning or aliasing.
11. Avoid duplicate code. Ensure that common functionality is extracted into reusable functions or utilities.
12. Avoid pronouns in comments and names to make the statements concise.
13. Unused functions and constructors aren't allowed except for in `debug_utils.hpp`.
14. `debug_utils.hpp` must never be included.
15. Assumptions on the user's behalf aren't allowed. For example, the implementation shouldn't adjust config values silently or with a warning; it should throw an exception instead.
16. Samples:
    - Avoid adding new samples unless there is a strong, clearly justified reason.
    - Keep command‑line arguments in samples minimal. Prefer hardcoding values.
    - Ensure new samples have corresponding tests.
