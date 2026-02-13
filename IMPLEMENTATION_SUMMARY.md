# Summary: Eagle3 Call Sequence Tracing Implementation

## Overview

This PR implements comprehensive call sequence tracing for the Eagle3 speculative decoding mode in OpenVINO GenAI. The implementation allows developers to trace the complete execution flow when `eagle3_mode` is enabled, making it easier to understand and debug the Eagle3 algorithm.

## Changes Made

### 1. Source Code Modifications

#### a. `src/cpp/src/llm/pipeline.cpp` (+4 lines)
- Added trace log when eagle3_mode is detected
- Logs which pipeline type is being initialized (Eagle3 vs FastDraft)

#### b. `src/cpp/src/speculative_decoding/stateful/eagle3_strategy.cpp` (+84 lines)
- Added `#include "logger.hpp"` to access logging infrastructure
- **StatefulEagle3LLMPipeline Constructor**: Traces all initialization steps
  - Device configuration
  - Draft iterations setup
  - Hidden layers extraction
  - Model transformations (vocabulary sharing, hidden state setup, FC layer movement)
  - Wrapper creation and configuration
  
- **generate_tokens() method**: Traces the three main phases
  - Configuration logging (max tokens, EOS token, prompt length)
  - Phase 1: Prefill execution
  - Phase 2: Speculative decoding loop (per-iteration logging)
  - Phase 3: Finalization with statistics
  
- **run_speculative_iteration() method**: Traces all 5 steps of each iteration
  - Step 1: First draft token generation using target hidden states
  - Step 2: Additional draft tokens using internal hidden states
  - Step 3: Target model validation
  - Step 4: Sequence and KV cache synchronization
  - Step 5: Hidden state updates
  
- **Eagle3TargetWrapper::forward()**: Traces validation operations
  - Input parameters
  - Inference execution
  - Token sampling (validation vs standard mode)
  
- **Eagle3DraftWrapper::forward()**: Traces draft generation
  - Input parameters
  - Hidden state source (target vs internal)
  - Inference execution
  - Token sampling

### 2. Documentation

#### a. `EAGLE3_CALL_TRACE.md` (+262 lines)
Comprehensive documentation covering:
- Overview of Eagle3 speculative decoding
- How to enable trace logging
- Complete call sequence with example log outputs for each phase
- Detailed explanation of each execution phase
- Key components (Eagle3TargetWrapper, Eagle3DraftWrapper, InferContext)
- Configuration details
- Performance metrics information
- File references

### 3. Demo Script

#### a. `eagle3_trace_demo.py` (+153 lines)
Python demonstration script that:
- Shows how to enable logging via `OPENVINO_LOG_LEVEL` environment variable
- Accepts command-line arguments for model paths, devices, and generation parameters
- Creates LLMPipeline with Eagle3 draft model
- Runs generation with detailed tracing
- Displays results and performance metrics
- Provides analysis of the traced call sequence

## Technical Details

### Logging Infrastructure

- Uses existing `GENAI_INFO` macro from `logger.hpp`
- All logs prefixed with `[EAGLE3_TRACE]` for easy filtering
- Respects `OPENVINO_LOG_LEVEL` environment variable
- Zero performance impact when logging is disabled

### Key Trace Points

1. **Pipeline Creation**: Detection and initialization
2. **Model Setup**: All transformation operations
3. **Prefill Phase**: Initial prompt processing
4. **Speculative Iterations**: Complete draft-validate cycle
5. **Finalization**: Statistics and metrics collection

### Log Format

```
[EAGLE3_TRACE] <description>: <details>
```

Examples:
```
[EAGLE3_TRACE] StatefulEagle3LLMPipeline constructor started
[EAGLE3_TRACE] Draft iterations set to: 4
[EAGLE3_TRACE] Calling m_target->forward() for prefill with 15 tokens
[EAGLE3_TRACE] Accepted 3 draft tokens out of 4, new target token: 3456
```

## Testing

### Manual Testing
The implementation can be tested using:

1. **C++ Sample**: `samples/cpp/text_generation/speculative_decoding_lm.cpp`
   ```bash
   export OPENVINO_LOG_LEVEL=INFO
   ./speculative_decoding_lm <target_model> <draft_model> "prompt"
   ```

2. **Python Demo**: `eagle3_trace_demo.py`
   ```bash
   export OPENVINO_LOG_LEVEL=INFO
   python eagle3_trace_demo.py --target <target_model> --draft <draft_model> --prompt "prompt"
   ```

3. **Python Tests**: Existing tests in `tests/python_tests/test_stateful_speculative_decoding.py`
   - `test_eagle3_string_inputs`
   - `test_eagle3_perf_metrics`

### Code Quality
- ✅ Code review completed: 1 issue found and resolved
- ✅ Security scan (CodeQL): No vulnerabilities found
- ✅ Follows existing OpenVINO GenAI code style
- ✅ Uses existing logging infrastructure
- ✅ No functional changes to algorithm logic

## Benefits

1. **Debugging**: Easy identification of execution flow issues
2. **Performance Analysis**: Understand where time is spent in each phase
3. **Learning**: Helps developers understand the Eagle3 algorithm
4. **Validation**: Verify correct execution of speculative decoding
5. **Documentation**: Serves as living documentation of the call sequence

## Usage

### Enabling Trace Logs

```bash
# Set logging level
export OPENVINO_LOG_LEVEL=INFO

# Run any Eagle3-enabled application
./your_app <arguments>

# Or use Python
python your_script.py
```

### Filtering Trace Logs

```bash
# See only Eagle3 traces
./your_app | grep "EAGLE3_TRACE"

# Save to file
./your_app 2>&1 | tee eagle3_trace.log
```

## Future Enhancements

Possible improvements for future work:
1. Add more granular tracing within model inference
2. Add timing information to each trace point
3. Add visualization tools for trace analysis
4. Extend tracing to continuous batching Eagle3 implementation

## References

- Sample: `samples/cpp/text_generation/speculative_decoding_lm.cpp`
- Implementation: `src/cpp/src/speculative_decoding/stateful/eagle3_strategy.cpp`
- Tests: `tests/python_tests/test_stateful_speculative_decoding.py`
- Documentation: `EAGLE3_CALL_TRACE.md`
- Demo: `eagle3_trace_demo.py`

## Commit History

1. `94f79cf` - Add comprehensive Eagle3 call tracing with GENAI_INFO logs
2. `e3fc90b` - Add comprehensive Eagle3 call sequence documentation
3. `416e0fe` - Add Eagle3 trace demo script with usage instructions
4. `f5722ea` - Fix log level check warning message in demo script
