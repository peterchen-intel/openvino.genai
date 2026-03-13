# Eagle3 Mode Call Sequence Trace

This document describes the call sequence tracing implemented for the Eagle3 speculative decoding mode in OpenVINO GenAI.

## Overview

Eagle3 is a speculative decoding algorithm that uses a draft model to generate candidate tokens and a target model to validate them in parallel. This document explains how to trace the execution flow when `eagle3_mode` is enabled.

## Enabling Trace Logging

The trace logs use the `GENAI_INFO` logging macro, which respects the OpenVINO logging level. To see the Eagle3 trace logs:

1. Set the environment variable:
   ```bash
   export OPENVINO_LOG_LEVEL=INFO
   ```

2. Run your application with Eagle3 speculative decoding enabled

3. Look for log messages with the `[EAGLE3_TRACE]` prefix

## Call Sequence

### 1. Pipeline Initialization

**Entry Point:** `samples/cpp/text_generation/speculative_decoding_lm.cpp`

```cpp
// Sample creates LLMPipeline with draft model
ov::genai::LLMPipeline pipe(
    main_model_path,
    main_device,
    ov::genai::draft_model(draft_model_path, draft_device));
```

**Trace Logs:**
```
[EAGLE3_TRACE] Draft model detected, eagle3_mode: true
[EAGLE3_TRACE] Initializing StatefulEagle3LLMPipeline
```

**Location:** `src/cpp/src/llm/pipeline.cpp:162-171`

The pipeline detects `eagle3_mode` from the draft model properties and creates a `StatefulEagle3LLMPipeline` instead of the standard `StatefulSpeculativeLLMPipeline`.

### 2. StatefulEagle3LLMPipeline Constructor

**Trace Logs:**
```
[EAGLE3_TRACE] StatefulEagle3LLMPipeline constructor started
[EAGLE3_TRACE] Target device: CPU, Draft device: CPU
[EAGLE3_TRACE] Draft iterations set to: 4
[EAGLE3_TRACE] Hidden layers list extracted: size = 3
[EAGLE3_TRACE] Starting model transformations
[EAGLE3_TRACE] share_vocabulary() completed
[EAGLE3_TRACE] extract_d2t_mapping_table() completed
[EAGLE3_TRACE] transform_hidden_state(target_model) completed
[EAGLE3_TRACE] move_fc_from_draft_to_main() completed
[EAGLE3_TRACE] transform_hidden_state(draft_model) completed
[EAGLE3_TRACE] Validation window size: 5
[EAGLE3_TRACE] Creating Eagle3DraftWrapper
[EAGLE3_TRACE] Eagle3DraftWrapper created and configured
[EAGLE3_TRACE] Creating Eagle3TargetWrapper
[EAGLE3_TRACE] Eagle3TargetWrapper created
[EAGLE3_TRACE] StatefulEagle3LLMPipeline constructor completed
```

**Location:** `src/cpp/src/speculative_decoding/stateful/eagle3_strategy.cpp:445-541`

Key operations:
- Extracts `hidden_layers_list` from draft model properties (must contain exactly 3 layers)
- Performs model transformations (vocabulary sharing, hidden state extraction, FC layer movement)
- Creates `Eagle3DraftWrapper` and `Eagle3TargetWrapper` instances
- Configures validation window size based on `num_assistant_tokens`

### 3. Generation (generate_tokens)

**Entry Point:** User calls `pipe.generate(prompt, config, streamer)`

**Trace Logs:**
```
[EAGLE3_TRACE] generate_tokens() called
[EAGLE3_TRACE] max_new_tokens: 100, eos_token_id: 2
[EAGLE3_TRACE] Prompt length: 15
[EAGLE3_TRACE] Resetting model states
[EAGLE3_TRACE] Initializing sequences for target and draft models
```

**Location:** `src/cpp/src/speculative_decoding/stateful/eagle3_strategy.cpp:542-683`

### 4. Phase 1: Prefill

**Trace Logs:**
```
[EAGLE3_TRACE] === Phase 1: Prefill - Processing prompt tokens ===
[EAGLE3_TRACE] Calling m_target->forward() for prefill with 15 tokens
[EAGLE3_TRACE] Eagle3TargetWrapper::forward() called with input_token_count=15, sample_count=1, num_tokens_to_validate=0
[EAGLE3_TRACE] Eagle3TargetWrapper: Calling infer()
[EAGLE3_TRACE] Eagle3TargetWrapper: Sampling tokens (validation mode: false)
[EAGLE3_TRACE] Eagle3TargetWrapper: Sampled 1 tokens
[EAGLE3_TRACE] Eagle3TargetWrapper::forward() completed
[EAGLE3_TRACE] Prefill completed, initial token sampled: 1234
```

The target model processes all prompt tokens and generates the first output token. Hidden states are stored for the draft model to use.

### 5. Phase 2: Speculative Decoding Loop

**Trace Logs (per iteration):**
```
[EAGLE3_TRACE] === Phase 2: Speculative Decoding Loop ===
[EAGLE3_TRACE] --- Iteration 1: Calling run_speculative_iteration() ---
```

Each iteration calls `run_speculative_iteration()`, which has 5 steps:

#### Step 1: Generate First Draft Token (Using Target Hidden States)

**Trace Logs:**
```
[EAGLE3_TRACE] run_speculative_iteration() started with input_token_count: 16
[EAGLE3_TRACE] Step 1: Generating first draft token using target hidden states
[EAGLE3_TRACE] Calling m_draft->forward() for first token
[EAGLE3_TRACE] Eagle3DraftWrapper::forward() called with input_token_count=16, use_target_hidden=true
[EAGLE3_TRACE] Using target hidden states
[EAGLE3_TRACE] Eagle3DraftWrapper: Calling infer()
[EAGLE3_TRACE] Eagle3DraftWrapper: Sampling tokens
[EAGLE3_TRACE] Eagle3DraftWrapper::forward() completed, sampled token: 5678
[EAGLE3_TRACE] First draft token sampled: 5678
```

The draft model uses hidden states from the target model to generate its first candidate token.

#### Step 2: Generate Additional Draft Tokens (Using Internal Hidden States)

**Trace Logs:**
```
[EAGLE3_TRACE] Step 2: Generating 3 additional draft tokens using internal hidden states
[EAGLE3_TRACE] Draft iteration 1: Calling m_draft->forward()
[EAGLE3_TRACE] Eagle3DraftWrapper::forward() called with input_token_count=1, use_target_hidden=false
[EAGLE3_TRACE] Using internal hidden states
[EAGLE3_TRACE] Eagle3DraftWrapper: Calling infer()
[EAGLE3_TRACE] Eagle3DraftWrapper: Sampling tokens
[EAGLE3_TRACE] Eagle3DraftWrapper::forward() completed, sampled token: 9012
[EAGLE3_TRACE] Draft iteration 1: token sampled: 9012
... (repeats for iterations 2 and 3)
```

The draft model generates additional candidate tokens using its own internal hidden states.

#### Step 3: Validate Draft Tokens with Target Model

**Trace Logs:**
```
[EAGLE3_TRACE] Step 3: Validating 4 draft tokens with target model
[EAGLE3_TRACE] Calling m_target->forward() for validation with window size: 5
[EAGLE3_TRACE] Eagle3TargetWrapper::forward() called with input_token_count=5, sample_count=5, num_tokens_to_validate=4
[EAGLE3_TRACE] Eagle3TargetWrapper: Calling infer()
[EAGLE3_TRACE] Eagle3TargetWrapper: Sampling tokens (validation mode: true)
[EAGLE3_TRACE] Eagle3TargetWrapper: Sampled 5 tokens
[EAGLE3_TRACE] Eagle3TargetWrapper::forward() completed
[EAGLE3_TRACE] Validation completed, received 5 validated tokens
[EAGLE3_TRACE] Accepted 3 draft tokens out of 4, new target token: 3456
```

The target model validates all draft predictions in parallel and generates a new token. The sampler determines how many draft tokens are accepted.

#### Step 4: Synchronize Sequences and KV Cache

**Trace Logs:**
```
[EAGLE3_TRACE] Step 4: Synchronizing sequences and KV cache
[EAGLE3_TRACE] Trimming KV cache for 1 rejected tokens
```

Both models synchronize their token sequences and trim KV cache for rejected draft tokens.

#### Step 5: Update Hidden States

**Trace Logs:**
```
[EAGLE3_TRACE] Step 5: Updating hidden states for next iteration
[EAGLE3_TRACE] run_speculative_iteration() completed: accepted=3, next_window=4, EOS=false
[EAGLE3_TRACE] Iteration 1 completed: accepted 3 tokens, next window size: 4, EOS: false
```

Hidden states are sliced to include only accepted tokens and stored for the next iteration.

### 6. Phase 3: Finalization

**Trace Logs:**
```
[EAGLE3_TRACE] === Phase 3: Finalization ===
[EAGLE3_TRACE] Total generated tokens: 100, Total draft accepted: 285, Total draft generated: 400
```

**Location:** `src/cpp/src/speculative_decoding/stateful/eagle3_strategy.cpp:632-683`

Final statistics are collected and performance metrics are calculated.

## Key Components

### Eagle3TargetWrapper

- **Purpose:** Validates draft predictions and generates final output tokens
- **Key Methods:**
  - `initialize_sequence()`: Sets up prompt tokens (full sequence)
  - `forward()`: Executes inference and sampling
  - `infer()`: Runs model inference with standard inputs (input_ids, attention_mask, position_ids)

### Eagle3DraftWrapper

- **Purpose:** Generates candidate tokens speculatively
- **Key Methods:**
  - `initialize_sequence()`: Sets up prompt tokens (uses tokens[1:] per Eagle3 spec)
  - `forward()`: Executes inference and sampling
  - `infer()`: Runs model inference with hidden states input

### InferContext

Configures each forward pass:
- `input_token_count`: Number of tokens to process
- `sample_count`: Number of positions to sample from
- `use_target_hidden`: Whether to use target's hidden states
- `target_sequence`: Source sequence for hidden states
- `num_tokens_to_validate`: Draft tokens to validate (0 for standard sampling)

## Configuration

Eagle3 mode is enabled via draft model properties:

```cpp
// In model configuration
draft_model_properties["eagle3_mode"] = true;
draft_model_properties["hidden_layers_list"] = std::vector<int32_t>{layer1, layer2, layer3};
```

The `hidden_layers_list` specifies which layers to extract hidden states from (must be exactly 3 layers).

## Performance Metrics

Eagle3 tracks:
- **Acceptance Rate:** Percentage of draft tokens accepted by target model
- **Draft Generated:** Total draft tokens generated
- **Draft Accepted:** Total draft tokens accepted
- **Speedup:** Improvement over non-speculative decoding

Access via:
```cpp
auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(result.extended_perf_metrics);
```

## Files Modified

1. `src/cpp/src/llm/pipeline.cpp` - Eagle3 mode detection and pipeline creation
2. `src/cpp/src/speculative_decoding/stateful/eagle3_strategy.cpp` - Core Eagle3 implementation with tracing

## References

- Eagle3 Paper: [Link to paper if available]
- OpenVINO GenAI Documentation: https://github.com/openvinotoolkit/openvino.genai
- Sample: `samples/cpp/text_generation/speculative_decoding_lm.cpp`
