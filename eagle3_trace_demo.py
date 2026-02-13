#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Eagle3 Call Tracing Demo

This script demonstrates how to enable and view the Eagle3 call sequence traces
when running speculative decoding with eagle3_mode enabled.

Prerequisites:
1. OpenVINO GenAI must be built with logging enabled
2. Eagle3-compatible models (target and draft) must be available
3. Set OPENVINO_LOG_LEVEL=INFO environment variable

Usage:
    export OPENVINO_LOG_LEVEL=INFO
    python eagle3_trace_demo.py --target <target_model_path> --draft <draft_model_path> --prompt "Your prompt here"
"""

import os
import sys
import argparse
import openvino_genai as ov_genai


def main():
    parser = argparse.ArgumentParser(description='Eagle3 Call Tracing Demo')
    parser.add_argument('--target', type=str, required=True, help='Path to target model')
    parser.add_argument('--draft', type=str, required=True, help='Path to draft model (Eagle3)')
    parser.add_argument('--prompt', type=str, default='Alan Turing was a', help='Prompt text')
    parser.add_argument('--max-tokens', type=int, default=20, help='Max new tokens to generate')
    parser.add_argument('--num-assistant', type=int, default=4, help='Number of assistant tokens (draft iterations)')
    parser.add_argument('--target-device', type=str, default='CPU', help='Target model device')
    parser.add_argument('--draft-device', type=str, default='CPU', help='Draft model device')
    
    args = parser.parse_args()
    
    # Check if logging is enabled
    log_level = os.environ.get('OPENVINO_LOG_LEVEL', '').upper()
    if log_level not in ['INFO', 'DEBUG', 'TRACE']:
        print("=" * 80)
        print("WARNING: OPENVINO_LOG_LEVEL is not set to INFO or higher!")
        print("To see Eagle3 trace logs, please run:")
        print("  export OPENVINO_LOG_LEVEL=INFO")
        print("  python", " ".join(sys.argv))
        print("=" * 80)
        print()
    
    print("=" * 80)
    print("Eagle3 Call Tracing Demo")
    print("=" * 80)
    print(f"Target model: {args.target}")
    print(f"Draft model: {args.draft}")
    print(f"Target device: {args.target_device}")
    print(f"Draft device: {args.draft_device}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Num assistant tokens: {args.num_assistant}")
    print("=" * 80)
    print()
    
    # Configure draft model with Eagle3
    # Note: eagle3_mode is automatically detected from the draft model structure
    draft_config = {}
    if args.draft_device == "NPU":
        draft_config["NPUW_DEVICES"] = "CPU"
        draft_config["GENERATE_HINT"] = "BEST_PERF"
    
    ov_draft_model = ov_genai.draft_model(args.draft, args.draft_device, **draft_config)
    
    # Configure target model
    target_config = {}
    if args.target_device == "NPU":
        target_config["NPUW_DEVICES"] = "CPU"
        target_config["GENERATE_HINT"] = "BEST_PERF"
    
    print("Creating LLMPipeline with Eagle3 speculative decoding...")
    print("Look for [EAGLE3_TRACE] log messages below:")
    print("-" * 80)
    
    # Create pipeline - this will log the initialization sequence
    ov_pipe = ov_genai.LLMPipeline(
        args.target, 
        args.target_device, 
        target_config, 
        draft_model=ov_draft_model
    )
    
    print("-" * 80)
    print("Pipeline created. Starting generation...")
    print("-" * 80)
    
    # Configure generation
    generation_config = ov_genai.GenerationConfig(
        max_new_tokens=args.max_tokens,
        num_assistant_tokens=args.num_assistant,
        do_sample=False
    )
    
    # Generate - this will log the generation sequence
    results = ov_pipe.generate([args.prompt], generation_config)
    
    print("-" * 80)
    print("Generation complete!")
    print("=" * 80)
    print()
    
    # Display results
    print("Generated text:")
    print(results.texts[0])
    print()
    
    # Display performance metrics if available
    if hasattr(results, 'extended_perf_metrics') and results.extended_perf_metrics:
        print("Performance Metrics:")
        print("-" * 80)
        perf = results.extended_perf_metrics
        
        if hasattr(perf, 'get_num_accepted_tokens'):
            print(f"Draft tokens accepted: {perf.get_num_accepted_tokens()}")
        
        if hasattr(perf, 'main_model_metrics'):
            main_metrics = perf.main_model_metrics
            print(f"Target model - Generated tokens: {main_metrics.get_num_generated_tokens()}")
            print(f"Target model - TTFT: {main_metrics.get_ttft().mean:.2f} ms")
            print(f"Target model - TPOT: {main_metrics.get_tpot().mean:.2f} ms")
        
        if hasattr(perf, 'draft_model_metrics'):
            draft_metrics = perf.draft_model_metrics
            print(f"Draft model - Generated tokens: {draft_metrics.get_num_generated_tokens()}")
            print(f"Draft model - TPOT: {draft_metrics.get_tpot().mean:.2f} ms")
    
    print("=" * 80)
    print()
    print("Trace Analysis:")
    print("-" * 80)
    print("The trace logs above show the complete call sequence for Eagle3 mode:")
    print("1. Pipeline initialization with Eagle3 mode detection")
    print("2. Model transformations (vocabulary sharing, hidden state setup)")
    print("3. Prefill phase (initial prompt processing)")
    print("4. Speculative decoding loop:")
    print("   - Draft model generates candidates using target hidden states")
    print("   - Draft model generates additional candidates using internal states")
    print("   - Target model validates all candidates in parallel")
    print("   - Sequences and KV cache are synchronized")
    print("   - Hidden states are updated for next iteration")
    print("5. Finalization with performance metrics")
    print("=" * 80)


if __name__ == '__main__':
    main()
