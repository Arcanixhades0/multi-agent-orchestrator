#!/usr/bin/env bash
set -e

echo "=== routing benchmark ==="
python -m benchmarks.routing_benchmark

echo
echo "=== parallel vs sequential benchmark ==="
python -m benchmarks.parallel_vs_sequential_benchmark

echo
echo "=== task completion benchmark ==="
python -m benchmarks.task_completion_benchmark
