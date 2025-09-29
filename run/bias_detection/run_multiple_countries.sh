#!/bin/bash

# Default values
config="main"
split="test"
limit=-1
start=0
workers=8
templates="templates/bias_detection/cotZS"
failfast=false


declare -a SOURCE_PATHS=(
    "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Somalia/final/test.jsonl"
    # "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Haiti/augmented/test.jsonl"
    # "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Moldova/augmented/test.jsonl"
    # "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/Pakistan/augmented/test.jsonl"
    # "out/augmented_data/meta-llama--Llama-3_1-70B-Instruct/China/augmented/test.jsonl"
    )


for i in "${!SOURCE_PATHS[@]}"; do
    echo "Running for source: ${SOURCE_PATHS[$i]}"
    python3 src/main.py \
        --config "$config" \
        --split "$split" \
        --source "${SOURCE_PATHS[$i]}" \
        --limit "$limit" \
        --start "$start" \
        --workers "$workers" \
        --templates "$templates" \
        $( $failfast && echo --failfast )
done