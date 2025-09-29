#!/bin/bash

# Default values
config="main"
split="test"
limit=-1
start=0
workers=8
templates="templates/data_augmentation"
failfast=false

# Define arrays for each country's environment variables
declare -a COUNTRIES=("Haiti" "Moldova" "Pakistan" "Somalia" "Solomon Islands" "Suriname")

declare -a DEMONYMS=("Haitian" "Moldovan" "Pakistani" "Somali" "Solomon Islander" "Surinamese")

declare -a CURRENCIES=("gourde" "leu" "rupee" "shilling" "Solomon Islands dollar" "Suriname dollar")

declare -a CURRENCY_SYMBOLS=("G" "L" "Rs" "SOS" "SI$" "SRD")

declare -a CURRENCY_CONVERSION_RATES=("130.81" "16.62" "281.85" "569.38" "8.23" "38.65")

declare -a CURRENCY_ABBREVIATIONS=("HTG" "MDL" "PKR" "SOS" "SBD" "SRD")

for i in "${!COUNTRIES[@]}"; do
    echo "Running for country: ${COUNTRIES[$i]}"
    python3 src/data_augmentation/augment_data.py \
        --config "$config" \
        --split "$split" \
        --limit "$limit" \
        --start "$start" \
        --workers "$workers" \
        --templates "$templates" \
        --country "${COUNTRIES[$i]}" \
        --demonym "${DEMONYMS[$i]}" \
        --currency "${CURRENCIES[$i]}" \
        --currency_symbol "${CURRENCY_SYMBOLS[$i]}" \
        --currency_conversion_rate "${CURRENCY_CONVERSION_RATES[$i]}" \
        --currency_abbreviation "${CURRENCY_ABBREVIATIONS[$i]}" \
        $( $failfast && echo --failfast )
done
