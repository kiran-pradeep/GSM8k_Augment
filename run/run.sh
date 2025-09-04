#!/bin/bash

# Default values
config="main"
split="test"
limit=1
start=0
workers=1
templates="templates"
failfast=false

COUNTRY="India" 
DEMONYM="Indian" 
CURRENCY="rupee" 
CURRENCY_SYMBOL="₹" 
CURRENCY_CONVERSION_RATE="87" # to 1 USD 
CURRENCY_ABBREVIATION="INR"

python3 src/main.py \
    --config "$config" \
    --split "$split" \
    --limit "$limit" \
    --start "$start" \
    --workers "$workers" \
    --templates "$templates" \
    --country "${COUNTRY}" \
    --demonym "${DEMONYM}" \
    --currency "${CURRENCY}" \
    --currency_symbol "${CURRENCY_SYMBOL}" \
    --currency_conversion_rate "${CURRENCY_CONVERSION_RATE}" \
    --currency_abbreviation "${CURRENCY_ABBREVIATION}" \
    $( $failfast && echo --failfast )
