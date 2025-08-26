# GSM8k_Augment

## Overview
**GSM8k_Augment** is a data augmentation pipeline for the GSM8k math word problem dataset.  
It converts culturally specific entities (names, places, currencies, food items, units) into Indian/SI equivalents while preserving the original reasoning style.  
The pipeline leverages LLMs for metric extraction, conversion, templating, recomputation, cultural adaptation, and style-preserving regeneration.

## Features
- Detects culturally specific entities in questions/answers.  
- Converts non-Indian units (e.g., dollars → rupees, miles → kilometers, pounds → kilograms).  
- Replaces culturally specific names, foods, or contexts with India-relevant alternatives.  
- Extracts measurable quantities and aligns them to variable placeholders.  
- Recomputes answers using converted values to ensure mathematical correctness.  
- Regenerates solutions in the **same chain-of-thought style** as the original.  
- Supports **multiprocessing** for parallel processing of dataset instances.  
- Saves **progressive intermediate results** for debugging and inspection.  

## Directory Structure

├── src/ # Source code
│ ├── utils/ # Helper utilities (LLM client, IO, dataset loader)
│ ├── extractor.py # Metric extraction with LLM
│ ├── converter.py # Unit conversion code generation + execution
│ ├── templatizer.py # Question/answer templatization
│ ├── recomputer.py # Recompute using converted assignments
│ ├── styler.py # Style-preserving CoT regeneration
│ ├── cultural_filter.py # Detect culturally specific entities
│ ├── cultural_adapter.py # Replace entities with Indian equivalents
│ └── main.py # Orchestrator script (pipeline entrypoint)
│
├── templates/ # Prompt templates for each pipeline stage
├── run/ # Run scripts
│ └── run.sh
├── out/ # Output directory
│ └── {MODEL}/{timestamp}/
│   ├── augmented/{split}.jsonl # Final augmented outputs
│   └── intermediate/{split}/{idx}.json # Step-by-step intermediates
├── requirements.txt # Python dependencies
└── README.md


## Installation
1. Clone the repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up your .env file with required API keys:
   ```
   VLLM_BASE_URL=...
   VLLM_API_KEY=...
   VLLM_MODEL=...
   GOOGLE_API_KEY=...   # If using Gemini
   ```

## Usage
To run the augmentation pipeline:
   ```sh
   bash run/run.sh
   ```

Or manually:
   ```sh
   python src/main.py --workers 8 --limit 15 --start 10
   ```

## Key Arguments
   ```sh
   --config : GSM8k config (main or socratic)
   --split : Dataset split (train or test)
   --limit : Number of items to process
   --start : Starting index
   --workers : Number of parallel processes
   ```

## Output
Augmented results:
out/{VLLM_MODEL}/{timestamp}/augmented/{split}.jsonl

Intermediate stepwise outputs per item:
out/{VLLM_MODEL}/{timestamp}/intermediate/{split}/{idx}.json