# Climate Question Processing Tools

This directory contains tools for batch processing of climate-related questions, including generation, visualization, and execution. These tools are designed to work together in a pipeline to generate climate questions, validate them, and process them using the Climsight system.

## Tools Overview

### 1. Question Generator (`question_generator.py`)

Generates climate-related questions with geographic locations and validates that the locations are on land.

**Key Features:**
- Generates questions by theme and type (general/specific)
- Validates question locations against land/water boundaries
- Can validate existing question files
- Saves questions to JSON files

**Usage:**
```bash
# Generate new questions
python question_generator.py

# Validate existing questions file
python question_generator.py questions_01.json
```

### 2. Question Map (`question_map.py`)

Interactive visualization tool for exploring climate questions on a map.

**Key Features:**
- Displays questions on an interactive map
- Color-codes questions by theme and type
- Supports filtering by theme and source file
- Shows summary statistics

**Usage:**
```bash
# Run the Streamlit app
streamlit run question_map.py
```

### 3. Question Runner (`question_runner.py`)

Processes batches of questions through the Climsight system to generate answers.

**Key Features:**
- Processes questions in sequence
- Saves progress to allow resuming interrupted runs
- Supports different LLM models
- Generates detailed output files

**Usage:**
```bash
# Process questions from a file
python question_runner.py --questions_file Q_1.json --llm_model gpt-4.1-nano
```

## Batch Processing Workflow

1. **Generate Questions** (optional):
   ```bash
   cd /path/to/climsight
   cd sequential
   python question_generator.py
   ```

2. **Visualize Questions** (optional):
   ```bash
   cd /path/to/climsight
   cd sequential
   streamlit run question_map.py
   ```

3. **Process Questions**:
   ```bash
   cd /path/to/climsight
   cd sequential
   python question_runner.py --questions_file Q_1.json --llm_model gpt-4.1-nano
   ```

## File Structure

- Generated question files: `questions_XX.json`
- Output answers: `answers_output.json`
- Temporary progress: `temporal_output.json`
- Configuration: `config.yml` (in project root)

## Notes

- All scripts should be run from the project sequential directory
- Ensure you have the required environment variables set (e.g., `OPENAI_API_KEY`)
