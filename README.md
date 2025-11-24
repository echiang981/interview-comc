# The Count of Monte Cristo

## Overview

This repository contains a coding challenge designed to measure your raw ability to code and solve problems quickly. Your task is to build an LLM-powered translation system that translates the first volume of **The Count of Monte Cristo** (_Le Comte de Monte-Cristo_), the classic French novel by Alexandre Dumas.

## The Challenge

Build a system that uses Large Language Models (LLMs) to translate the French source text (`source/comc.txt`) into high-quality English.

## Evaluation Criteria

You will be judged on two primary factors:

1. **Code Quality**: Architecture, readability, error handling, efficiency, and overall software engineering practices
2. **Translation Quality**: Accuracy, fluency, preservation of literary style, and faithfulness to the original text

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Access to an LLM API (e.g., OpenAI, Anthropic, etc.)

### Installation

```bash
# If you are doing this in Python, navigate to the python directory
cd python
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Then edit .env and add your API keys
```

### Source Material

The French source text is located in `source/comc.txt`.

## Your Task

Design and implement a translation system that:

- Reads the French source text
- Uses LLM(s) to translate it to English
- Produces a high-quality English translation

## Deliverables

- Working translation system code
- Translated English output
- Any additional documentation you feel is necessary

Good luck!
