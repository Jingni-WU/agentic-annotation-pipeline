# Agentic Annotation Pipeline

A LangGraph-based multi-agent annotation pipeline with human-in-the-loop review and iterative guideline refinement.

## Overview

This project explores an agentic workflow for building and improving annotation datasets. Multiple LLM agents take on different roles in the annotation process, including taxonomy generation, independent annotation, disagreement analysis, and adjudication.

The pipeline uses **Cohen's Kappa** to measure inter-annotator agreement. When agreement falls below a predefined threshold, the system analyzes disagreements, revises the annotation guidelines, requests human approval, and re-runs annotation until the agreement threshold or maximum number of iterations is reached.

## Workflow

```text
Annotation Request
        ↓
Taxonomy & Guideline Generation
        ↓
Human Review
        ↓
Annotator A ─────┐
                 ├──→ Inter-Annotator Agreement (Cohen's κ)
Annotator B ─────┘
                         ↓
                 κ meets threshold?
                    /          \
                  Yes           No
                   ↓             ↓
             Adjudication   Disagreement Analysis
                   ↓             ↓
             Data Cleaning  Guideline Revision
                   ↓             ↓
                 Export      Human Review
                                 ↓
                            Re-annotation ↺
```

## Current Features

- LLM-generated annotation taxonomy and guidelines
- Human-in-the-loop taxonomy/guideline review
- Independent multi-model annotation
- Inter-annotator agreement using Cohen's Kappa
- Automated disagreement analysis
- LLM-based adjudication
- Duplicate and invalid-label cleaning
- JSON and CSV export
- Iterative guideline refinement based on annotation disagreement

## Tech Stack

- Python
- LangGraph
- LangChain
- Anthropic Claude / other LLM backends
- Pydantic
- scikit-learn

## Status

🚧 **MVP / Work in Progress**

Current development focuses on building the iterative feedback loop and evaluating whether disagreement-driven guideline refinement improves annotation agreement and final annotation quality.
