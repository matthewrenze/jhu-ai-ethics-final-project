# Natural Language Explanations for Explainable AI Systems using Large Language Models with Self-correcting Architectures

**Author:** Matthew Renze  
**Class:** EN.705.612  
**Date:** 2023-04-30

## Abstract
In this paper, we explore the generation of Natural Language Explanations (NLEs) for eXplainable AI (XAI) systems using Large Language Models (LLMs) with self-correcting architectures.

First, a dataset was created from a modified version of the COMPAS database and SHapley Additive exPlanations (SHAP). Next, a rule-based NLE generator was hand-coded to assess optimal NLE performance. Then, a Generative Pretrained Transformer (GPT) was used with few-shot learning to explain each case record using their corresponding SHAP values. A second GPT task verified each NLE and identified factual errors. A third GPT task corrected NLEs based on any factual errors identified. Finally, a hand-coded rule-based evaluator was used to assess the final performance. 

GPT-4 with self-correction outperformed all other GPT models in terms of factual accuracy. However, it underperformed relative to GPT-3.5 in terms of runtime performance and cost efficiency. This research demonstrates the potential of using GPTs to create more trustworthy AI, highlighting areas of further investigation and improvement.

## Paper
- [Natural Language Explanations for Explainable AI Systems using Large Language Models with Self-correcting Architectures](research-paper.pdf)

## Code
- [Prepare](code/Prepare/) - contains the data pre-processing scripts
- [Explain](code/Explain/) - contains the rule- and gpt-based explainers
- [Correct](code/Correct/) - contains the gpt-based verifier and corrector
- [Evaluate](code/Evaluate/) - contains the performance evaluation scripts
- [Analyze](code/Analyze/) - contains the performance, runtime, and cost analysis

## Data
- [Source](data/Source/) - contains the source COMPAS Synthetic data
- [Prepared](data/Prepared/) - contains the pre-processed data
- [Prompts](data/Prompts/) - contains the GPT system prompts
- [Templates](data/Templates/) - contains templates for rule- and GPT-based tasks
- [Examples](data/Examples/) - contains the few-shot learning examples
- [Cases](data/Cases/) - contains the plain-text case records
- [Explainations](data/Explanations/) - contains the generated explainations
- [Verifications](data/Verifications/) - contains the verifications for each explaination
- [Errors](data/Errors) - contains the factual error identified in each verification
- [Corrections](data/Corrections/) - contains the corrected explainations
- [Results](data/Results/) - contains the results of the model evalaluation

## Analysis
- [CSV](Analysis/CSV/) - contains analysis output in tabular form
- [PNG](Analysis/PNG/) - contains PNG images of data visualizations
- [SVG](Analysis/SVG/) - contains SVG files for data visualizations
