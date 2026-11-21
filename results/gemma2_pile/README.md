# Gemma 2 2B + The Pile — Exploratory Data Analysis

## Purpose

This analysis characterizes an initial streamed sample of The Pile before Gemma 2 2B model-level experiments. The goal is to identify data-quality issues, source heterogeneity, token-length variation, and potential context-length constraints.

## Sample

- Documents analyzed: 200
- Pile sources represented: 12
- Empty documents: 0
- Exact duplicate documents: 0

## Document Length

- Mean word count: 628.48
- Median word count: 255.0
- Mean Gemma token count: 1181.08
- Median Gemma token count: 407.5
- Minimum Gemma token count: 38
- Maximum Gemma token count: 15626

## Context-Length / Truncation Risk

- 43.5% of sampled documents exceed 512 Gemma tokens.
- 26.0% of sampled documents exceed 1,024 Gemma tokens.
- 13.5% of sampled documents exceed 2,048 Gemma tokens.
- 7.5% of sampled documents exceed 4,096 Gemma tokens.
- 2.0% of sampled documents exceed 8,192 Gemma tokens.

## Key Findings

1. The Pile is highly heterogeneous across text sources.
2. Document length varies substantially across sources, with some technical and scientific sources producing much longer Gemma token sequences.
3. A meaningful fraction of documents exceed common context-length thresholds, so preprocessing must use an explicit truncation or chunking policy.
4. No empty documents or exact duplicates were found in this initial sample.

## Limitation

These findings are based on an initial streamed sample of 200 documents and should be treated as exploratory rather than population-level estimates for the full Pile.

## Recommended Next Step

Define a consistent Gemma-compatible context-length, truncation, or chunking strategy before beginning activation-level model analysis.
