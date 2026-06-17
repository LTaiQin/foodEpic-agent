---
name: paper-review
description: "Perform structured academic paper reviews with configurable venue-specific criteria"
---

# Paper Review Skill

This skill provides a structured workflow for performing academic paper reviews with venue-specific criteria and detailed analysis.

## Usage

```
/paper-review [venue] [paper_path]
```

### Parameters

- `venue`: Target venue (e.g., TPAMI, CVPR, ICCV, ECCV, NeurIPS, ICLR, IJCV)
- `paper_path`: Path to the paper file (PDF, LaTeX, or Markdown)

## Workflow

### Phase 1: Paper Analysis

1. **Read and understand the paper**
   - Identify the research problem
   - Understand the proposed method
   - Extract key contributions

2. **Generate paper summary**
   - Research question
   - Core methodology
   - Main contributions
   - Overall storyline

### Phase 2: Structured Review

Generate a comprehensive review with the following sections:

#### 1. Paper Summary
- Research problem identification
- Core method explanation
- Contribution summary
- Storyline analysis

#### 2. Novelty & Contribution
- Originality assessment
- Contribution significance
- Comparison with existing work
- Innovation level

#### 3. Technical Quality
- Methodology soundness
- Experimental design
- Results analysis
- Ablation studies

#### 4. Clarity & Presentation
- Writing quality
- Figure/table clarity
- Organization structure
- Language issues

#### 5. Related Work
- Literature coverage
- Comparison fairness
- Citation completeness
- Positioning accuracy

#### 6. Weaknesses & Concerns
- Technical limitations
- Experimental gaps
- Presentation issues
- Potential biases

#### 7. Strengths
- Novel contributions
- Strong experimental results
- Clear presentation
- Practical impact

#### 8. Questions for Authors
- Clarification requests
- Additional experiments
- Methodology details
- Result interpretation

#### 9. Suggestions for Improvement
- Major revisions
- Minor corrections
- Additional analyses
- Presentation improvements

#### 10. Overall Assessment
- Recommendation (Accept/Weak Accept/Weak Reject/Reject)
- Confidence level (1-5)
- Summary justification

## Venue-Specific Criteria

### TPAMI (Transactions on Pattern Analysis and Machine Intelligence)
- Emphasis on methodological novelty
- Strong experimental validation
- Comprehensive literature review
- Clear technical contribution

### CVPR/ICCV/ECCV
- Novel visual understanding
- Strong quantitative results
- Qualitative analysis
- Reproducibility

### NeurIPS/ICLR
- Theoretical contributions
- Novel architectures
- Comprehensive ablations
- Clear insights

## Output Format

The review should be structured as a markdown document with clear sections and actionable feedback. Each section should include:
- Specific observations
- Evidence from the paper
- Constructive suggestions
- Severity ratings where appropriate

## Best Practices

1. **Be constructive** - Focus on improvement suggestions
2. **Be specific** - Reference exact sections, figures, or results
3. **Be fair** - Consider the paper's goals and constraints
4. **Be thorough** - Cover all important aspects
5. **Be professional** - Maintain academic tone throughout