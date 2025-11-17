# Understanding CIRCE Co-accessibility Scores

CIRCE computes a co-accessibility score for each pair of genomic regions, summarizing how strongly their accessibility varies together across single cells. These scores help identify regulatory relationships, shared chromatin contexts, and coordinated patterns of chromatin opening and closing.

Co-accessibility scores generally range from **–1 to +1**, where the sign and magnitude reflect how consistently the two regions share accessibility states.

---

## Positive scores

A **positive co-accessibility score** indicates that two regions tend to be **accessible in the same cells** and **inaccessible in the same cells**.

Biologically, this pattern can arise from:

- **Shared regulatory activity**  
  Distal enhancers, promoters, or other regulatory elements may be active in the same cell populations and therefore open together. This often reflects coordinated transcription factor binding or shared participation in a regulatory program.

- **Chromatin architecture**  
  Regions within the same chromatin loop, sub-domain, or regulatory neighborhood frequently adopt similar accessibility states across cells. A positive score is a statistical signature of being in a shared 3D context, even though it does not directly measure chromatin contacts.

- **Cell-state–dependent activation**  
  During differentiation or stimulus responses, groups of regulatory elements can turn on together. Positive scores highlight these co-activated modules.

### What positive scores *do not* imply

A positive score does **not** mean that the two regions have identical biological roles. Co-accessible elements:

- do **not** necessarily regulate the same gene,  
- may have **unequal** or unrelated regulatory impact,  
- may use different transcription factors,  
- do **not** guarantee physical interaction.

Thus, co-accessibility reflects **coordinated behavior across cells**, not equivalence of function.

---

## Negative scores

A **negative co-accessibility score** means that when one region is accessible in a given cell, the other tends to be inaccessible. This can arise from:

- **Mutually exclusive regulatory programs**  
  For example, two enhancers used in distinct lineages or cell states.

- **Switch-like behavior**  
  Elements that participate in alternative regulatory modes during differentiation or branching trajectories.

Negative scores generally occur less frequently and can be harder to interpret; they do not imply repression or antagonism, only **opposite accessibility patterns** across cells.

---

## What co-accessibility tells you — and what it does not

Co-accessibility scores capture the **population-level coordination** of chromatin accessibility. They are useful for generating hypotheses about regulatory architecture and for linking distal elements to possible targets.

However, co-accessibility alone does **not** confirm:

- direct enhancer–gene regulation,  
- causal influence on gene expression,  
- physical chromatin looping,  
- functional equivalence between elements.

Integrating co-accessibility with additional evidence (expression, motif enrichment, 3D contact data, etc.) provides stronger regulatory interpretation.

---

## Summary

Co-accessibility scores quantify how similarly two genomic regions behave across single cells.

- **Positive scores** reflect coordinated accessibility and shared regulatory context.  
- **Negative scores** indicate mutually exclusive or divergent accessibility patterns.  
- Co-accessibility is a descriptive signal and does not, by itself, establish direct regulatory effects.

These scores help map the structure and dynamics of the regulatory landscape, guiding downstream interpretation and hypothesis generation.

---
## Sources

Pliner,H.A. et al. (2018) Cicero predicts cis-regulatory DNA interactions from single-cell chromatin accessibility data. Mol. Cell, 71, 858-871.e8. [https://doi.org/10.1016/j.molcel.2018.06.044](https://doi.org/10.1016/j.molcel.2018.06.044)

Trimbour,R et al. (2025) CIRCE: a scalable Python package to predict cis-regulatory DNA interactions from single-cell chromatin accessibility data. bioRxiv. 2025.09.23.678054. [https://doi.org/10.1101/2025.09.23.678054](https://doi.org/10.1101/2025.09.23.678054)
