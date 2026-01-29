# BA820-B1-Team10

## Project Overview
This repository contains the exploratory analysis and supporting materials for our BA820 project, which investigates structural patterns in two complex, real-world datasets using unsupervised, exploratory techniques. The primary focus is on identifying latent structure, imbalance, and variability that motivate domain-driven analytical questions rather than producing predictive models or final conclusions.

Our primary dataset examines age disparities in on-screen romantic pairings in Hollywood films, while a secondary (backup) dataset explores patterns in U.S. fair use case law. Together, these datasets allow us to study how implicit structure emerges in cultural and legal domains.

---

## Datasets

### Primary Dataset: Hollywood Age Gap
- **Description:** Age differences between romantic leads in Hollywood films  
- **Scope:** 1,155 romantic couples across 830 films (1935–2020s)  
- **Key variables:**  
  - **Numerical:** age gap, actor ages, release year  
  - **Categorical:** director, couple type, film  
- **Domain focus:** Representation, gender, aging, and casting norms in film  

### Secondary Dataset: Fair Use Case Law
- **Description:** U.S. fair use cases with structured metadata and legal text  
- **Key variables:**  
  - **Categorical:** jurisdiction, content type, outcome  
  - **Textual:** issue statements, judicial holdings  
- **Domain focus:** Judicial reasoning patterns, precedent similarity, and legal interpretation  

> Both datasets were provided in pre-cleaned form and did not require additional data cleaning.
---
## Repository Structure
ba820-b1-team10/
- │
- ├── main/ # Final proposal, figures, and written deliverables
-   └── README.md # Project overview and repository guide
- ├── EDA/ # Jupyter notebooks containing exploratory data analysis
- ├── dataset/ # Raw datasets used for analysis (as provided)



---

## Exploratory Data Analysis (EDA)

The EDA focuses on:  
- Distributional patterns and imbalance (e.g., across decades, courts)  
- Structural heterogeneity (e.g., director-level profiles, jurisdictional variation)  
- Variability in textual reasoning and interpretive language  
- Outliers, clustering tendencies, and co-occurrence patterns  

EDA findings directly motivated the formulation of four domain-driven questions per dataset, emphasizing interpretability and real-world relevance.

---

## Reproducibility

- All EDA code is executable and organized in the `EDA/` directory  
- Figures referenced in the proposal are generated from these notebooks  
- No proprietary libraries or external credentials are required to run the analysis  

---

## Team Contributions

- **Parisha Patel:** Project coordination, dataset selection, EDA, synthesis of findings  
- **Bhavya Bavishi:** Dataset feasibility analysis, exploratory visualizations  
- **Inchara Ashok:** Trend analysis, variability assessment, EDA interpretation  
- **Shanmathi Shivkumar:** Domain framing, business question formulation, editing  

---

## Use of Generative AI Tools

ChatGPT was used as a supportive tool for idea generation, drafting assistance, and clarification of conceptual questions. All outputs were critically evaluated, refined, and adapted by the team. Final analyses and conclusions reflect the team’s independent reasoning.

---

## Course Information

- **Course:** BA820 – Unsupervised Learning  
- **Instructor:** [Instructor Name]  
- **Institution:** Boston University – Questrom School of Business
