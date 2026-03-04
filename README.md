# Beyond the Timeline: Mapping the Evolution of Casting Norms and Romantic Age Gaps in Cinema  
**BA820 (Section B1) — Team 10**

## Project Overview
Hollywood films shape cultural ideas about romance, aging, and gendered power. This project investigates whether on-screen romantic age gaps are **random narrative choices** or **structural patterns** that persist across time, directors, and film “packaging.” Our primary dataset is the **Hollywood Age Gap** database (1,155 couples across 830 films since 1935), used to examine how casting norms vary by decade, director, and couple type, and whether disparities concentrate in “headline” pairings within films. :contentReference[oaicite:0]{index=0}

## Why This Matters
Preliminary EDA suggests:
- Age gaps vary substantially by decade with **discrete inflection points** rather than smooth trends. :contentReference[oaicite:1]{index=1}  
- Directors show pronounced heterogeneity, implying potential **persistent creative tendencies** (not just film-specific noise). :contentReference[oaicite:2]{index=2}  
- In multi-couple films, disparities can be unevenly distributed (possible “headline couple” concentration). :contentReference[oaicite:3]{index=3}  
- Mixed-gender couples dominate; same-gender couples are sparse and variable, motivating careful comparisons. :contentReference[oaicite:4]{index=4}  

## Research / Business Questions
We focus on four guiding questions:
1. **Decade shifts:** How has the “typical” romantic age gap shifted across decades (1930s–2020s), and where are the biggest inflection points? :contentReference[oaicite:5]{index=5}  
2. **Director profiles:** Do directors cluster into distinct age-gap “casting profiles,” and how stable are these across their careers? :contentReference[oaicite:6]{index=6}  
3. **Within-film packaging:** In films with multiple couples, do age gaps concentrate in a single “headline” pairing, or recur across couples within the same film? :contentReference[oaicite:7]{index=7}  
4. **Couple-type differences:** How do age-gap distributions differ across couple types (mixed-gender, man–man, woman–woman), and are same-gender pairings closer to parity? :contentReference[oaicite:8]{index=8}  

## Stakeholders
Intended users of these insights include studios, casting directors, creative executives, and DEI teams—especially for monitoring representation risk and accountability in casting decisions. :contentReference[oaicite:9]{index=9}

---

## Data
### Primary dataset: `age_gaps.csv`
Sourced from the **Hollywood Age Gap** website (distributed via TidyTuesday). The dataset is designed to include *actual love interests*, with constraints such as minimum age and exclusion of animated characters. :contentReference[oaicite:10]{index=10}  

**Key columns (data dictionary):** :contentReference[oaicite:11]{index=11}  
- `movie_name`, `release_year`, `director`  
- `age_difference` (in whole years)  
- `couple_number` (identifier when multiple couples exist in a film)  
- actor names, character genders, birthdates, and ages at release (`actor_1_age`, `actor_2_age`, etc.)

> Note: The dataset includes character “gender” labels (often `man`/`woman`) that reflect character identification as submitted, and may not match actor identity. :contentReference[oaicite:12]{index=12}  

### Getting the data
If you’re pulling directly from the TidyTuesday repository (R): :contentReference[oaicite:13]{index=13}  
```r
age_gaps <- readr::read_csv(
  "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-02-14/age_gaps.csv"
)
```
## Methods (High-level)

Our analysis is built around uncovering *structure* (not just averages), including:

- **Temporal regime analysis** (decade-level shifts and inflection points)
- **Director-level clustering / profiling** (persistent casting “signatures”)
- **Within-film inequality packaging** (headline vs. recurring disparity across couples)
- **Couple-type comparisons** (mixed-gender vs same-gender; parity framing)

These choices are motivated by the EDA patterns described in the proposal.

---

## Team

- Parisha Patel  
- Inchara Ashok  
- Shanmathi Shivkumar  
- Bhavya Bavishi  

---

## Generative AI Usage (Course Documentation)

We used ChatGPT as a supportive tool for idea generation, drafting, and clarifying technical/conceptual questions. All AI suggestions were critically evaluated and adapted; final analysis and interpretation reflect the team’s work.
