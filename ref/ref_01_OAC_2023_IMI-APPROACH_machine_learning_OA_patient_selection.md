# Development and Validation of a Machine Learning-Supported Strategy of Patient Selection for Osteoarthritis Clinical Trials: The IMI-APPROACH Study

**Link**: [https://doi.org/10.1016/j.ocarto.2023.100406](https://doi.org/10.1016/j.ocarto.2023.100406)

This study investigates how machine learning models can enhance patient selection efficiency in clinical trials for knee osteoarthritis (OA), focusing specifically on selecting patients who are likely to experience disease progression. Below is a detailed breakdown of the abstract.

## Research Objectives
The study aims to develop a recruitment strategy using machine learning to select knee OA patients with a higher likelihood of disease progression. The goal is to increase the efficiency of clinical trials evaluating new OA treatments by enriching the patient population with those more likely to exhibit disease progression.

## Study Design
A two-stage recruitment process supported by machine learning models was designed, where each stage employed distinct models:

- **Stage 1 Model**: This model uses existing patient cohort data (historical data) to predict the likelihood of disease progression, selecting patients for screening visits.
- **Stage 2 Model**: This model uses screening visit data to make the final inclusion decisions.

The study utilized data from the prospective **IMI-APPROACH** knee OA study and evaluated the effectiveness of this strategy by assessing actual disease progression over 24 months.

## Results
- Out of 3500 candidate patients, 433 knee OA patients were screened, with 297 eventually enrolled and 247 completing the 2-year follow-up.
- The disease progression was categorized into three groups:
  - **Pain Progression (P)**: 30% of patients.
  - **Structural Progression (S)**: 13% of patients.
  - **Combined Pain and Structural Progression (P + S)**: 5% of patients.
  - **Non-progressors (N)**: 52%, about 15% lower than an unenriched population.

- The model predicted pain progression with an **AUC** of 0.86 (95% CI, 0.81–0.90), demonstrating strong predictive performance. However, the prediction for structural progression was less effective, with an **AUC** of 0.61 (95% CI, 0.52–0.70).
- Progressors were ranked higher than non-progressors in the P + S, P, and S categories, with AUC values of 0.75, 0.71, and 0.57, respectively.

## Conclusions
The machine learning-supported recruitment process successfully enriched the study population with patients more likely to experience disease progression. The strategy performed particularly well in predicting pain-related progression but showed limitations in predicting structural progression. Future research should focus on improving the model's ability to predict structural changes and apply this strategy in interventional trials to assess its value in evaluating treatment outcomes.

---

## Summary
This study highlights the potential of machine learning technology to improve patient selection in clinical trials, specifically in enriching the population with knee osteoarthritis patients more likely to exhibit disease progression. However, predicting structural progression remains challenging, and further optimization of the model and validation in interventional trials are necessary.
