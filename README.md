# DSA 2050A — Week 9: Model selection and defensible evaluation

**Group members name:** Dancan Ratemo & Gregory Achochi  
**Member's student IDs:** Dancan Ratemo (666172); Gregory Achochi (672745)  
**Submission:** One shared GitHub repository link per group  
**Course:** DSA 2050A Data Science Methodology  
**Week:** 9

## Group responsibilities

Each member contributed equally (50% each) across planning, data/model evaluation, interpretation, documentation, and review.

| Group member | Student ID | Work completed | GitHub username or commit reference |
|---|---|---|---|
| Dancan Ratemo | 666172 | Shared equally: planning, analysis, interpretation, documentation, and review (50%) | Billionaire001-del (repository owner) |
| Gregory Achochi | 672745 | Shared equally: planning, analysis, interpretation, documentation, and review (50%) | @4achochi

## Study objective

Use customer information available before cancellation to identify customers at risk of churn. This is a model-selection exercise, not merely a search for the highest score: compare candidates fairly on development data, inspect errors and subgroup performance, document a defensible choice, and evaluate the untouched final test set only once after freezing the choice.

## Data statement

The fictional Kenyan internet provider data are synthetically generated for teaching. They do not represent real customers or actual telecom behaviour.

## Setup and reproduction

1. Open this project folder in VS Code.
2. Create/select a Python environment (Python 3.10–3.13 recommended).
3. From the project root, install dependencies: `python -m pip install -r requirements.txt`.
4. Open `notebooks/Week9_Model_Selection_Lab.ipynb` and select that environment as the kernel.
5. Run cells in order. The notebook includes the completed rationale and saved outputs. Do not rerun the final evaluation or use its result to retune.

## Prediction-time features and excluded fields

The target is `churn` (1 = cancelled, 0 = stayed); each row represents one customer. The candidate prediction-time features are `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, and `customer_segment`, subject to the assumption that the values are recorded before the prediction point. `customer_id` is excluded because it is an identifier, not a meaningful behavioural predictor. `cancellation_confirmed` is excluded because it is a deliberately leaked post-outcome field that would expose the answer. Numeric missing values are median-imputed and scaled; categorical missing values are mode-imputed and one-hot encoded. All preprocessing is inside the model pipeline and refitted within each CV fold.

## Experiments and model selection

The primary metric was F1, declared before comparison. All candidates used the same shuffled, stratified five-fold split (`random_state=42`). The dummy classifier predicted the majority class and achieved mean F1 0.000 (recall 0.000): majority-class accuracy can look reasonable when most customers stay while detecting no churners.

| Model/configuration | Mean F1 | F1 SD | Fold F1 scores | Mean precision | Mean recall |
|---|---:|---:|---|---:|---:|
| Logistic regression, C=0.1 | 0.465 | 0.054 | 0.388, 0.533, 0.454, 0.432, 0.519 | 0.625 | 0.371 |
| Logistic regression, C=1 | 0.470 | 0.053 | 0.391, 0.534, 0.472, 0.434, 0.519 | 0.624 | 0.378 |
| Logistic regression, C=10 | 0.470 | 0.053 | 0.391, 0.534, 0.472, 0.434, 0.519 | 0.624 | 0.378 |
| Decision tree, depth=3 | 0.398 | 0.049 | 0.392, 0.457, 0.316, 0.437, 0.391 | 0.566 | 0.318 |
| Decision tree, depth=5 | 0.382 | 0.023 | 0.376, 0.383, 0.365, 0.362, 0.426 | 0.583 | 0.287 |
| Decision tree, depth=8 | 0.372 | 0.051 | 0.298, 0.439, 0.338, 0.370, 0.417 | 0.495 | 0.300 |

The selected candidate is **logistic regression with C=1** (the `C=10` result ties at the displayed precision and fold scores). It has the strongest observed mean F1, with variability similar to the strongest tree and higher precision/recall. The 0.072 gap over the best tree is more meaningful than a 0.005 gain considered in isolation, though CV fold spread and operational error costs still matter. Logistic regression is comparatively simple to explain and maintain. Its recall remains limited, so the selection is defensible for this teaching comparison, not evidence of deployment readiness. The pre-test decision and rationale are stored in `reports/frozen_selection.json`.

## Error and subgroup analysis

The out-of-fold development confusion matrix uses rows = actual `[stay, churn]` and columns = predicted `[stay, churn]`:

|  | Predicted stay | Predicted churn |
|---|---:|---:|
| Actual stay | 1,605 | 205 |
| Actual churn | 566 | 344 |

There were 566 false negatives (actual churners predicted to stay) and 205 false positives (stayers flagged as churn risks). False negatives may mean missed opportunities for timely retention outreach; false positives may consume staff time or trigger unnecessary offers. The out-of-fold churn recall was 0.378.

| Customer segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---:|---:|---:|---:|---:|
| Established | 2,268 | 649 | 0.216 | 0.593 | 509 |
| New | 452 | 261 | 0.782 | 0.652 | 57 |

Established customers have markedly lower measured recall. This is an important limitation of the aggregate results; subgroup scores are descriptive, and the smaller new-customer sample warrants caution. These results do not by themselves establish why the gap exists.

## Permutation importance

On a held-out slice of development data, the two largest mean F1 reductions after shuffling were `tenure_months` (0.079, SD 0.024) and `support_tickets_90d` (0.035, SD 0.016). This measures predictive reliance for this fitted model and validation sample, not causality: changing a feature is not shown to cause churn to change. Correlated predictors can divide or mask importance, and estimates from one slice and five shuffles are uncertain. Full values are in `reports/permutation_importance.csv`.

## Final test (evaluated once after the frozen selection)

The untouched final test contains 680 rows. The selected pipeline was refit on all development rows and evaluated once:

- **Accuracy:** 0.7000
- **F1:** 0.4486
- **Precision:** 0.5845
- **Recall:** 0.3640
- **Confusion matrix** (actual rows `[stay, churn]`, predicted columns `[stay, churn]`): `[[393, 59], [145, 83]]`

Final-test F1 is about 0.0215 below the development CV mean F1 of 0.4701; recall is also close to, but slightly below, the CV mean recall of 0.378. This modest difference does not indicate a dramatic collapse, but the single held-out test is only one estimate. It must not be reused for tuning or selecting a replacement model. Metrics are saved in `reports/final_metrics.json`.

## Limitations and ethics

The records are synthetic and may not capture real customer behaviour, measurement quality, or population diversity. A random split does not evaluate temporal drift or future-period generalization. The classifier misses many churners overall and has especially low recall for established customers; a single F1 score hides these differences. Before any real use, validate on representative later-period data, investigate subgroup performance and error costs, assess calibration and privacy, and monitor for drift. Use human review and proportionate retention interventions rather than treating scores as certain outcomes.

## Evidence checklist

- [x] Executed notebook with written explanations and visible outputs
- [x] `reports/experiment_log.csv`
- [x] `reports/subgroup_metrics.csv`
- [x] `reports/permutation_importance.csv`
- [x] `reports/frozen_selection.json` created before the final test
- [x] `reports/final_metrics.json` generated in the one ordered run
- [x] Two generated PNG charts
- [x] Complete member names, student IDs, contributions, and commit/co-authorship evidence
- [x] New repository URL recorded in this README

## Final group review

- [x] One group member verified a clean run from start to finish
- [x] Group discussed and approved the written model-selection rationale
- [x] No one used final-test scores to revise the model
- [x] One representative will submit the shared URL to the course LMS

## GitHub repository URL

https://github.com/Billionaire001-del/DSA2050_Week9_GitHub_Lab
