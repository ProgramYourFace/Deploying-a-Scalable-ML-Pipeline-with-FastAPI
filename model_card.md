# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a scikit-learn random forest classifier with `min_samples_leaf=5` and `random_state=42`.

## Intended Use

The model predicts whether a person earns more than $50K a year from their census information. It is served through a FastAPI app in this project.

## Training Data

The data is `census.csv` from the UCI Census Income dataset (32,561 rows). I used 80% of it for training (26,048 rows), split with stratification on `salary`.

## Evaluation Data

The evaluation data is the remaining 20% of `census.csv` (6,513 rows).

## Metrics

I used precision, recall and F1. On the test set the model has a precision of 0.7852, a recall of 0.6154 and an F1 of 0.6900. The metrics for each categorical slice are in `slice_output.txt`. Some examples are below.

| Slice | Count | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| sex: Male | 4,355 | 0.7732 | 0.6289 | 0.6936 |
| sex: Female | 2,158 | 0.8693 | 0.5429 | 0.6683 |
| race: White | 5,533 | 0.7880 | 0.6220 | 0.6952 |
| race: Black | 662 | 0.8305 | 0.5269 | 0.6447 |
| education: HS-grad | 2,120 | 0.8017 | 0.2743 | 0.4088 |

## Ethical Considerations

The model uses race and sex as features. Recall is lower for women (0.5429) than for men (0.6289), and lower for Black people (0.5269) than for White people (0.6220). This means the model misses high earners more often in these groups.

## Caveats and Recommendations

The data is from 1994, so it may not reflect current incomes. Some slices have very few rows, so their metrics are not reliable.
