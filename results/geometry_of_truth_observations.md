# Geometry of Truth EDA — Observations

## 1. Dataset Structure

The Geometry of Truth cities dataset contains 1,496 statements and 5 original columns: `statement`, `label`, `city`, `country`, and `correct_country`.

During EDA, two additional text-derived features were created: character length and word count.

## 2. Data Quality

No missing values were found in the dataset.

There were no duplicate rows and no duplicate statements.

This indicates that the dataset is clean and suitable for further analysis without requiring major preprocessing.

## 3. Label Balance

The dataset is perfectly balanced between true and false statements.

- True statements: 748
- False statements: 748

Each class therefore represents 50% of the dataset.

This balanced structure is useful for later classification and concept-detection experiments because class imbalance is unlikely to bias the evaluation.

## 4. Text Length

The average statement length was approximately 35 characters and 7.38 words.

Most statements were short, with a median of 7 words.

The average word count was very similar across both labels:

- False statements: approximately 7.39 words
- True statements: approximately 7.36 words

The word-count distributions for true and false statements also strongly overlapped.

This suggests that statement length is unlikely to act as a simple shortcut for distinguishing true statements from false statements.

## 5. Paired True and False Structure

The dataset contains 748 unique cities.

Each city appears exactly twice: once in a true statement and once in a false statement.

For true statements, the stated country matches the correct country.

For false statements, the same city is paired with an incorrect country.

This paired design keeps sentence structure very similar while changing factual correctness, which makes the dataset useful for studying truth-related representations.

## 6. Geographic Coverage

The dataset contains 108 unique correct countries.

However, geographic representation is not evenly distributed.

China contributes 384 cities and India contributes 170 cities.

The top two countries account for approximately 37.03% of all cities, while the top five countries account for approximately 49.33%.

Therefore, although the dataset is balanced by truth label, it is geographically concentrated.

This should be considered when interpreting future truthfulness and SAE feature-detection results.

## 7. Overall Observation

The Geometry of Truth cities dataset is clean, balanced by truth label, and strongly controlled through paired true and false statements.

Its structure makes it suitable for later truthfulness concept-detection experiments in the SAE-Faithful project.

However, the geographic concentration of the dataset should be treated as a limitation when evaluating how broadly the findings may generalize.