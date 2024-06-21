# Parkinson's Disease Detection

This project aims to build a machine learning model to detect Parkinson's disease using various features derived from voice measurements. The project involves feature selection, data preprocessing, algorithm selection, cross-validation, and handling imbalanced datasets to ensure robust and accurate detection.

## Table of Contents
1. [Feature Selection](#feature-selection)
2. [Data Preprocessing](#data-preprocessing)
3. [Algorithm Selection](#algorithm-selection)
4. [Cross-Validation Techniques](#cross-validation-techniques)
5. [Imbalanced Datasets](#imbalanced-datasets)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Feature Selection

Feature selection is crucial for improving the model's accuracy and reducing overfitting. For Parkinson's disease detection, commonly considered features include:

- **Jitter**: Measures frequency variation.
- **Shimmer**: Measures amplitude variation.
- **HNR (Harmonics-to-Noise Ratio)**: Indicates the ratio of harmonic sound to noise.
- **RPDE (Recurrence Period Density Entropy)**: Measures the entropy of the signal.
- **DFA (Detrended Fluctuation Analysis)**: Indicates the self-similarity of the signal.
- **PPE (Pitch Period Entropy)**: Represents the randomness in the pitch.

These features are selected because they capture the irregularities in voice that are symptomatic of Parkinson's disease, thereby contributing significantly to the model's accuracy.

## Data Preprocessing

Preprocessing is essential to prepare the dataset for modeling. The steps include:

- **Handling Missing Data**: Use imputation techniques like mean, median, or KNN imputation to fill missing values.
- **Outliers Removal**: Detect and remove outliers using statistical methods or domain knowledge to ensure they don't skew the model.
- **Normalization/Standardization**: Scale features to ensure they contribute equally to the model, typically using StandardScaler or MinMaxScaler.
- **Data Quality Assurance**: Verify data integrity and consistency by checking for duplicates and ensuring proper formatting.

## Algorithm Selection

Different machine learning algorithms can be employed for Parkinson's disease detection, including:

- **Logistic Regression**: Simple and interpretable, suitable for binary classification.
- **Support Vector Machines (SVM)**: Effective for high-dimensional spaces, but sensitive to feature scaling.
- **Random Forest**: Robust to overfitting and capable of handling complex data structures.
- **K-Nearest Neighbors (KNN)**: Non-parametric and simple but can be computationally expensive with large datasets.

Factors influencing algorithm choice include:

- **Model Interpretability**: Important for medical applications to understand the decision-making process.
- **Performance on Imbalanced Data**: Algorithms like Random Forest can handle class imbalance better.
- **Scalability and Computational Efficiency**: Necessary for large datasets.

## Cross-Validation Techniques

Cross-validation is vital to evaluate model performance reliably:

- **K-Fold Cross-Validation**: Splits the dataset into `k` subsets, training the model `k` times, each time using a different subset as the test set and the remaining `k-1` subsets as training data.
- **Stratified K-Fold Cross-Validation**: Ensures each fold has a similar distribution of classes, particularly important for imbalanced datasets.

These techniques mitigate overfitting by ensuring the model generalizes well to unseen data, providing a more robust estimate of performance.

## Imbalanced Datasets

Parkinson's disease datasets often have more negative samples than positive ones, leading to class imbalance. Addressing this involves:

- **Resampling Techniques**: 
  - **Oversampling**: Adding more instances of the minority class using techniques like SMOTE (Synthetic Minority Over-sampling Technique).
  - **Undersampling**: Reducing instances of the majority class.
- **Algorithmic Approaches**: 
  - **Class Weight Adjustment**: Modify the algorithm to give more importance to the minority class.
  - **Ensemble Methods**: Combine multiple models to improve minority class predictions.

### Evaluation Metrics
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall (Sensitivity)**: The ratio of true positive predictions to the actual positives.
- **F1 Score**: The harmonic mean of precision and recall, balancing the two.
- **ROC-AUC**: Measures the model's ability to distinguish between classes.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/wissalmezghiche/parkinsons-disease-detection.git
cd parkinsons-disease-detection
pip install -r requirements.txt
```

## Usage

Run the main script to train the model and make predictions:

```bash
python main.py
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

