# AI-Generated Audit Report: ABC HealthCare Payment Process

## 1. Executive Summary

This report leverages data science and AI techniques to audit ABC HealthCare's payment process, focusing on payment time prediction, outlier detection, and anomaly identification. The objective is to identify potential areas of inefficiency, non-compliance, or fraudulent activity.

## 2. Findings from Section 1: Descriptive & Diagnostic Analytics

### A. Regression Analysis for Payment Time Prediction

**Objective:** Predict time to payment using regression modeling.
**Hypothesis:** All invoices take approximately the same time (±1 day) to be paid.

**Results:**
*   **Mean Absolute Error (MAE):** 6.68 days
*   **Mean Squared Error (MSE):** 64.83
*   **Root Mean Squared Error (RMSE):** 8.05 days
*   **R-squared (R2):** -0.19

**Interpretation:**
The regression model developed to predict payment time yielded a Mean Absolute Error (MAE) of 6.68 days. This indicates that, on average, the model's predictions deviate from the actual payment time by more than six days. The R-squared value of -0.19 suggests that the model is performing worse than simply predicting the mean payment time. Therefore, the hypothesis that "all invoices take ~same time ±1 day" is **not supported** by this model. This implies that there are significant variations in payment processing times that are not adequately explained by the current set of features in a linear relationship.

### B. Outlier Detection using Z-scores for Audit Sampling

**Objective:** Detect statistical outliers in payment time for audit sampling.
**Hypothesis:** Invoices are random; there is no fraud pattern.

**Results:**
*   **Number of Outliers Identified (Z-score > 3):** 0

**Interpretation:**
Using a Z-score threshold of 3, no statistical outliers were detected in the 'time to payment' data. This suggests that there are no data points that are extremely far from the mean payment time under the assumption of a normal distribution. While this might indicate a lack of extreme anomalies based on payment time alone, it could also imply that the Z-score method with this threshold is not sensitive enough for the underlying data distribution or that the 'time to payment' feature itself may not be the primary indicator of fraud.

## 3. Findings from Section 2: Machine Learning & AI

### A. Supervised Model to Predict Fraudulent Payments

**Objective:** Build a supervised model to predict fraudulent payments.

**Results:**
*   **Accuracy:** 0.96
*   **Confusion Matrix:**
    ```
    [[134   0]
     [  5   1]]
    ```
*   **Classification Report:**
    ```
                  precision    recall  f1-score   support

               0       0.96      1.00      0.98       134
               1       1.00      0.17      0.29         6

        accuracy                           0.96       140
       macro avg       0.98      0.58      0.63       140
    weighted avg       0.97      0.96      0.95       140
    ```
*   **ROC-AUC Score:** 0.78

**Interpretation:**
The supervised model for fraud detection achieved an accuracy of 0.96. The confusion matrix shows that out of 140 predictions, the model correctly identified 134 non-fraudulent cases (True Negatives) and 1 fraudulent case (True Positive), but missed 5 actual fraudulent cases (False Negatives). The precision for fraud (class 1) is 1.00, meaning when the model predicts fraud, it is always correct. However, the recall for fraud is low at 0.17, indicating that the model only caught 17% of the actual fraudulent cases. The F1-score for fraud is 0.29, which is also low, reflecting the poor recall. The ROC-AUC score of 0.78 suggests a moderately good ability to distinguish between fraudulent and non-fraudulent payments. The model struggles with correctly identifying all fraudulent transactions, likely due to the imbalanced nature of the dataset (very few fraudulent cases). This highlights a need for strategies to address class imbalance during training.

### B. Unsupervised Model to Detect Anomalies (Isolation Forest)

**Objective:** Build an unsupervised model to detect anomalies in payment data.

**Results:**
*   **Number of Anomalies Identified (assuming 5% contamination):** 100

**Top 10 Anomalies (lowest anomaly score):**
```
     Invoice number  Invoice value  time_to_payment  anomaly_score
276       INV-92598        3601.59             19.0      -0.019247
173       INV-59664        3703.89             19.0      -0.016396
915       INV-76994         993.08             17.0      -0.012411
1516      INV-92840        4329.24             19.0      -0.012269
820       INV-20043        4970.75             19.0      -0.011822
181       INV-49273        3414.86             25.0      -0.011220
697       INV-99639        1514.00             22.0      -0.010958
1243      INV-76075        3973.68             19.0      -0.010942
1638      INV-92058        1413.77             19.0      -0.010880
1935      INV-21661        4206.95             29.0      -0.010846
```

**Interpretation:**
The Isolation Forest model identified 100 instances as anomalies within the payment data, assuming a 5% contamination rate. These anomalies represent data points that deviate significantly from the normal patterns observed in the dataset across various features such as 'Invoice value' and 'time_to_payment'. These identified anomalies warrant further investigation as they could indicate unusual payment activities, potential errors, or even fraudulent transactions that are not easily captured by simple outlier rules.

## 4. Audit Insights and Recommendations

### Insights:

1.  **Variable Payment Times:** The significant MAE in the regression analysis indicates that payment times are highly variable and not easily predictable with a simple linear model and current features. This variability could stem from operational inefficiencies, different processing paths for various payment types, or external factors not captured in the data.
2.  **Limited Z-score Effectiveness:** The Z-score method, in this context, did not identify any extreme outliers. This might be due to the nature of the 'time to payment' distribution or the threshold used. It suggests that a more nuanced approach is needed for outlier detection if anomalies are expected in this particular feature.
3.  **Potential Anomalies Detected by Unsupervised Learning:** The Isolation Forest model successfully identified a significant number of anomalies. These instances, characterized by their lower anomaly scores, represent transactions that are statistically unusual compared to the bulk of the payment data. These could be indicative of errors, process deviations, or fraudulent activities.
4.  **Incomplete Fraud Detection:** The supervised fraud detection model successfully identified 134 non-fraudulent cases and 1 fraudulent case, but missed 5 actual fraudulent cases. This highlights a need for strategies to address class imbalance during training.

### Recommendations:

1.  **Deep Dive into Payment Time Variability:**
    *   **Action:** Conduct further exploratory data analysis on features influencing 'time to payment', including categorical breakdowns (e.g., by 'Research team', 'Type of expense', 'Company').
    *   **Goal:** Understand the root causes of payment time variations.
    *   **Benefit:** Identify bottlenecks in the payment process and opportunities for optimization.

2.  **Enhance Outlier Detection Strategy:**
    *   **Action:** Experiment with different Z-score thresholds (e.g., 2.0, 2.5) or implement alternative outlier detection methods like the Interquartile Range (IQR) method for 'time to payment' to cast a wider net for anomalies.
    *   **Goal:** Improve the identification of unusual payment times that might not be captured by a strict Z-score threshold.
    *   **Benefit:** Provide a more comprehensive set of potential anomalies for manual review and audit sampling.

3.  **Investigate Identified Anomalies from Unsupervised Model:**
    *   **Action:** Prioritize the investigation of the 100 anomalies identified by the Isolation Forest model. This involves a manual review of the full transaction details for these 'Invoice numbers' to determine if they represent errors, process deviations, or genuine fraudulent activities.
    *   **Goal:** Validate the effectiveness of the unsupervised model and uncover previously unknown irregular patterns.
    *   **Benefit:** Proactive identification and mitigation of emerging fraud risks or operational issues.

4.  **Reinstate and Improve Supervised Fraud Detection:**
    *   **Action:** Rebuild and execute the supervised fraud detection model using `fraud_cases_master.csv`. Consider additional feature engineering (e.g., creating ratios, interaction terms) and exploring more advanced classification algorithms (e.g., Gradient Boosting, SVMs) and techniques for imbalanced datasets (e.g., SMOTE).
    *   **Goal:** Develop a robust model to predict known fraudulent payment patterns and enhance the ability to flag suspicious transactions automatically.
    *   **Benefit:** Automate the detection of common fraud types, freeing up audit resources for more complex investigations.

5.  **Integrate Findings for Holistic View:**
    *   **Action:** Combine the insights from regression analysis, outlier detection, and anomaly detection to create a holistic view of payment process health.
    *   **Goal:** Provide a comprehensive risk assessment of the payment process.
    *   **Benefit:** Enable data-driven decision-making for audit planning and risk management.

## 5. Next Steps

Based on these findings and recommendations, the immediate next steps should include:
1.  Re-establishing the supervised fraud detection model.
2.  Deeper analysis of the 100 anomalies identified by the Isolation Forest.
3.  Further exploration of factors driving payment time variability.
