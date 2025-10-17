# DA253053_Assignment_6

This notebook explores different strategies for handling missing data and evaluates their impact on the performance of a Logistic Regression model for predicting credit card default.

The steps performed are as follows:

1.  **Data Loading and Preparation:**
    *   Loaded the `UCI_Credit_Card.csv` dataset into a pandas DataFrame.
    *   Artificially introduced missing values (10%) into the 'AGE' and 'BILL_AMT1' columns to simulate real-world scenarios.

2.  **Missing Data Handling Strategies:**
    *   **Dataset A (Median Imputation):** Created a copy of the dataframe and imputed missing values in each column with the median of that column. Visualized the distributions of 'AGE' and 'BILL_AMT1' after median imputation.
    *   **Dataset B (Linear Regression Imputation):** Created a copy, dropped rows with missing 'BILL_AMT1', and used Linear Regression to predict and fill missing 'AGE' values based on other numerical features.
    *   **Dataset C (Decision Tree Imputation):** Created a copy, dropped rows with missing 'BILL_AMT1', and used Decision Tree Regression to predict and fill missing 'AGE' values based on other numerical features.
    *   **Dataset D (Listwise Deletion):** Created a copy of the original dataframe and dropped all rows containing any missing values.

3.  **Model Training and Evaluation:**
    *   For each of the four datasets (`df_a`, `df_b`, `df_c`, and `df_d`), the data was split into training and testing sets (80/20 split).
    *   Features in each dataset were standardized using `StandardScaler`.
    *   A Logistic Regression classifier with balanced class weights was trained on the training set of each dataset, with 'default.payment.next.month' as the target variable.
    *   The performance of each model was evaluated on its respective test set using a classification report, including metrics like Accuracy, Precision, Recall, and F1-score for both classes.

4.  **Performance Comparison:**
    *   A summary table was created to compare the key performance metrics (Accuracy, Precision, Recall, and F1-score for Class 1 - default) across the four models.

5.  **Discussion of Findings:**
    *   Discussed the trade-offs between Listwise Deletion and Imputation methods.
    *   Analyzed the performance comparison, noting that median imputation resulted in the highest F1-score for the minority class, although the differences between imputation methods and listwise deletion were not substantial in this case.
    *   Discussed the performance difference between Linear Regression and Decision Tree imputation, suggesting potential non-linear relationships if Decision Tree performed slightly better (though the difference was minimal).

The analysis suggests that while imputation methods retain more data, the impact on model performance compared to listwise deletion was limited in this specific scenario and with the chosen model. Median imputation showed a slight edge in predicting the minority class. Further steps could involve exploring different models, addressing class imbalance, or trying more advanced imputation techniques.
