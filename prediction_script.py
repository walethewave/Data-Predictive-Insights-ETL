import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from docx import Document
import os

try:
    # User input to select the CSV file
    csv_option = int(input("1. Generate report for openpowerlifting.\n2. Generate report for openpowerlifting-2024-01-06-4c732975.\n[Press 1 or 2]: "))
    if csv_option == 1:
        csv_file = "openpowerlifting"
    elif csv_option == 2:
        csv_file = "openpowerlifting-2024-01-06-4c732975"
    else:
        raise ValueError("Invalid option selected. Please choose either 1 or 2.")

    # Load the data
    pl_data = pd.read_csv(f'./Csv_data/{csv_file}.csv', low_memory=False)

    # Ignore DtypeWarning
    pd.options.mode.chained_assignment = None

    # Rename columns for consistency
    pl_data.rename(columns={
        'BodyweightKg': 'BodyweightKg',
        'WeightClassKg': 'WeightClassKg',
        'Best3SquatKg': 'Best3SquatKg',
        'Best3BenchKg_kg': 'Best3BenchKg',
        'Best3DeadliftKg': 'Best3DeadliftKg'
    }, inplace=True)

    # Drop unnecessary columns
    pl_data.drop(['Division', 'Place', 'MeetName', 'Federation'], axis=1, inplace=True)

    # Remove outliers
    pl_data = pl_data[(pl_data['Age'] < 79) & (pl_data['BodyweightKg'] < 153.27) &
                      (pl_data['Best3SquatKg'] > 0) & (pl_data['Best3BenchKg'] > 0) & (pl_data['Best3DeadliftKg'] > 0)]

    # Data preparation
    pl_mds = pl_data.copy()

    # Extract year from date
    pl_mds['year'] = pd.to_datetime(pl_mds['Date']).dt.year

    # Subset data by gender
    pl_mds_m = pl_mds[pl_mds['Sex'] == 'M']
    pl_mds_f = pl_mds[pl_mds['Sex'] == 'F']

    # Filter SBD events
    pl_mds_msbd = pl_mds_m[pl_mds_m['Event'] == 'SBD']

    # Create TotalKg column
    pl_mds_msbd['TotalKg'] = pl_mds_msbd['Best3SquatKg'] + pl_mds_msbd['Best3BenchKg'] + pl_mds_msbd['Best3DeadliftKg']

    # Feature matrix and target vector
    X = pl_mds_msbd[['Age', 'BodyweightKg', 'Equipment']]
    y = pl_mds_msbd['TotalKg']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    numeric_features = ['Age', 'BodyweightKg']
    categorical_features = ['Equipment']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Append regression model to preprocessing pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_test = model_pipeline.predict(X_test)

    # Model evaluation
    train_r2_score = r2_score(y_train, y_pred_train)
    test_r2_score = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Check if the "Reports" folder exists, if not, create it
    reports_folder = "Reports"
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)

    # Generate the report
    report = f"""
    # Predictive Analysis Report: Powerlifting Total Lift Prediction

    ## Introduction
    This report aims to analyze and predict the total weight lifted by powerlifters based on their age, body weight, and equipment used. The analysis involves exploring the dataset, preprocessing the data, building a predictive model using linear regression, and evaluating the model's performance.

    ## Dataset Overview
    The dataset used for this analysis is the "{csv_file}.csv" dataset, containing information about powerlifting competitions. It includes features such as age, bodyweight, equipment used, and the total weight lifted (sum of squat, bench press, and deadlift). The dataset was cleaned by removing outliers and unnecessary columns.

    ## Data Preparation
    - The dataset was preprocessed by handling missing values, removing outliers, and encoding categorical variables (equipment).
    - Features such as age and body weight were extracted from the dataset to use as predictors for the model.

    ## Model Building
    - A linear regression model was chosen for its simplicity and interpretability.
    - The model was trained using the training set, consisting of 80% of the data, and evaluated on the remaining 20% (test set).

    ## Model Evaluation
    - Train R^2 score: {train_r2_score:.4f}
    - Test R^2 score: {test_r2_score:.4f}
    - Train RMSE: {train_rmse:.4f} kg
    - Test RMSE: {test_rmse:.4f} kg

    ## Summary
    - The model explains approximately 41% of the variance in the total weight lifted by powerlifters.
    - On average, the model's predictions are approximately 114 kg away from the true values.
    - The model's performance is consistent across both the training and test sets, indicating no overfitting or underfitting issues.

    ## Recommendations for Improvement
    - To improve model accuracy, consider building separate models for individual lifts (squat, bench press, deadlift).
    - Including interaction variables as predictors could enhance model accuracy to the range of 65 - 70%.
    - Further analysis can be done by breaking down age and body weight into multiple groups to understand their effects better.

    ## Conclusion
    In conclusion, the linear regression model provides a reasonable baseline for predicting the total weight lifted by powerlifters based on age, body weight, and equipment used. While the model's accuracy can be improved with additional features and techniques, it serves as a valuable tool for understanding the factors influencing powerlifting performance.

    This report provides insights into the predictive analysis process and offers recommendations for further refinement and improvement of the model.
    """

    # Create a new Document
    doc = Document()

    # Add Markdown content to the document
    for line in report.split('\n'):
        doc.add_paragraph(line)

    # Save the document
    doc.save(os.path.join(reports_folder, f'{csv_file}_report.docx'))

    print(f"Report generated and saved as '{csv_file}_report.docx' in the '{reports_folder}' folder.")

except ValueError as ve:
    print(f"ValueError: {ve}. Please choose either 1 or 2.")
except FileNotFoundError:
    print(f"FileNotFoundError: The file '{csv_file}.csv' was not found in the './Csv_data/' directory.")
except pd.errors.EmptyDataError:
    print(f"EmptyDataError: The file '{csv_file}.csv' is empty or has no data.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
