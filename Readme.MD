# My Project

This is a description of my project step by step.
..
## Data Source..

You can download the Powerlifting Database from the following link (Note: the file is large):

[Download the Powerlifting Database here](https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database)


# Data-Predictive-Insights-ETL

## Overview

This project focuses on developing a comprehensive data analytics and reporting application for analyzing the Powerlifting Database from Kaggle. The dataset includes detailed information about powerlifting meets and competitors, covering performance metrics in the Squat, Bench, and Deadlift categories.

## Features

### Selected Data Sources

- **Datasets**: The application uses two primary data sources: "openpowerlifting-2024-01-06-4c732975.csv" and "openpowerlifting.csv".
- **Data Storage**: Data is stored in a Data Warehouse (DWH) format after undergoing necessary staging and transformation steps.
- **Database**: PostgreSQL is used for storing the processed data.

### ETL/ELT Jobs

- **ETL Processes**: ETL jobs are implemented to extract data from the CSV files, transform it to meet reporting requirements, and load it into the PostgreSQL database.
- **Data Transformation**: Steps include data cleansing, normalization, and aggregation to derive meaningful insights.
- **Job Scheduling**: ETL jobs are scheduled to run in batch processing mode at regular intervals to keep the database updated with new data.

### Chosen Technologies

- **Database**: Microsoft SQL is used for storing processed data.
- **ETL Engine**: SSIS (SQL Server Integration Services) is utilized for scalable data processing.
- **Programming Language**: Python is employed for various data manipulation and analysis tasks.
- **Data Visualization**: Microsoft Power BI is utilized for data visualization.
- **Data Analysis/Data Science**: Scikit-learn is used for predictive modeling and analysis.

## Project Components

1. **Data Extraction**: Extracts data from the source CSV files.
2. **Data Transformation**: Cleanses, normalizes, and aggregates the data.
3. **Data Loading**: Loads the transformed data into the PostgreSQL database.
4. **Predictive Modeling**: Utilizes Scikit-learn  for predictive analytics.
5. **Reporting and Visualization**: Creates interactive reports and visualizations using Microsoft Power BI.

# Predictive Analysis Report: Powerlifting Total Lift Prediction

## Introduction
This report focuses on predicting the total weight lifted by powerlifters based on their age, body weight, and equipment used. The goal of this data science project is to build and evaluate a predictive model using linear regression. This process involves data exploration, preprocessing, model training, and performance evaluation.

## Dataset Overview
The dataset utilized in this analysis is from the file `"openpowerlifting.csv"` or `"openpowerlifting-2024-01-06-4c732975.csv"`. This dataset contains various attributes related to powerlifting competitions, including:

- **Age**: The age of the powerlifter.
- **BodyweightKg**: The body weight of the powerlifter in kilograms.
- **Best3SquatKg**: The best squat lift recorded.
- **Best3BenchKg**: The best bench press lift recorded.
- **Best3DeadliftKg**: The best deadlift lift recorded.
- **Equipment**: Type of equipment used (e.g., raw or equipped).

### Data Preparation
- **Data Cleaning**: Removed outliers and unnecessary columns. Columns dropped include 'Division', 'Place', 'MeetName', and 'Federation'.
- **Outlier Removal**: Excluded entries with age above 79 years, body weight over 153.27 kg, or non-positive lift records.
- **Feature Engineering**: Created a new column `TotalKg` which is the sum of `Best3SquatKg`, `Best3BenchKg`, and `Best3DeadliftKg`.
- **Data Splitting**: The dataset was divided into training (80%) and testing (20%) sets.

### Model Building
- **Model Choice**: A Linear Regression model was selected due to its simplicity and interpretability.
- **Preprocessing Pipeline**:
  - **Numeric Features**: Scaled using `StandardScaler`.
  - **Categorical Features**: Encoded using `OneHotEncoder` to handle different equipment types.
- **Training**: The model was trained using the processed training data.

### Model Evaluation
- **Train R^2 Score**: Measures how well the model explains the variance in the training data.
- **Test R^2 Score**: Indicates how well the model generalizes to unseen data.
- **Train RMSE (Root Mean Squared Error)**: Reflects the average prediction error on the training set.
- **Test RMSE**: Shows the average prediction error on the test set.

#### Performance Metrics
- **Train R^2 Score**: 0.4110
- **Test R^2 Score**: 0.4110
- **Train RMSE**: 114.0 kg
- **Test RMSE**: 114.0 kg

### Summary
- The linear regression model explains approximately 41% of the variance in total weight lifted.
- The average prediction error is about 114 kg.
- Consistent performance across training and test sets suggests no overfitting or underfitting.

### Recommendations for Improvement
1. **Model Enhancement**: Develop separate models for individual lifts (squat, bench press, deadlift) for potentially better accuracy.
2. **Feature Engineering**: Consider adding interaction terms or breaking down age and body weight into more granular groups to capture their effects more precisely.
3. **Advanced Techniques**: Explore more complex algorithms such as Random Forests or Gradient Boosting for improved prediction accuracy.

### Conclusion
The linear regression model provides a foundational approach to predicting the total weight lifted by powerlifters based on age, body weight, and equipment. Although the current model offers a baseline, further refinements and additional features could enhance its accuracy and utility.

This analysis is valuable for understanding key factors influencing powerlifting performance and can guide future improvements in predictive modeling within the domain of sports analytics.



## Installation:
# 1. Clone or download the repository to your local machine.
# 2. Ensure you have Python 3.x installed on your system.

## Configuration:
# Before running the scripts, ensure to configure the `config.py` file with your PostgreSQL database credentials:
# ```python
# user = " default username is [postgres] "
# password = " your password here "
# host = " localhost or 127.0.0.1 "
# port = " default port is [5432] "
# ```

## Scripts:
# 1. `csv_to_postgres.py`:
#    This script allows you to import CSV data into your PostgreSQL database.

### Usage:
# Run the script using the following command:

# ```bash
# python3 csv_to_postgres.py
# ```
# Follow the prompts and choose an option:
# - Option 1: Import data from CSV to PostgreSQL.
# - Option 2: Exit the script.

# 2. `prediction_script.py`:
#    This script enables you to make predictions based on the data stored in the PostgreSQL database.

### Usage:
# Run the script using the following command:

# ```bash
# python3 prediction_script.py
# ```
# Follow the prompts and choose an option:
# - Option 1: Make predictions based on the data.
# - Option 2: Exit the script.

## Dependencies:
# - Python 3.x
# - PostgreSQL 16

## License:
# This project is licensed under the MIT License.

# Creating the README.md file
cat <<EOL > README.md
# Data Management and Prediction

## Overview:
This application provides functionalities for managing data stored in a PostgreSQL database and making predictions based on the data using Python scripts.

## Installation:
1. Clone or download the repository to your local machine.
2. Ensure you have Python 3.x installed on your system.

## Configuration:
Before running the scripts, ensure to configure the \`config.py\` file with your PostgreSQL database credentials:
\`\`\`python
user = " default username is [postgres] "
password = " your password here "
host = " localhost or 127.0.0.1 "
port = " default port is [5432] "
\`\`\`

## Scripts:
1. \`csv_to_postgres.py\`:
   This script allows you to import CSV data into your PostgreSQL database.

### Usage:
Run the script using the following command:

bash
python3 csv_to_postgres.py

Follow the prompts and choose an option:
- Option 1: Import data from CSV to PostgreSQL.
- Option 2: Exit the script.

2. \`prediction_script.py\`:
   This script enables you to make predictions based on the data stored in the PostgreSQL database.

### Usage:
Run the script using the following command:

\`\`\`bash
python3 prediction_script.py

Follow the prompts and choose an option:
- Option 1: Make predictions based on the data.
- Option 2: Exit the script.
# Power BI Report

The Power BI report is too large to upload directly to GitHub. You can download it from the following link:

[Download Power BI Report](https://drive.google.com/file/d/1mCELiqU9P7DpTOsy4NNIxFv6Brnrqk_O/view?usp=sharing)

## Dependencies:
- Python 3.x
- PostgreSQL 16


