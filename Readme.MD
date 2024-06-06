# My Project

This is a description of my project step by step.

## Data Source

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

\`\`\`bash
python3 csv_to_postgres.py
\`\`\`
Follow the prompts and choose an option:
- Option 1: Import data from CSV to PostgreSQL.
- Option 2: Exit the script.

2. \`prediction_script.py\`:
   This script enables you to make predictions based on the data stored in the PostgreSQL database.

### Usage:
Run the script using the following command:

\`\`\`bash
python3 prediction_script.py
\`\`\`
Follow the prompts and choose an option:
- Option 1: Make predictions based on the data.
- Option 2: Exit the script.

## Dependencies:
- Python 3.x
- PostgreSQL 16


