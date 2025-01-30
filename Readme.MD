# **Data Predictive Insights: ETL & Predictive Analytics**

## **Overview**
This project develops a **data analytics and reporting application** for analyzing the **Powerlifting Database** from Kaggle. It leverages **ETL (Extract, Transform, Load)** processes to transform raw data into actionable insights, stored in a **PostgreSQL database**. The application also includes **predictive modeling** using **Scikit-learn** and **data visualization** with **Microsoft Power BI**.

---

## **Key Features**
1. **ETL Pipeline**:
   - Extracts data from CSV files.
   - Transforms data through cleansing, normalization, and aggregation.
   - Loads processed data into a PostgreSQL database.
2. **Predictive Analytics**:
   - Predicts total weight lifted by powerlifters using **Linear Regression**.
   - Evaluates model performance using metrics like **R² Score** and **RMSE**.
3. **Data Visualization**:
   - Creates interactive reports and dashboards using **Microsoft Power BI**.
4. **Scalable Architecture**:
   - Utilizes **SSIS (SQL Server Integration Services)** for scalable data processing.
   - Implements batch processing for regular data updates.

---

## **Project Components**
### **1. Data Extraction**
- **Source**: CSV files (`openpowerlifting.csv` and `openpowerlifting-2024-01-06-4c732975.csv`).
- **Process**: Data is extracted and prepared for transformation.

### **2. Data Transformation**
- **Cleansing**: Removes outliers and unnecessary columns (e.g., `Division`, `Place`).
- **Normalization**: Scales numeric features and encodes categorical variables.
- **Aggregation**: Creates new features like `TotalKg` (sum of `Best3SquatKg`, `Best3BenchKg`, and `Best3DeadliftKg`).

### **3. Data Loading**
- **Storage**: Transformed data is loaded into a **PostgreSQL database**.
- **Schema**: Designed for efficient querying and reporting.

### **4. Predictive Modeling**
- **Model**: Linear Regression.
- **Features**: Age, body weight, and equipment type.
- **Evaluation**:
  - **R² Score**: 0.411 (explains 41.1% of variance).
  - **RMSE**: 114 kg (average prediction error).

### **5. Reporting & Visualization**
- **Tool**: Microsoft Power BI.
- **Output**: Interactive dashboards for exploring powerlifting data and insights.

---

## **Technology Stack**
- **Database**: PostgreSQL, Microsoft SQL.
- **ETL Engine**: SSIS (SQL Server Integration Services).
- **Programming Language**: Python.
- **Libraries**: Pandas, Scikit-learn, NumPy.
- **Visualization**: Microsoft Power BI.

---

## **Key Insights**
1. **Predictive Model**:
   - The model explains **41.1% of the variance** in total weight lifted.
   - Average prediction error is **114 kg**.
2. **Data-Driven Decisions**:
   - Insights can guide **training programs**, **equipment selection**, and **competition strategies** for powerlifters.
3. **Scalability**:
   - The ETL pipeline and predictive model are designed to handle large datasets and regular updates.

---

## **How to Use This Repository**
### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/walethewave/Data-Predictive-Insights-ETL.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Configuration**
Update the `config.py` file with your PostgreSQL credentials:
```python
user = "postgres"  # Default username
password = "your_password_here"
host = "localhost"  # or 127.0.0.1
port = "5432"  # Default port
```

### **Scripts**
1. **Import Data**:
   - Run `csv_to_postgres.py` to load CSV data into PostgreSQL:
     ```bash
     python3 csv_to_postgres.py
     ```
   - Follow the prompts to import data or exit the script.

2. **Make Predictions**:
   - Run `prediction_script.py` to generate predictions:
     ```bash
     python3 prediction_script.py
     ```
   - Follow the prompts to make predictions or exit the script.

---



---

## **Power BI Report**
The Power BI report is too large to upload directly to GitHub. You can download it from the following link:

[Download Power BI Report](#)

---

## **Contributing**
Feel free to contribute to this project by:
- Reporting issues
- Suggesting improvements
- Submitting pull requests

---

## **Contact**
For questions or collaborations, reach out to me at Lekanolawale477@gmail.com or connect with me on [LinkedIn](https://www.linkedin.com/in/olawale-goodluck-691114255/).

---

