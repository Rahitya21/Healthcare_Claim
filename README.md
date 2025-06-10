#  Health Claim Cost Prediction Application

A Streamlit-based web application designed to forecast and analyze healthcare insurance claim costs using machine learning and time series modeling. This app enables stakeholders to explore patient data, visualize cost trends, simulate scenarios, and make informed financial decisions in the healthcare domain.

>  **Original Repository:** [https://github.com/dhanunjayreddie/healthcare_claim/blob/main/app-2.py]  
>  **Collaborators:** Rahitya Ragi,  Dhanunjay Reddy 

---

##  Application Features

- **Secure Login Interface** for protected access  
- **Key Metrics Dashboard** showing average cost, duration, age, patient count  
- **Claim Forecasting** using Prophet time series model  
- **Scenario-Based Predictions** using XGBoost  
- **Dynamic Filters** by age, gender, ethnicity, and encounter type  
- **Export Capabilities** in CSV, Excel, and PDF formats  

---

## My Contributions

As a co-developer of this project, I was responsible for:

###  Time Series Forecasting with Prophet
- Built the claim forecasting module using Facebook Prophet  
- Engineered features including lagged values and rolling means  
- Integrated regressors (demographic + clinical) to enhance future predictions  
- Delivered forecasts extended through the year 2030  

### XGBoost Scenario Analysis Engine
- Designed and trained the XGBoost model to predict individual claim costs  
- Encoded categorical features and handled missing data programmatically  
- Enabled what-if simulations in the “Cost Prediction” tab using user input fields  

###  Visualization and Metrics Reporting
- Built dynamic dashboards using Plotly Express and Graph Objects  
- Created custom-styled metric cards to display KPIs like average cost, coverage, and patient volume  
- Enabled interactive histograms and bar plots across age groups, races, and encounter types  

---

##  Tech Stack

| Tool               | Purpose                    |
|--------------------|----------------------------|
| **Streamlit**      | UI framework               |
| **Prophet**        | Time series forecasting    |
| **XGBoost**        | Cost prediction modeling   |
| **Pandas, NumPy**  | Data manipulation          |
| **Plotly**         | Visualizations             |
| **Joblib**         | Model serialization        |
| **FPDF / xlsxwriter** | PDF and Excel export    |

---

##  Live Demo
Explore the full application here:  
🔗 [Healthcare Cost Prediction Dashboard](https://healthcareclaim-js5rdd6gswvt8urqkxzcll.streamlit.app/)


---

## 🗂️ Dataset

Used a cleaned and merged dataset generated from Synthea, containing:
- Patient demographics  
- Encounter history  
- Medications
- Claim
- Claim Transactions
- Procedures 

File used: `final_merged_synthea_cleaned98.csv`

---



