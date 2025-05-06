import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')
import logging
import plotly.graph_objects as go
from fpdf import FPDF
import base64
from prophet import Prophet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False
    logger.warning("xlsxwriter is not installed. Excel export will be disabled.")

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("fpdf is not installed. PDF export will be disabled.")

st.markdown("""
<style>
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #2c3e50;
}
h1, h2, h3 {
    color: #34495e;
}
.section {
    margin: 1em 0;
    padding: 1em;
    border: 1px solid #ecf0f1;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.st-expander {
    background-color: #ffffff;
    border: 1px solid #ecf0f1;
    border-radius: 5px;
}
.stMetric {
    background-color: #ffffff !important;
    padding: 1em;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: #34495e !important;
}
.stMetric label {
    color: #34495e !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #34495e !important;
    font-size: 1.2em;
}
.key-metrics-section {
    background-color: #ffffff !important;
    padding: 1em;
    color: #34495e !important;
}
.key-metrics-section * {
    color: #34495e !important;
}
.logout-button {
    position: fixed;
    bottom: 10px;
    width: 200px;
}
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted #666;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
.input-label {
    font-weight: 600;
    color: #34495e;
    margin-bottom: 0.5em;
}
.input-section {
    padding: 1em;
    border-radius: 5px;
    background-color: #f9fbfc;
}
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = pd.DataFrame()
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("final_merged_synthea_cleaned98.csv")
        if data.empty:
            logger.error("Loaded dataset is empty.")
            st.warning("Loaded dataset is empty.")
            return pd.DataFrame()
        return data
    except FileNotFoundError:
        logger.error("Dataset file 'final_merged_synthea_cleaned98.csv' not found.")
        st.error("Dataset file 'final_merged_synthea_cleaned98.csv' not found. Please check the file paths.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_xgb_model():
    try:
        model = joblib.load("xgb_model_new.pkl")
        st.session_state.model_loaded = True
        logger.info("XGBoost model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error("Model file 'xgb_model_new.pkl' not found.")
        st.error("Model file 'xgb_model_new.pkl' not found. Please check the file paths.")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

if not st.session_state.logged_in:
    st.title("Health Claim Cost Prediction - Login")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    data = load_data()
    xgb_model = load_xgb_model()

    if not data.empty and xgb_model is not None:
        st.session_state.filtered_data = data.copy()

        required_columns = ["AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", 
                           "ENCOUNTERCLASS", "CODE", "TOTAL_CLAIM_COST"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Dataset is missing required columns: {missing_columns}")
        else:
            patient_id_column = next((col for col in ["PATIENTID", "PATIENT_ID", "Id", "ID", "PATIENT"] if col in data.columns), None)
            if patient_id_column:
                data = data.rename(columns={patient_id_column: "PATIENT"})
            else:
                data["PATIENT"] = [f"patient_{i}" for i in range(len(data))]

            data = data.rename(columns={"TOTAL_CLAIM_COST": "TOTALCOST"})
            if "START" in data.columns and "STOP" in data.columns:
                data["ENCOUNTER_DURATION"] = (pd.to_datetime(data["STOP"]) - pd.to_datetime(data["START"])).dt.days.fillna(0)
            else:
                data["ENCOUNTER_DURATION"] = 0

            model_expected_columns = ["PAYER_COVERAGE", "BASE_ENCOUNTER_COST", "AVG_CLAIM_COST", "STATE",
                                     "NUM_DIAG1", "HEALTHCARE_EXPENSES", "NUM_ENCOUNTERS", "NUM_DIAG2", "HEALTHCARE_COVERAGE"]
            for col in model_expected_columns:
                if col not in data.columns:
                    data[col] = 0 if col != "STATE" else "MA"

            data = data.fillna(0)

            def categorize_age(age):
                if pd.isna(age): return "0-17"
                if age < 18: return "0-17"
                elif 18 <= age <= 24: return "18-24"
                elif 25 <= age <= 34: return "25-34"
                elif 35 <= age <= 44: return "35-44"
                elif 45 <= age <= 54: return "45-54"
                elif 55 <= age <= 64: return "55-64"
                return "65+"
            data["AGE_GROUP"] = data["AGE"].apply(categorize_age)

            data["START_YEAR"] = pd.to_datetime(data["START"], errors="coerce").dt.year.fillna(2025)

            features = ["AGE", "GENDER", "RACE", "ETHNICITY", "INCOME", "ENCOUNTERCLASS", "CODE", "ENCOUNTER_DURATION",
                        "PAYER_COVERAGE", "BASE_ENCOUNTER_COST", "AVG_CLAIM_COST", "STATE",
                        "NUM_DIAG1", "HEALTHCARE_EXPENSES", "NUM_ENCOUNTERS", "NUM_DIAG2"]
            X = data[features]
            categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE"]
            for col in categorical_cols:
                X[col] = X[col].astype(str)
            X_encoded = pd.get_dummies(X, columns=categorical_cols)
            model_features = X_encoded.columns.tolist()
            categories = {col: data[col].astype(str).unique() for col in categorical_cols}

        st.session_state.filtered_data = data.copy()

    with st.sidebar:
        st.header("Filters")
        with st.expander("Date Range", expanded=True):
            if "START_YEAR" in data.columns:
                years = sorted(data["START_YEAR"].dropna().unique())
                start_year = st.selectbox("Start Year", years, index=0)
                end_year = st.selectbox("End Year", years, index=len(years)-1)
            else:
                start_year, end_year = 2025, 2025

        def multiselect_with_select_all(label, options):
            select_all = st.checkbox(f"Select All {label}", value=True, key=f"all_{label}")
            return st.multiselect(label, options, default=options if select_all else [], key=label)

        if not data.empty:
            selected_age_groups = multiselect_with_select_all("Age Group", sorted(data["AGE_GROUP"].dropna().unique()))
            selected_genders = multiselect_with_select_all("Gender", sorted(data["GENDER"].dropna().unique()))
            selected_races = multiselect_with_select_all("Race", sorted(data["RACE"].dropna().unique()))
            selected_ethnicities = multiselect_with_select_all("Ethnicity", sorted(data["ETHNICITY"].dropna().unique()))
            selected_encounters = multiselect_with_select_all("Encounter Class", sorted(data["ENCOUNTERCLASS"].dropna().unique()))

            filtered_data = data[
                (data["START_YEAR"] >= start_year) &
                (data["START_YEAR"] <= end_year) &
                (data["AGE_GROUP"].isin(selected_age_groups)) &
                (data["GENDER"].isin(selected_genders)) &
                (data["RACE"].isin(selected_races)) &
                (data["ETHNICITY"].isin(selected_ethnicities)) &
                (data["ENCOUNTERCLASS"].isin(selected_encounters))
            ].copy()
        else:
            filtered_data = pd.DataFrame()

        st.session_state.filtered_data = filtered_data
        if filtered_data.empty:
            st.warning("No data matches the selected filters.")

        st.button("Logout", key="logout", on_click=lambda: setattr(st.session_state, "logged_in", False) or st.rerun())

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Key Metrics", "Claim Forecast", "Data Visualizations", 
        "Resource Allocation", "Cost Prediction", "Data Export"
    ])

    with tab1:
        st.header("Key Metrics")
        st.markdown("<div class='section key-metrics-section'>", unsafe_allow_html=True)
        if not filtered_data.empty:
            required_metrics_cols = ["TOTALCOST", "AGE", "PATIENT", "ENCOUNTER_DURATION", "HEALTHCARE_COVERAGE"]
            missing_cols = [col for col in required_metrics_cols if col not in filtered_data.columns]
            if missing_cols:
                st.warning(f"Missing columns for metrics: {missing_cols}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Claim Cost", f"${filtered_data['TOTALCOST'].mean():,.2f}")
                    st.metric("Total Patients", f"{filtered_data['PATIENT'].nunique():,}")
                with col2:
                    st.metric("Avg Coverage", f"${filtered_data['HEALTHCARE_COVERAGE'].mean():,.2f}")
                    st.metric("Avg Age", f"{filtered_data['AGE'].mean():.1f} years")
                with col3:
                    st.metric("Total Claims", f"{len(filtered_data):,}")
                    st.metric("Avg Duration", f"{filtered_data['ENCOUNTER_DURATION'].mean():.1f} days")
        else:
            st.warning("No filtered data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.header("Claim Forecast")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        if "filtered_data" in st.session_state and not st.session_state.filtered_data.empty:
            df = st.session_state.filtered_data.copy()

            # Clean and prepare the data
            df["START"] = pd.to_datetime(df["START"], errors="coerce").dt.tz_localize(None)
            df.dropna(subset=["START"], inplace=True)
            df = df[df["TOTALCOST"] >= 0]
            df["TOTALCOST"] = pd.to_numeric(df["TOTALCOST"], errors="coerce")
            df = df.sort_values("START")

            df["START_MONTH"] = df["START"].dt.to_period("M").dt.to_timestamp()
            regressors = ['AGE', 'NUM_STATUS1', 'NUM_DIAG1', 'NUM_DIAG2', 
                          'INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']
            cat_vars = ['RACE', 'ETHNICITY', 'GENDER','ENCOUNTERCLASS']

            # Handle numeric and categorical
            df[regressors] = df[regressors].apply(pd.to_numeric, errors='coerce')
            df_encoded = pd.get_dummies(df[cat_vars], drop_first=True)
            df_final = pd.concat([df[['START_MONTH', 'TOTALCOST']], df[regressors], df_encoded], axis=1)

            grouped = df_final.groupby('START_MONTH').agg({
                'TOTALCOST': 'sum',
                **{col: 'mean' for col in regressors + list(df_encoded.columns)}
            }).reset_index()

            grouped.rename(columns={'START_MONTH': 'ds', 'TOTALCOST': 'y'}, inplace=True)
            grouped['ds'] = pd.to_datetime(grouped['ds'], errors='coerce')
            grouped = grouped[grouped['ds'].dt.year >= 2000].drop_duplicates(subset='ds').sort_values('ds')

            grouped['lag_1'] = grouped['y'].shift(1)
            grouped['rolling_mean_3'] = grouped['y'].shift(1).rolling(window=3).mean()
            grouped = grouped.dropna().reset_index(drop=True)

            # Fit Prophet with regressors
            prophet_model = Prophet(interval_width=0.95, mcmc_samples=300)
            for col in grouped.columns:
                if col not in ['ds', 'y']:
                    prophet_model.add_regressor(col)
            prophet_model.fit(grouped)

            import matplotlib.pyplot as plt

            # Forecast until 2030
            last_date = grouped['ds'].max()
            future_end = pd.to_datetime('2030-12-01')
            months_to_forecast = (future_end.year - last_date.year) * 12 + (future_end.month - last_date.month)
            future = prophet_model.make_future_dataframe(periods=months_to_forecast, freq='MS')

            # Add additional regressors if any
            for col in grouped.columns:
                if col not in ['ds', 'y']:
                    future[col] = grouped[col].iloc[-1]  # placeholder: last known value

            # Generate forecast
            forecast = prophet_model.predict(future)

            import plotly.graph_objects as go

            # Filter forecast from 2015 onward
            forecast_recent = forecast[forecast['ds'] >= '2015-01-01']

            # Streamlit header
            st.subheader("Interactive Forecast of Monthly Claim Costs")

            # Create interactive forecast plot
            fig = go.Figure()

            # Add forecast line only
            fig.add_trace(go.Scatter(
                x=forecast_recent['ds'],
                y=forecast_recent['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='blue')
            ))

            # Update layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Predicted Total Claim Cost',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )

            # Display plot
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No filtered data available. Please adjust filters in the sidebar.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.header("Data Visualizations")
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        if not filtered_data.empty and "TOTALCOST" in filtered_data.columns:
            fig = px.histogram(filtered_data, x="TOTALCOST", nbins=20, title="Cost Distribution",
                               labels={"TOTALCOST": "Total Cost ($)"}, color_discrete_sequence=["#3498db"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data or missing 'TOTALCOST' column.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.header("Resource Allocation")
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        if not filtered_data.empty and "AGE_GROUP" in filtered_data.columns and "TOTALCOST" in filtered_data.columns:
            for group_by, title in [("AGE_GROUP", "Age Group"), ("RACE", "Race"), ("ENCOUNTERCLASS", "Encounter Class"),
                                  ("GENDER", "Gender"), ("ETHNICITY", "Ethnicity")]:
                avg_cost = filtered_data.groupby(group_by)["TOTALCOST"].mean().reset_index()
                if not avg_cost.empty:
                    fig = px.bar(avg_cost, x=group_by, y="TOTALCOST", title=f"Avg Cost by {title}",
                                 labels={group_by: title, "TOTALCOST": "Avg Cost ($)"}, color_discrete_sequence=["#3498db"])
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data or missing required columns.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.header("Cost Prediction")
        st.markdown("<div class='section'>", unsafe_allow_html=True)

        st.markdown("""
        ### Predict Healthcare Costs
        Use this tool to estimate claim costs based on patient and encounter details. The prediction model leverages historical data to provide insights.
        """)

        st.subheader("Patient Demographics")
        with st.container():
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("<div class='input-label'>Age (years) <span class='tooltip'>?<span class='tooltiptext'>Enter patient's age (0-100)</span></span></div>", unsafe_allow_html=True)
                age = st.number_input("", min_value=0, max_value=100, value=30, step=1, key="pred_age", label_visibility="hidden")
            with col2:
                st.markdown("<div class='input-label'>Gender <span class='tooltip'>?<span class='tooltiptext'>Select patient's gender</span></span></div>", unsafe_allow_html=True)
                gender = st.selectbox("", ["M", "F"], key="pred_gender", label_visibility="hidden")
            with col3:
                st.markdown("<div class='input-label'>Income ($) <span class='tooltip'>?<span class='tooltiptext'>Annual income in dollars</span></span></div>", unsafe_allow_html=True)
                income = st.slider("", 0, 100000, 50000, key="pred_income", format="%d", label_visibility="hidden")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Encounter Details")
        with st.container():
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("<div class='input-label'>Encounter Class <span class='tooltip'>?<span class='tooltiptext'>Type of medical encounter</span></span></div>", unsafe_allow_html=True)
                encounter_class = st.selectbox("", data["ENCOUNTERCLASS"].dropna().unique(), key="pred_encounter_class", label_visibility="hidden")
            with col2:
                st.markdown("<div class='input-label'>Procedure Code <span class='tooltip'>?<span class='tooltiptext'>Medical procedure code</span></span></div>", unsafe_allow_html=True)
                code = st.selectbox("", data["CODE"].dropna().unique(), key="pred_code", label_visibility="hidden")
            with col3:
                st.markdown("<div class='input-label'>Duration (days) <span class='tooltip'>?<span class='tooltiptext'>Encounter duration in days</span></span></div>", unsafe_allow_html=True)
                encounter_duration = st.slider("", 0, 30, 1, key="pred_encounter_duration", format="%d", label_visibility="hidden")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Clinical Information")
        with st.container():
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("<div class='input-label'>Race <span class='tooltip'>?<span class='tooltiptext'>Patient's racial category</span></span></div>", unsafe_allow_html=True)
                race = st.selectbox("", data["RACE"].dropna().unique(), key="pred_race", label_visibility="hidden")
            with col2:
                st.markdown("<div class='input-label'>Ethnicity <span class='tooltip'>?<span class='tooltiptext'>Patient's ethnicity</span></span></div>", unsafe_allow_html=True)
                ethnicity = st.selectbox("", data["ETHNICITY"].dropna().unique(), key="pred_ethnicity", label_visibility="hidden")
            with col3:
                st.markdown("<div class='input-label'>Chronic Conditions <span class='tooltip'>?<span class='tooltiptext'>Select known chronic conditions</span></span></div>", unsafe_allow_html=True)
                chronic_conditions = st.multiselect("", ["Diabetes", "Hypertension", "Asthma", "Heart Disease"], key="pred_chronic")

            if "DIAGNOSIS1" in data.columns or "DIAGNOSIS2" in data.columns:
                col4, col5 = st.columns([1, 1])
                if "DIAGNOSIS1" in data.columns:
                    with col4:
                        st.markdown("<div class='input-label'>Diagnosis 1 <span class='tooltip'>?<span class='tooltiptext'>Primary diagnosis</span></span></div>", unsafe_allow_html=True)
                        diagnosis1 = st.selectbox("", data["DIAGNOSIS1"].dropna().unique(), key="pred_diagnosis1", label_visibility="hidden")
                else:
                    diagnosis1 = "Not Available"

                if "DIAGNOSIS2" in data.columns:
                    with col5:
                        st.markdown("<div class='input-label'>Diagnosis 2 <span class='tooltip'>?<span class='tooltiptext'>Secondary diagnosis</span></span></div>", unsafe_allow_html=True)
                        diagnosis2 = st.selectbox("", data["DIAGNOSIS2"].dropna().unique(), key="pred_diagnosis2", label_visibility="hidden")
                else:
                    diagnosis2 = "Not Available"
            else:
                diagnosis1 = "Not Available"
                diagnosis2 = "Not Available"

            st.markdown("</div>", unsafe_allow_html=True)

        if any(col in data.columns for col in ["PROVIDERID", "REASONCODE", "CODE_1"]):
            st.subheader("Provider and Medication")
            with st.container():
                st.markdown("<div class='input-section'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 1])

                if "PROVIDERID" in data.columns:
                    with col1:
                        st.markdown("<div class='input-label'>Provider ID <span class='tooltip'>?<span class='tooltiptext'>Healthcare provider ID</span></span></div>", unsafe_allow_html=True)
                        provider_id = st.selectbox("", data["PROVIDERID"].dropna().unique(), key="pred_provider_id", label_visibility="hidden")
                else:
                    provider_id = "Not Available"

                if "REASONCODE" in data.columns:
                    with col2:
                        st.markdown("<div class='input-label'>Reason Code <span class='tooltip'>?<span class='tooltiptext'>Reason for encounter</span></span></div>", unsafe_allow_html=True)
                        reason_code = st.selectbox("", data["REASONCODE"].dropna().unique(), key="pred_reason_code", label_visibility="hidden")
                else:
                    reason_code = "Not Available"

                if "CODE_1" in data.columns:
                    with col3:
                        st.markdown("<div class='input-label'>Medication Code <span class='tooltip'>?<span class='tooltiptext'>Medication code if applicable</span></span></div>", unsafe_allow_html=True)
                        code_1 = st.selectbox("", data["CODE_1"].dropna().unique(), key="pred_code_1", label_visibility="hidden")
                else:
                    code_1 = "Not Available"

                st.markdown("</div>", unsafe_allow_html=True)
        else:
            provider_id = "Not Available"
            reason_code = "Not Available"
            code_1 = "Not Available"

        if st.button("Predict Cost", use_container_width=True, type="primary"):
            chronic_condition = 1 if chronic_conditions or any(kw in str(diagnosis1).lower() or kw in str(diagnosis2).lower() for kw in ["diabetes", "hypertension", "asthma", "heart disease"]) else 0

            input_data = pd.DataFrame({
                "AGE": [age], "GENDER": [gender], "RACE": [race], "ETHNICITY": [ethnicity], "INCOME": [income],
                "ENCOUNTERCLASS": [encounter_class], "CODE": [code], "ENCOUNTER_DURATION": [encounter_duration],
                "PAYER_COVERAGE": [5000], "BASE_ENCOUNTER_COST": [200], "AVG_CLAIM_COST": [filtered_data["TOTALCOST"].mean() if not filtered_data.empty else 1000], "STATE": ["MA"],
                "NUM_DIAG1": [1], "HEALTHCARE_EXPENSES": [300], "NUM_ENCOUNTERS": [2], "NUM_DIAG2": [0],
                "REASONCODE": [reason_code if reason_code != "Not Available" else "General"],
                "CODE_1": [code_1 if code_1 != "Not Available" else "None"],
                "DESCRIPTION": [data["DESCRIPTION"].iloc[0] if "DESCRIPTION" in data.columns else "General Checkup"],
                "PROVIDERID": [provider_id if provider_id != "Not Available" else "General Provider"],
                "DIAGNOSIS1": [str(diagnosis1) if diagnosis1 != "Not Available" else "General Diagnosis"],
                "DIAGNOSIS2": [str(diagnosis2) if diagnosis2 != "Not Available" else "None"],
                "CHRONIC_CONDITION": [chronic_condition]
            })

            categorical_cols = ["GENDER", "RACE", "ETHNICITY", "ENCOUNTERCLASS", "CODE", "STATE", "REASONCODE", "CODE_1",
                               "DESCRIPTION", "PROVIDERID", "DIAGNOSIS1", "DIAGNOSIS2"]
            for col in categorical_cols:
                if col in input_data.columns:
                    input_data[col] = input_data[col].astype(str)

            input_data_encoded = pd.get_dummies(input_data, columns=[col for col in categorical_cols if col in input_data.columns])
            input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)

            try:
                # Check if the model is a Prophet model (which would cause errors in this context)
                if hasattr(xgb_model, 'predict') and not isinstance(xgb_model, Prophet):
                    prediction = xgb_model.predict(input_data_encoded)[0]

                    st.markdown("### Prediction Results")
                    st.metric("Predicted Cost", f"${prediction:.2f}")
                else:
                    st.error("Prediction failed: The loaded model is not suitable for this prediction task. Expected an XGBoost model, but found a different model type.")
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                st.error(f"Prediction failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab6:
        st.header("Data Export")
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        export_formats = ["CSV"]
        if XLSXWRITER_AVAILABLE: export_formats.append("Excel")
        if FPDF_AVAILABLE: export_formats.append("PDF")
        export_format = st.selectbox("Export Format", export_formats)

        if st.button("Preview Data"):
            st.dataframe(st.session_state.filtered_data.head(10))

        if st.button("Export Data"):
            try:
                if export_format == "CSV":
                    buffer = io.StringIO()
                    st.session_state.filtered_data.to_csv(buffer, index=False)
                    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data_export.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                elif export_format == "Excel" and XLSXWRITER_AVAILABLE:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        st.session_state.filtered_data.to_excel(writer, index=False, sheet_name="Data")
                    b64 = base64.b64encode(buffer.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="filtered_data_export.xlsx">Download Excel File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                elif export_format == "PDF" and FPDF_AVAILABLE:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=10)
                    pdf.cell(40, 10, "Filtered Data Export", ln=True, align="C")
                    for column in st.session_state.filtered_data.columns:
                        pdf.cell(40, 10, str(column), border=1)
                    pdf.ln()
                    for index, row in st.session_state.filtered_data.head(10).iterrows():
                        for item in row:
                            pdf.cell(40, 10, str(item)[:20], border=1)
                        pdf.ln()
                    pdf_output = pdf.output(dest="S").encode("latin1")
                    b64 = base64.b64encode(pdf_output).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="filtered_data_export.pdf">Download PDF File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Export error: {e}")
                st.error(f"Export failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)