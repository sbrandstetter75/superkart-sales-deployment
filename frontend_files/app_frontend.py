import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(page_title="SuperKart Sales Forecaster", layout="wide")

st.title("🛒 SuperKart Sales Forecasting Service")
st.write("Deploys predictions for store-level product sales.")

# Side-by-side tabs for Single vs Batch Prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

# Configure URL targeting the backend container name inside the Docker network
backend_url_single = "http://superkart-backend:7860/v1/predict"
backend_url_batch = "http://superkart-backend:7860/v1/predictbatch"

with tab1:
    st.header("Real-Time Single Product Sales Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=1.0, max_value=50.0, value=12.66)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_area = st.number_input("Product Allocated Area Ratio", min_value=0.001, max_value=1.0, value=0.027, format="%.4f")
        product_mrp = st.number_input("Product MRP", min_value=1.0, max_value=1000.0, value=117.08)
        product_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
        
    with col2:
        store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
        store_city = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        product_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
        store_age = st.number_input("Store Age (Years)", min_value=1, max_value=100, value=16)

    if st.button("Forecast Sales", key="single_predict"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": product_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city,
            "Store_Type": store_type,
            "Product_Id_char": product_char,
            "Store_Age_Years": int(store_age),
            "Product_Type_Category": product_category
        }
        try:
            response = requests.post(backend_url_single, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Product Store Sales Total: ${result['prediction']:.2f}")
            else:
                st.error(f"Backend Error: {response.text}")
        except Exception as e:
            st.error(f"Could not reach prediction service: {e}")

with tab2:
    st.header("Bulk Batch Sales Prediction")
    st.write("Upload a CSV file containing the features to receive model forecasts.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Show preview of uploaded file
        input_df = pd.read_csv(uploaded_file)
        st.write("File Preview:")
        st.dataframe(input_df.head())
        
        if st.button("Run Batch Forecast"):
            try:
                # Reset buffer back to starting byte position
                uploaded_file.seek(0)
                files = {'file': (uploaded_file.name, uploaded_file.read(), 'text/csv')}
                response = requests.post(backend_url_batch, files=files)
                
                if response.status_code == 200:
                    predictions_dict = response.json()
                    predictions_series = pd.Series(predictions_dict).sort_index(key=lambda x: x.astype(int))
                    input_df['Predicted_Product_Store_Sales_Total'] = predictions_series.values
                    
                    st.success("Batch Forecasting Complete!")
                    st.dataframe(input_df)
                    
                    # Allow downloading the result
                    csv_buffer = io.StringIO()
                    input_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download Forecasted Sales CSV",
                        data=csv_buffer.getvalue(),
                        file_name="SuperKart_Sales_Forecasts.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")