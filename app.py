import streamlit as st
import pandas as pd

from main import (
    read_csv,
    drop_columns,
    encode_categorical_columns,
    fill_na_with_mean,
    split_train_test,
    train_regression_models,
    predict_from_user_input
)

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Option to drop columns
    st.subheader("🧹 Column Selection")
    columns_to_drop = st.multiselect("Select columns to drop",options=df.columns.tolist())
    if columns_to_drop:
        df = drop_columns(df, columns_to_drop)
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
        st.dataframe(df.head())

    # Select target
    target = st.selectbox("Select Target Column", df.columns)

    # Identify columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    # Preprocessing
    df, encoders = encode_categorical_columns(df, cat_cols)
    df, mean_values = fill_na_with_mean(df, num_cols)

    # Train-test split
    x_train, x_test, y_train, y_test = split_train_test(df, target)

    # Train models
    st.subheader("📊 Model Training Results")
    results = train_regression_models(x_train, x_test, y_train, y_test)

    for model_name, metrics in results.items():
        st.write(f"### {model_name}")
        st.write(f"**R2 Score:** {metrics['R2']:.4f}")
        st.write(f"**RMSE:** {metrics['RMSE']:.2f}")

    # Choose model
    selected_model = st.selectbox(
        "Select Model for Prediction",
        list(results.keys())
    )

    model = results[selected_model]["model"]
    feature_columns = x_train.columns.tolist()

    # User input section
    st.subheader("🔢 Enter Feature Values")

    user_input = {}
    for col in feature_columns:
        if col in encoders:
            user_input[col] = st.selectbox(col,options=encoders[col].classes_.tolist())
    else:
        user_input[col] = st.number_input(col, value=0.0)

    # Predict button
    if st.button("Predict House Price"):
        prediction = predict_from_user_input(
            model,
            user_input,
            feature_columns,
            encoders,
            mean_values
        )

        st.success(f"🏷 Predicted House Price: {prediction:,.2f}")

else:
    st.info("Please upload a CSV file to start.")
