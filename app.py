
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Classification",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Classification")
st.write(
    "Machine Learning Classification Models for Loan Approval Prediction"
)

# --------------------------------------------------
# Model files
# --------------------------------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/k-nearest_neighbors.pkl",
    "Gaussian Naive Bayes": "model/gaussian_naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

# --------------------------------------------------
# Model selection
# --------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select a Machine Learning Model",
    list(model_files.keys())
)

# --------------------------------------------------
# Upload test data
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload the test_data.csv file to continue.")

else:

    # Read uploaded CSV
    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.write(f"Rows: {test_data.shape[0]}")
    st.write(f"Columns: {test_data.shape[1]}")
    st.dataframe(test_data.head())

    # Check target column
    if "loan_status" not in test_data.columns:

        st.error(
            "The uploaded CSV must contain the 'loan_status' column."
        )

    else:

        # Separate features and target
        X_test = test_data.drop("loan_status", axis=1)
        y_test = test_data["loan_status"]

        # --------------------------------------------------
        # Load selected model
        # --------------------------------------------------

        model = joblib.load(
            model_files[selected_model]
        )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Metrics for selected model
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        # --------------------------------------------------
        # Display metrics
        # --------------------------------------------------

        st.header(
            f"Evaluation Metrics — {selected_model}"
        )

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{precision:.4f}")

        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.header("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        st.header("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4)
        )

        # --------------------------------------------------
        # Compare all five models
        # --------------------------------------------------

        st.header("Comparison of All Models")

        all_results = []

        for model_name, model_path in model_files.items():

            comparison_model = joblib.load(model_path)

            pred = comparison_model.predict(X_test)
            prob = comparison_model.predict_proba(X_test)[:, 1]

            all_results.append({
                "ML Model": model_name,
                "Accuracy": accuracy_score(y_test, pred),
                "AUC": roc_auc_score(y_test, prob),
                "Precision": precision_score(
                    y_test,
                    pred,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test,
                    pred,
                    zero_division=0
                ),
                "F1": f1_score(
                    y_test,
                    pred,
                    zero_division=0
                ),
                "MCC": matthews_corrcoef(
                    y_test,
                    pred
                )
            })

        comparison_df = pd.DataFrame(all_results)

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True
        )

        # --------------------------------------------------
        # Overall winner
        # --------------------------------------------------

        winner = comparison_df.loc[
            comparison_df["F1"].idxmax(),
            "ML Model"
        ]

        st.success(
            f"Overall Winner based on F1 Score: {winner}"
        )
