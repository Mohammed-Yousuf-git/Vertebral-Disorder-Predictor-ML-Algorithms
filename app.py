import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# Load and train model
@st.cache_resource
def train_model():
    # Read .dat file
    df = pd.read_csv("column_3C.dat", sep=" ", header=None)
    df.columns = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
                  'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class']
    
    X = df.drop("class", axis=1)
    y = df["class"]
    
    model = RandomForestClassifier(class_weight='balanced',n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X, y

model, X_full, y_full = train_model()
shap.initjs()

st.set_page_config(page_title="Vertebral Column Disorder Predictor", layout="wide")
st.title("🦴 Vertebral Column Disorder Predictor")
st.markdown("This app predicts the spinal condition based on six biomechanical features from X-rays.")

# Autofill example values
example_input = {
    'pelvic_incidence': 60.0,
    'pelvic_tilt': 20.0,
    'lumbar_lordosis_angle': 45.0,
    'sacral_slope': 40.0,
    'pelvic_radius': 120.0,
    'degree_spondylolisthesis': 10.0,
}

with st.sidebar:
    st.header("📝 Input Features")
    autofill = st.checkbox("Use Example Values")

    user_input = {}
    for feature in X_full.columns:
        min_val = float(X_full[feature].min())
        max_val = float(X_full[feature].max())
        default_val = example_input[feature] if autofill else float((min_val + max_val) / 2)

        user_input[feature] = st.slider(
            label=feature.replace("_", " ").capitalize(),
            min_value=min_val,
            max_value=max_val,
            value=default_val,
        )

    uploaded_file = st.file_uploader("📂 Upload a .dat file to predict from", type=["dat"])

# Make prediction
st.subheader("🔮 Prediction")
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file, sep=" ", header=None)
        test_df.columns = X_full.columns.tolist() + ['class']
        test_X = test_df.drop("class", axis=1)
        predictions = model.predict(test_X)

        st.write("✅ Predicted Classes:")
        st.write(predictions)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_X)

        for i in range(min(3, len(test_X))):  # Show SHAP for first 3 rows max
            st.markdown(f"### 🔍 SHAP Explanation for Sample #{i+1}")
            plt.clf()
            shap.bar_plot(shap_values[0][i], feature_names=X_full.columns, max_display=6, show=False)
            st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Class: {prediction}")
    if prediction == "DH":
        st.markdown("**Disc Hernia (DH):** Herniated disc condition affecting the spine.")
    elif prediction == "SL":
        st.markdown("**Spondylolisthesis (SL):** Displacement of vertebra in the spine.")
    else:
        st.markdown("**Normal:** No major spinal abnormalities detected.")

    # SHAP Explanation
    st.subheader("🔍 SHAP Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_full)
        plt.clf()
        shap.bar_plot(shap_values[0][0], feature_names=X_full.columns, max_display=6, show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"Could not display SHAP explanation: {e}")