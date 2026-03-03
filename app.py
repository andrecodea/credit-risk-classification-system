"""Credit Risk Classification - Gradio App"""

import gradio as gr
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from service import predict_and_recommend, get_metrics

# Load test data for showcase
X_test = joblib.load("artifacts/X_test.pkl")
y_test = joblib.load("artifacts/y_test.pkl")
pipeline = joblib.load("artifacts/rf_smote_pipeline.pkl")

# Feature options (must match service.py)
sex_options = ["male", "female"]
housing_options = ["own", "rent", "free"]
saving_options = ["little", "moderate", "quite rich", "rich", "none"]
checking_options = ["little", "moderate", "rich", "none"]
job_options = [0, 1, 2, 3]


def create_roc_plot():
    """Create ROC curve plot"""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Credit Risk Model", fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()


def create_confusion_matrix_plot():
    """Create confusion matrix plot"""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14)
    plt.colorbar()

    classes = ["Good (0)", "Bad (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    return plt.gcf()


# =====================
# GRADIO INTERFACE
# =====================
with gr.Blocks(title="Credit Risk Classification") as demo:
    gr.Markdown("""
    # 🎯 Credit Risk Classification System
    
    **German Credit Risk Dataset** | Random Forest + SMOTE | F1 = 0.84
    
    ---
    This application predicts credit risk using machine learning and provides 
    strategic recommendations via LLM analysis.
    """)

    with gr.Tabs():
        # =====================
        # TAB 1: MODEL SHOWCASE
        # =====================
        with gr.Tab("📊 Model Showcase"):
            gr.Markdown("## Performance Metrics")

            metrics = get_metrics()
            gr.JSON(metrics)

            gr.Markdown("---")
            gr.Markdown("## ROC Curve")
            roc_fig = create_roc_plot()
            gr.Plot(roc_fig)

            gr.Markdown("---")
            gr.Markdown("## Confusion Matrix")
            cm_fig = create_confusion_matrix_plot()
            gr.Plot(cm_fig)

            gr.Markdown("""
            ---
            **Model Performance Summary:**
            - F1 Score (Bad): 0.84
            - ROC AUC: 0.758
            - Recall (Bad): 85.7%
            - Precision (Bad): 82.8%
            """)

        # =====================
        # TAB 2: MAKE PREDICTION
        # =====================
        with gr.Tab("🔮 Make Prediction"):
            gr.Markdown("## Select Applicant Features")
            gr.Markdown(
                "Fill in the customer information below to get a risk prediction and strategic recommendation."
            )

            with gr.Row():
                with gr.Column():
                    age = gr.Slider(18, 80, value=30, step=1, label="Age")
                    sex = gr.Dropdown(sex_options, value="male", label="Sex")
                    job = gr.Dropdown(
                        job_options,
                        value=1,
                        label="Job (0=unskilled, 3=highly skilled)",
                    )

                with gr.Column():
                    housing = gr.Dropdown(housing_options, value="own", label="Housing")
                    savings = gr.Dropdown(
                        saving_options, value="moderate", label="Savings Account"
                    )
                    checking = gr.Dropdown(
                        checking_options, value="little", label="Checking Account"
                    )

            with gr.Row():
                credit_amount = gr.Number(
                    value=5000, label="Credit Amount ($)", minimum=0
                )
                duration = gr.Slider(1, 72, value=24, step=1, label="Duration (months)")

            gr.Markdown("---")

            with gr.Row():
                predict_btn = gr.Button(
                    "Analyze & Get Recommendation", variant="primary", size="lg"
                )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    prediction_output = gr.Label(label="Model Prediction")
                with gr.Column():
                    confidence_output = gr.Textbox(label="Confidence", lines=1)

            gr.Markdown("---")

            # Display LLM recommendation as Markdown
            recommendation_output = gr.Markdown(label="Strategic Recommendation (LLM)")

            # Connect button to prediction function
            predict_btn.click(
                predict_and_recommend,
                inputs=[
                    age,
                    sex,
                    job,
                    housing,
                    savings,
                    checking,
                    credit_amount,
                    duration,
                ],
                outputs=[prediction_output, confidence_output, recommendation_output],
            )

    gr.Markdown("""
    ---
    **© 2024 Credit Risk Classification System**
    
    Built with ❤️ using Gradio, scikit-learn, and OpenAI-compatible LLM
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
