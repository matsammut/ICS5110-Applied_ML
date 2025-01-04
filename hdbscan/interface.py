import gradio as gr
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan

def predict_and_visualize(age, capital_gain, capital_loss, hours_per_week):
    """
    Make prediction and create visualization with context
    """
    # Load the saved model
    model = joblib.load('hdbscan_model.joblib')
    
    # Create input data array
    input_data = np.array([[age, capital_gain, capital_loss, hours_per_week]])
    
    # Make prediction using hdbscan module's function
    cluster, prob = hdbscan.approximate_predict(model, input_data)
    cluster = cluster[0]  # Get the single prediction
    prob = prob[0]  # Get the probability
    
    # Create visualization
    fig = plt.figure(figsize=(10, 6))
    
    # Plot new point
    plt.scatter(age, hours_per_week, c='red', marker='*', s=200, label='Your Data Point')
    
    if cluster == -1:
        result = "Your profile was classified as Noise (confidence: {:.2f})".format(prob)
    else:
        result = f"You belong to Cluster {cluster} (confidence: {prob:.2f})"
        
    plt.title(result)
    plt.xlabel("Age")
    plt.ylabel("Hours per Week")
    plt.legend()
    
    # Add additional information
    additional_info = f"""
    Your input:
    Age: {age}
    Capital Gain: ${capital_gain:,.2f}
    Capital Loss: ${capital_loss:,.2f}
    Hours per Week: {hours_per_week}
    """
    
    return fig, result, additional_info

# Create Gradio interface
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=[
        gr.Slider(minimum=17, maximum=90, label="Age"),
        gr.Number(label="Capital Gain ($)"),
        gr.Number(label="Capital Loss ($)"),
        gr.Slider(minimum=1, maximum=99, label="Hours per Week"),
    ],
    outputs=[
        gr.Plot(label="Visualization"),
        gr.Textbox(label="Cluster Prediction"),
        gr.Textbox(label="Input Summary")
    ],
    title="Income Cluster Predictor",
    description="""
    Enter your demographic information to see which socioeconomic cluster you belong to.
    The visualization will show where your data point falls relative to existing clusters.
    """,
    article="Note: This is a demo version. Predictions are based on historical data patterns."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()