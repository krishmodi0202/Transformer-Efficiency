from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np

# Define custom objects for loading models
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load trained models with exception handling
try:
    predictive_maintenance_model = tf.keras.models.load_model(
        r"D:\Codee\Extra\Transformer\UI\predictive_maintenance_model.h5", custom_objects=custom_objects
    )
    autoencoder_fault_detection = tf.keras.models.load_model(
        r"D:\Codee\Extra\Transformer\UI\autoencoder_fault_detection.h5", custom_objects=custom_objects
    )
    multi_agent_dqn_0 = tf.keras.models.load_model(
        r"D:\Codee\Extra\Transformer\UI\multi_agent_dqn_agent_0.h5", custom_objects=custom_objects
    )
    multi_agent_dqn_1 = tf.keras.models.load_model(
        r"D:\Codee\Extra\Transformer\UI\multi_agent_dqn_agent_1.h5", custom_objects=custom_objects
    )
    multi_agent_dqn_2 = tf.keras.models.load_model(
        r"D:\Codee\Extra\Transformer\UI\multi_agent_dqn_agent_2.h5", custom_objects=custom_objects
    )

    print("\n All models loaded successfully!\n")

    # Print input shapes for debugging
    print(f"Predictive Maintenance Model Input Shape: {predictive_maintenance_model.input_shape}")
    print(f"Fault Detection Model Input Shape: {autoencoder_fault_detection.input_shape}")
    print(f"Multi-Agent RL Model Input Shape: {multi_agent_dqn_0.input_shape}")

except Exception as e:
    print(f"\n Error loading models: {e}\n")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("\n Received JSON Input:", data)

        if not data:
            return jsonify({"error": "No JSON received"}), 400
        
        #  Expected 10 input features
        required_fields = ["voltage", "current", "PFL1", "temperature", "Load", 
                           "humidity", "vibration", "DGA_H2", "frequency", "transformer_age"]
        
        #  Validate required fields
        for field in required_fields:
            if field not in data:
                print(f"\n Missing input parameter: {field}")
                return jsonify({"error": f"Missing input parameter: {field}"}), 400

        #  Convert input to NumPy array (10 features)
        input_features = np.array([
            data["voltage"], data["current"], data["PFL1"], data["temperature"], data["Load"],
            data["humidity"], data["vibration"], data["DGA_H2"], data["frequency"], data["transformer_age"]
        ])

        #  Reshape for Predictive Maintenance Model (10x10)
        predictive_input = np.tile(input_features, (10, 1)).reshape(1, 10, 10)

        #  Reshape for Fault Detection Model (1D, 10 features)
        fault_input = input_features.reshape(1, 10)

        #  Select 3 features for Multi-Agent RL models (example: voltage, Load, frequency)
        rl_input = np.array([data["voltage"], data["Load"], data["frequency"]]).reshape(1, 3)

        print(f"Reshaped Predictive Input Shape: {predictive_input.shape}")
        print(f" Reshaped Fault Detection Input Shape: {fault_input.shape}")
        print(f" Reshaped RL Model Input Shape: {rl_input.shape}")

        # Make predictions
        rul_prediction = predictive_maintenance_model.predict(predictive_input)[0][0]
        anomaly_score = autoencoder_fault_detection.predict(fault_input)[0][0]

        # Multi-agent RL models (for load balancing)
        agent_0_balance = multi_agent_dqn_0.predict(rl_input)[0][0]
        agent_1_balance = multi_agent_dqn_1.predict(rl_input)[0][0]
        agent_2_balance = multi_agent_dqn_2.predict(rl_input)[0][0]

        # Print predictions for debugging
        print("\n Model Predictions:")
        print(f" - Remaining Useful Life (RUL): {rul_prediction}")
        print(f" - Anomaly Score: {anomaly_score}")
        print(f" - Agent 0 Balance: {agent_0_balance}")
        print(f" - Agent 1 Balance: {agent_1_balance}")
        print(f" - Agent 2 Balance: {agent_2_balance}\n")

        # 🔄 Response JSON
        response = {
            "Remaining Useful Life (RUL)": float(rul_prediction),
            "Anomaly Score": float(anomaly_score),
            "Agent 0 Balance": float(agent_0_balance),
            "Agent 1 Balance": float(agent_1_balance),
            "Agent 2 Balance": float(agent_2_balance)
        }


        return jsonify(response)

    except Exception as e:
        print(f"\n Exception Occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n Starting Flask API...")
    app.run(debug=True)
