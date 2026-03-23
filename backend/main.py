from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Banana Disease API")

# Allow local network Expo frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_inference_model():
    """Rebuild the exact V3 BananaGuard architecture for inference.
    This bypasses Keras 3 h5 deserialization issues caused by the baked-in
    mobilenet_v2.preprocess_input (TrueDivide) node in the saved model config.
    Weights are loaded separately by name, which is fully version-compatible.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    # Preprocessing baked in — same as training. Accepts raw 0-255 pixels.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# Load model globally so it's loaded only once when the server starts
model = None
model_error_details = None
try:
    model_path = os.path.join(os.path.dirname(__file__), 'banana_model.h5')
    model = build_inference_model()
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    print(f"Model weights loaded successfully from {model_path}")
except Exception as e:
    model_error_details = {
        "error": str(e),
        "path": os.getcwd(),
        "files_present": os.listdir()
    }
    print(f"Error loading model: {e}")

# Class mapping corresponding to alphabetical folder names: Healthy, Panama Disease, Sigatoka Disease, not detected
CLASS_NAMES = ['Healthy_Leaf', 'Not_Detected', 'Panama_Initial', 'Panama_Critical', 'Yellow_Sigatoka_Initial', 'Yellow_Sigatoka_Critical']

def get_treatment_recommendation(disease_name):
    if disease_name == "Healthy_Leaf":
        return "No treatment needed. Continue regular maintenance."
    elif disease_name == "Not_Detected":
        return "Could not detect a banana leaf. Please ensure the image is clear."
    elif disease_name == "Panama_Initial":
        return "WARNING: Isolate the area. Disinfect tools and monitor closely. Avoid unnecessary watering."
    elif disease_name == "Panama_Critical":
        return "CRITICAL: Total destruction of the affected plant. Do not replant bananas in this soil. Implement strict quarantine."
    elif disease_name == "Yellow_Sigatoka_Initial":
        return "WARNING: Light pruning of affected leaves. Apply mild protective fungicides to prevent spread."
    elif disease_name == "Yellow_Sigatoka_Critical":
        return "CRITICAL: Apply heavy chemical fungicide treatment immediately. Prune and burn all infected tissue. Ensure strict wide spacing."
    else:
        return "No recommendation available."

@app.get("/")
def read_root():
    return {"message": "Banana Disease Detection API is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return model_error_details or {"error": "Model not loaded. Please train the model first."}
        
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess: send raw 0-255 float pixels — the model has mobilenet_v2
        # preprocess_input baked in as TrueDivide(127.5) inside the graph.
        # Normalizing externally would cause double-scaling.
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)   # Raw 0-255, NO division
        img_array = np.expand_dims(img_array, axis=0)   # Add batch dim
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        
        class_name = CLASS_NAMES[predicted_class_idx]
        treatment = get_treatment_recommendation(class_name)
        
        return {
            "class_name": class_name,
            "confidence_percentage": round(confidence, 2),
            "treatment_recommendation": treatment
        }
        
    except Exception as e:
        import traceback
        print(f"Backend Prediction Exception Caught: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/weather-risk")
def get_weather_risk(temp: float = 25.0, humidity: float = 80.0):
    # Abstract simplified logic based on our discussion
    risk_level = "Low"
    message = "Conditions are generally safe."
    
    if humidity > 85.0 and temp > 27.0:
        risk_level = "High"
        message = "High risk of Sigatoka spore germination. Apply protective fungicides."
    elif humidity > 75.0:
        risk_level = "Medium"
        message = "Moderate risk. Monitor crops closely for leaf spots."
        
    return {
        "temperature": temp,
        "humidity": humidity,
        "sigatoka_risk_level": risk_level,
        "recommendation": message
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
