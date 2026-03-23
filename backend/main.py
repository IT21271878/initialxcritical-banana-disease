from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import keras
from pydantic import BaseModel

@tf.keras.utils.register_keras_serializable()
class TrueDivide(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # Pop 'name' and other arguments to satisfy the base Layer constructor
        kwargs.pop('name', None)
        super().__init__(**kwargs)
    def call(self, x):
        return x
import os
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

# Load model globally so it's loaded only once when the server starts
model = None
model_error_details = None
try:
    model_path = os.path.join(os.path.dirname(__file__), 'banana_model.h5')
    custom_objects = {'TrueDivide': TrueDivide}
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects=custom_objects, 
        compile=False, 
        safe_mode=False
    )
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    import os
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
        
        # Preprocess for MobileNetV2
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        img_array = img_array / 255.0 # Rescale like in training
        
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
