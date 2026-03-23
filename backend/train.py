import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(num_classes):
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base layers for transfer learning
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train():
    data_dir = r"c:\Code\banana-disease-app\backend\dataset"
    model_save_path = r"c:\Code\banana-disease-app\backend\banana_model.h5"
    
    # Use ImageDataGenerator for validation split and preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2 # 80/20 split
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )
    
    # Save a mapping of class indices to names
    class_indices = train_generator.class_indices
    class_mapping = {v: k for k, v in class_indices.items()}
    print(f"Class mapping: {class_mapping}")
    
    # Build and train model
    model = build_model(num_classes=len(class_indices))
    
    print("Starting high-speed transfer learning training...")
    model.fit(
        train_generator,
        epochs=3, # Low epochs for quick setup since it's transfer learning
        validation_data=val_generator
    )
    
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    train()
