import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'new_samples')
    model_path = os.path.join(base_dir, 'banana_model.h5')
    new_model_path = os.path.join(base_dir, 'banana_model_v2.h5')

    print(f"Loading existing model from {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Setting up data augmentation...")
    # Data Augmentation: horizontal flips, 20-degree rotation, 0.2 zoom
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # Use 20% of data for validation to see accuracy/loss
    )

    print(f"Loading data from {data_dir}...")
    
    # Assuming the data is structured as data/new_samples/class_name/image.jpg
    # If it's a flat directory, we might need a different approach, but flow_from_directory is standard.
    # The models typically expect (224, 224) or (256, 256). We'll assume (224, 224) based on typical transfer learning (e.g., MobileNet/ResNet).
    # Let's inspect the model input shape first.
    input_shape = model.input_shape[1:3]
    if input_shape == (None, None):
        input_shape = (224, 224) # Fallback
    
    print(f"Model expects input shape: {input_shape}")

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape,
        batch_size=32,
        class_mode='categorical', # Or whatever the original classification mode was
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    print("Fine-tuning model with learning rate 0.0001...")
    # Compile the model with a very low learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', # Or sparse_categorical_crossentropy depending on original definition
                  metrics=['accuracy'])

    # Determine steps
    steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
    validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)

    print("Beginning training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10, # Train for 10 epochs (can be adjusted)
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    print(f"Saving newly trained model to {new_model_path}...")
    model.save(new_model_path)
    
    # Validation results
    print("\n--- Training Results ---")
    final_loss, final_accuracy = model.evaluate(validation_generator)
    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")
    print("Done! Model v2 is ready.")

if __name__ == "__main__":
    train_model()
