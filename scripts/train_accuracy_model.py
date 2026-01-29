import numpy as np
import os
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def get_perfect_landmarks(image_path):
    """Processes a single 'perfect' reference image and returns its landmarks."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load the perfect reference image at {image_path}")
        return None
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        print(f"Error: No pose detected in the perfect reference image at {image_path}")
        return None

def generate_synthetic_data(perfect_landmarks, num_samples=500):
    """
    Creates synthetic data by adding noise to the perfect landmarks.
    The amount of noise determines the accuracy score.
    """
    X = []
    y = []
    
    # Add the perfect sample with a score of 100
    X.append(perfect_landmarks)
    y.append(100.0)
    
    for _ in range(num_samples - 1):
        noise_level = np.random.uniform(0.001, 0.15)
        noisy_landmarks = perfect_landmarks + np.random.normal(0, noise_level, perfect_landmarks.shape)
        accuracy_score = max(0, 100.0 - (noise_level * 700))
        
        X.append(noisy_landmarks)
        y.append(accuracy_score)
        
    return np.array(X), np.array(y)

def train_accuracy_model_for_pose(pose_name, reference_image_path):
    """Trains and saves a regression model for a single pose."""
    print(f"\n--- Training Accuracy Model for: {pose_name.upper()} ---")
    
    perfect_landmarks = get_perfect_landmarks(reference_image_path)
    if perfect_landmarks is None:
        return
        
    print("Generating synthetic training data...")
    X_data, y_data = generate_synthetic_data(perfect_landmarks)
    
    print("Training the regression model...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=5)
    regressor.fit(X_data, y_data)
    
    # --- FIX STARTS HERE ---
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__) 
    # Create a robust path to the output directory
    output_dir = os.path.join(script_dir, '..', 'application')
    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True) 
    
    output_path = os.path.join(output_dir, f'{pose_name}_accuracy_model.pkl')
    # --- FIX ENDS HERE ---
    
    print(f"Saving model to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(regressor, f)
    print("Model saved successfully.")

if __name__ == "__main__":
    pose_references = {
        'downdog': r'C:\Users\donay\Downloads\Yoga_Pose_Project\DATASET\TEST\downdog\00000001.JPG',   # <--- CHANGE THIS
        'goddess': r'C:\Users\donay\Downloads\Yoga_Pose_Project\DATASET\TEST\goddess\00000002.jpeg',   # <--- CHANGE THIS
        'plank':   r'C:\Users\donay\Downloads\Yoga_Pose_Project\DATASET\TEST\plank\00000005.jpg',     # <--- CHANGE THIS
        'tree':    r'C:\Users\donay\Downloads\Yoga_Pose_Project\DATASET\TEST\tree\00000069.jpg',      # <--- CHANGE THIS
        'warrior2':r'C:\Users\donay\Downloads\Yoga_Pose_Project\DATASET\TEST\warrior2\00000003.jpg' # <--- CHANGE THIS
    }
    
    for pose_name, image_path in pose_references.items():
        if not os.path.exists(image_path):
            print(f"\nWARNING: Reference image not found for '{pose_name}' at '{image_path}'. Skipping.")
            continue
        train_accuracy_model_for_pose(pose_name, image_path)