# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from mtcnn import MTCNN
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

def load_config():
    """Load configuration from yaml file"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def save_uploaded_image(uploaded_image, upload_path):
    """Save the uploaded image to the specified path"""
    try:
        create_directory(upload_path)
        with open(os.path.join(upload_path, uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return False

def get_model():
    """Get the FaceNet model"""
    model = InceptionResnetV1(pretrained='vggface2').eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def extract_features(img_path, model, detector, transform):
    """Extract facial features from the image"""
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face
        results = detector.detect_faces(img)
        if not results:
            raise ValueError("No face detected in the image")

        # Extract face ROI
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        # Convert to PIL Image
        face_image = Image.fromarray(face)
        
        # Transform image
        img_tensor = transform(face_image).unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
        
        return features[0].cpu().numpy()
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def recommend(feature_list, features):
    """Find the most similar face in the database"""
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), 
                                         feature_list[i].reshape(1, -1))[0][0])
    
    index_pos = sorted(list(enumerate(similarity)), 
                      reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

def main():
    st.set_page_config(page_title="Face Matching App", 
                      page_icon="👥",
                      layout="wide")
    
    st.title("Face Matching Application")
    st.write("Upload a photo to find who you look like!")

    # Load configurations
    config = load_config()

    # Setup paths
    artifacts_dir = config['artifacts']['artifacts_dir']
    upload_dir = os.path.join(artifacts_dir, config['artifacts']['upload_image_dir'])
    
    # Initialize model and detector
    model = get_model()
    detector = MTCNN()
    transform = get_transform()

    # Load feature files
    feature_path = os.path.join(
        artifacts_dir,
        config['artifacts']['feature_extraction_dir'],
        config['artifacts']['extracted_features_name']
    )
    filename_path = os.path.join(
        artifacts_dir,
        config['artifacts']['pickle_format_data_dir'],
        config['artifacts']['img_pickle_file_name']
    )

    try:
        feature_list = pickle.load(open(feature_path, 'rb'))
        filenames = pickle.load(open(filename_path, 'rb'))
        st.success("Models and features loaded successfully!")
    except Exception as e:
        st.error(f"Error loading pre-computed features: {str(e)}")
        return

    # File uploader
    uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        if save_uploaded_image(uploaded_image, upload_dir):
            # Display the upload
            display_image = Image.open(uploaded_image)

            with st.spinner("Processing image..."):
                # Extract features
                features = extract_features(
                    os.path.join(upload_dir, uploaded_image.name),
                    model,
                    detector,
                    transform
                )

            if features is not None:
                # Verify feature dimensions
                if features.shape[0] != feature_list[0].shape[0]:
                    st.error(f"Feature dimension mismatch. Expected {feature_list[0].shape[0]}, got {features.shape[0]}")
                    return

                # Get recommendation
                index_pos = recommend(feature_list, features)
                predicted_person = " ".join(os.path.basename(filenames[index_pos]).split('_'))

                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.header('Your uploaded image')
                    st.image(display_image)
                with col2:
                    st.header(f"Looks most like: {predicted_person}")
                    st.image(filenames[index_pos], width=300)

                # Display similarity score
                similarity_score = cosine_similarity(features.reshape(1, -1), 
                                                  feature_list[index_pos].reshape(1, -1))[0][0]
                st.info(f"Similarity score: {similarity_score:.2%}")

if __name__ == "__main__":
    main()