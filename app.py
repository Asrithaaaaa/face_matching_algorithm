# app.py
import streamlit as st
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from PIL import Image
import numpy as np
import cv2
from mtcnn import MTCNN
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import yaml

def load_config():
    """Load configuration from yaml file"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_params():
    """Load parameters from yaml file"""
    with open('params.yaml', 'r') as f:
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

def get_model(model_name='resnet50'):
    """Get the VGGFace model with consistent configuration"""
    return VGGFace(
        model=model_name,
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )

def extract_features(img_path, model, detector):
    """Extract facial features from the image"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = detector.detect_faces(img)
        if not results:
            raise ValueError("No face detected in the image")

        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        # Convert to PIL Image and resize
        image = Image.fromarray(face)
        image = image.resize((224, 224))

        # Convert to array and preprocess
        face_array = np.asarray(image)
        face_array = face_array.astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        
        # Extract features
        result = model.predict(preprocessed_img).flatten()
        return result
    
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

def extract_and_save_features(image_paths, model, detector, feature_path):
    """Extract and save features for all images in the dataset"""
    features = []
    valid_paths = []
    
    for path in image_paths:
        try:
            feat = extract_features(path, model, detector)
            if feat is not None:
                features.append(feat)
                valid_paths.append(path)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Save features and valid paths
    with open(feature_path, 'wb') as f:
        pickle.dump(features, f)
    
    return features, valid_paths

def main():
    st.set_page_config(page_title="Face Matching App", 
                      page_icon="👥",
                      layout="wide")
    
    st.title("Face Matching Application")
    st.write("Upload a photo to find who you look like!")

    # Load configurations
    config = load_config()
    params = load_params()

    # Setup paths
    artifacts_dir = config['artifacts']['artifacts_dir']
    upload_dir = os.path.join(artifacts_dir, config['artifacts']['upload_image_dir'])
    
    # Initialize model and detector with consistent configuration
    model = get_model(params['base']['BASE_MODEL'])
    detector = MTCNN()

    # Load or create feature files
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
    except Exception as e:
        st.error(f"Error loading pre-computed features: {str(e)}")
        return

    # File uploader
    uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        if save_uploaded_image(uploaded_image, upload_dir):
            # Display the upload
            display_image = Image.open(uploaded_image)

            # Extract features
            features = extract_features(
                os.path.join(upload_dir, uploaded_image.name),
                model,
                detector
            )

            if features is not None:
                # Verify feature dimensions
                if features.shape[0] != feature_list[0].shape[0]:
                    st.error(f"Feature dimension mismatch. Expected {feature_list[0].shape[0]}, got {features.shape[0]}")
                    return

                # Get recommendation
                index_pos = recommend(feature_list, features)
                predicted_person = " ".join(filenames[index_pos].split('\\')[1].split('_'))

                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.header('Your uploaded image')
                    st.image(display_image)
                with col2:
                    st.header(f"Looks most like: {predicted_person}")
                    st.image(filenames[index_pos], width=300)

if __name__ == "__main__":
    main()