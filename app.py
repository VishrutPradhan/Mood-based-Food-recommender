import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import folium
from streamlit_folium import folium_static

# Specific file paths
FOOD_CHOICES_PATH = r'C:\Projects\Food Recomender\data\food_choices.csv'
ZOMATO_PATH = r'C:\Projects\Food Recomender\data\zomato.csv'
EMOTION_MODEL_PATH = r'C:\Projects\Food Recomender\model\emotion_model.h5'

# Emotion Detection Function
def detect_emotion_from_frame(frame, emotion_model):
    try:
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If no faces detected, return None
        if len(faces) == 0:
            return None
        
        # Take the first detected face
        (x, y, w, h) = faces[0]
        
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize and preprocess for model
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        
        # Predict emotion
        prediction = emotion_model.predict(roi_gray)
        emotion_index = np.argmax(prediction)
        emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Stressed']
        
        return emotion_labels[emotion_index]
    
    except Exception as e:
        st.error(f"Emotion detection error: {e}")
        return None

# Load data function
def load_data_safely(file_path, encoding='latin-1'):
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}")
        return pd.DataFrame()

# Preprocess restaurant data
def load_and_preprocess_restaurant_data(res_data):
    # Preprocess data
    res_data = res_data.loc[(res_data['Country Code'] == 1) & (res_data['City'] == 'New Delhi'), :]
    res_data = res_data.loc[res_data['Longitude'] != 0, :]
    res_data = res_data.loc[res_data['Latitude'] != 0, :]
    res_data = res_data.loc[res_data['Latitude'] < 29]
    res_data['Cuisines'] = res_data['Cuisines'].astype(str)
    
    return res_data

# Food recommendation function
def recommend_food(mood):
    recommendations = {
        'Happy': ['Ice Cream', 'Pizza'],
        'Sad': ['Chocolate', 'Fried Food'],
        'Stressed': ['Tea', 'Pasta'],
        'Angry': ['Comfort Food', 'Salad'],
        'Neutral': ['Pizza', 'Pasta']
    }
    return recommendations.get(mood, ['Pizza'])

# Restaurant filter
def get_restaurants(food, res_data):
    food_to_cuisine_map = {
        "pizza": "pizza",
        "ice cream": "ice cream",
        "chocolate": "bakery",
        "pasta": "italian",
        "burger": "burger"
    }
    cuisine = food_to_cuisine_map.get(food.lower(), None)
    if cuisine:
        filtered = res_data[res_data['Cuisines'].str.contains(cuisine, case=False)]
        return filtered.sort_values(by='Aggregate rating', ascending=False).head(5)
    return pd.DataFrame()

# Main Streamlit App
def main():
    st.title("Mood-Based Food and Restaurant Recommendation")
    
    # Load emotion detection model
    try:
        emotion_model = load_model(EMOTION_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        return

    # Mood Detection Section
    st.header("Step 1: Detect Your Mood")
    
    # Detection method selection
    detection_method = st.radio("Choose Mood Detection Method", 
                                ["Web Camera", "Manual Selection"])
    
    # Initialize detected_emotion variable
    detected_emotion = None
    
    if detection_method == "Web Camera":
        # Camera input for mood detection
        img_file_buffer = st.camera_input("Take a picture to detect your mood")
        
        if img_file_buffer is not None:
            # Read the image
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Detect emotion
            detected_emotion = detect_emotion_from_frame(cv2_img, emotion_model)
            
            if detected_emotion:
                st.success(f"Detected Mood: {detected_emotion}")
            else:
                st.warning("No face detected. Please try again.")
                # Fallback to manual selection if no face is detected
                detected_emotion = st.selectbox("Select Mood", 
                                                ['Angry', 'Happy', 'Neutral', 'Sad', 'Stressed'])
    else:
        # Manual mood selection
        detected_emotion = st.selectbox("Select Mood", 
                                        ['Angry', 'Happy', 'Neutral', 'Sad', 'Stressed'])
    
    # Ensure an emotion is selected
    if detected_emotion is None:
        st.warning("Please select or detect a mood")
        return

    # Load food choices and restaurant data
    food_data = load_data_safely(FOOD_CHOICES_PATH)
    res_data = load_data_safely(ZOMATO_PATH, encoding='latin-1')
    
    # Preprocess restaurant data
    if not res_data.empty:
        res_data = load_and_preprocess_restaurant_data(res_data)
    else:
        st.error("Failed to load restaurant data")
        return
    
    # Recommendations Section
    st.header("Step 2: Your Recommendations")
    comfort_foods = recommend_food(detected_emotion)
    st.write(f"Based on your mood ({detected_emotion}), we recommend: **{', '.join(comfort_foods)}**")

    # Restaurants Section
    st.header("Step 3: Top Restaurants")
    for food in comfort_foods:
        st.subheader(f"Restaurants for {food}:")
        top_restaurants = get_restaurants(food, res_data)
        
        if not top_restaurants.empty:
            st.dataframe(top_restaurants[['Restaurant Name', 'Cuisines', 'Aggregate rating']])
            
            # Map visualization
            m = folium.Map(location=[28.6139, 77.2090], zoom_start=12)
            for _, row in top_restaurants.iterrows():
                folium.Marker([row['Latitude'], row['Longitude']], 
                              popup=row['Restaurant Name']).add_to(m)
            folium_static(m)
        else:
            st.write(f"No restaurants found for {food}.")

# Run the main application
if __name__ == '__main__':
    main()