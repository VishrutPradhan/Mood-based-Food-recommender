# Mood-Based Food and Restaurant Recommendation System

## Overview
This project is an interactive web application that recommends food and restaurants based on the user's current emotional state. The system uses computer vision and deep learning to detect emotions from facial expressions through a webcam, or allows manual mood selection, and provides personalized food and restaurant recommendations in New Delhi.


### Mood Detection
![Mood Detection](assets/1.png)
*Real-time emotion detection interface using webcam, showing the detected mood of the user*

### Food Recommendations
![Food Recommendations](assets/2.png)
*Personalized food recommendations based on the detected emotional state*

### Restaurant Recommendations
![Restaurant List](assets/3.png)
*Curated list of top restaurants with their cuisines and ratings based on recommended foods*

### Restaurant Locations
![Restaurant Map](assets/4.png)
*Interactive map visualization showing the locations of recommended restaurants in New Delhi*


## Features
- Real-time emotion detection using webcam
- Manual mood selection option
- Food recommendations based on emotional state
- Restaurant recommendations based on recommended foods
- Interactive map visualization of restaurant locations
- Restaurant ratings and cuisine information display

## Technical Stack
- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **Deep Learning**: TensorFlow/Keras
- **Data Analysis**: Pandas, NumPy
- **Mapping**: Folium
- **Data**: CSV files (food_choices.csv, zomato.csv)

## Data Preprocessing

### Emotion Detection
- Grayscale conversion of input images
- Face detection using Haar Cascade Classifier
- Image resizing to 48x48 pixels
- Pixel normalization (0-1 range)
- 5-class emotion classification: Angry, Happy, Neutral, Sad, Stressed

### Restaurant Data (Zomato Dataset)
- Geographic filtering:
  - Filtered for restaurants in New Delhi (Country Code = 1)
  - Removed entries with invalid coordinates (0,0)
  - Filtered latitude values below 29° for accuracy
- Cuisine data type conversion to string
- Rating-based sorting for top recommendations

### Food-Cuisine Mapping
- Implemented mapping between recommended foods and cuisine types:
  - Pizza → pizza
  - Ice Cream → ice cream
  - Chocolate → bakery
  - Pasta → italian
  - Burger → burger


## How It Works
1. **Emotion Detection**:
   - Captures image through webcam
   - Processes image using OpenCV
   - Detects face using Haar Cascade
   - Predicts emotion using pre-trained model

2. **Food Recommendation**:
   - Maps detected emotion to comfort food categories
   - Provides multiple food suggestions per emotion

3. **Restaurant Recommendation**:
   - Filters restaurants based on recommended food
   - Ranks restaurants by aggregate rating
   - Displays top 5 restaurants with details
   - Shows restaurant locations on interactive map


## Data Requirements
- `food_choices.csv`: Contains food preference data
- `zomato.csv`: Restaurant database with locations and ratings
- `emotion_model.h5`: Pre-trained emotion detection model

## Limitations
- Restaurant recommendations are limited to New Delhi
- Requires good lighting conditions for accurate emotion detection
- Limited to 5 emotion categories
- Requires webcam for automatic emotion detection

## Future Improvements
- Expand restaurant database to more cities
- Add more sophisticated food-emotion mapping
- Implement user feedback system
- Add dietary preference filtering
- Enhance emotion detection accuracy
- Add more cuisine type mappings


