import streamlit as st
import pandas as pd
import nltk

# Ensure necessary NLTK data is downloaded
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    st.stop()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords
stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

# Load datasets
try:
    food_data = pd.read_csv(r'./food_choices.csv')
    res_data = pd.read_csv(r'./zomato.csv', encoding='latin-1')
    
    # Check for required columns
    required_columns = ['Country Code', 'City', 'Longitude', 'Latitude', 'Rating text', 'Cuisines']
    if not all(col in res_data.columns for col in required_columns):
        st.error("Zomato dataset missing required columns.")
        st.stop()

    # Filter and clean restaurant data
    res_data = res_data.loc[(res_data['Country Code'] == 1) & (res_data['City'] == 'New Delhi')]
    res_data = res_data.loc[(res_data['Longitude'] != 0) & (res_data['Latitude'] != 0)]
    res_data = res_data.loc[res_data['Latitude'] < 29]
    res_data = res_data.loc[res_data['Rating text'] != 'Not rated']
    res_data['Cuisines'] = res_data['Cuisines'].fillna('').astype(str)
except FileNotFoundError as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Functions
def search_comfort(mood):
    lemmatizer = WordNetLemmatizer()
    foodcount = {}
    for i in range(len(food_data)):
        reasons = [
            temps.strip().replace('.', '').replace(',', '').lower()
            for temps in str(food_data["comfort_food_reasons"][i]).split(' ')
            if temps.strip() not in stop
        ]
        if mood in reasons:
            food_items = [
                lemmatizer.lemmatize(temps.strip().replace('.', '').replace(',', '').lower())
                for temps in str(food_data["comfort_food"][i]).split(',')
                if temps.strip() not in stop
            ]
            for item in food_items:
                foodcount[item] = foodcount.get(item, 0) + 1
    sorted_food = sorted(foodcount, key=foodcount.get, reverse=True)
    return sorted_food

def find_my_comfort_food(mood):
    topn = search_comfort(mood)
    return topn[:3]

# Streamlit App Configuration
st.set_page_config(page_title="Mood-Based Food Recommender", page_icon="🍕", layout="wide")

# Title and Description
st.title("🍽️ Mood-Based Food Recommender 🍕")
st.write("Welcome to the **Mood-Based Food Recommender**! Select your mood, and we'll suggest comfort foods and the best restaurants in New Delhi for you to enjoy. 🌟")

# Moods and Emojis Mapping
emoji_mood_mapping = {
    "😊 Happy": "happy",
    "😔 Sad": "sad",
    "😠 Angry": "angry",
    "😴 Tired": "tired",
    "🤩 Excited": "excited",
    "🤢 Disgusted": "disgusted"
}

# Precompute available moods
available_moods = [emoji for emoji, mood in emoji_mood_mapping.items() if search_comfort(mood)]

if available_moods:
    mood = st.radio("How are you feeling today? Select an emoji that matches your mood:", available_moods, index=0, key="mood_radio")
    mood_text = emoji_mood_mapping.get(mood)

    # Recommendations
    result = find_my_comfort_food(mood_text)
    if result and len(result) >= 3:
        st.subheader(f"🍴 Comfort Food Recommendations for Your Mood: {mood}")
        st.markdown(f"Try **{result[0]}**, **{result[1]}**, or **{result[2]}**!")

        # Food to Cuisine Mapping
        food_to_cuisine_map = {
            "pizza": "pizza",
            "ice cream": "ice cream",
            "chicken wings": "mughlai",
            "chinese": "chinese",
            "chip": "bakery",
            "chocolate": "bakery",
            "burger": "burger",
            "pasta": "italian",
        }

        # Restaurants
        restaurants_list = []
        for item in result:
            cuisine = food_to_cuisine_map.get(item)
            if cuisine:
                restaurants = res_data[res_data.Cuisines.str.contains(cuisine, case=False)].sort_values(by='Aggregate rating', ascending=False).head(3)
                restaurants_list.extend(restaurants.to_dict('records'))

        if restaurants_list:
            st.subheader("🍽️ Top Restaurant Recommendations:")
            for idx, restaurant in enumerate(restaurants_list):
                st.markdown(f"### **Restaurant {idx + 1}: {restaurant['Restaurant Name']}**")
                st.write(f"**Cuisine:** {restaurant['Cuisines']}")
                st.write(f"**Rating:** {restaurant['Aggregate rating']}")
                st.write(f"**Address:** {restaurant['Address']}")
                st.write("---")
    else:
        st.error("Sorry, not enough data for the selected mood. Try another!")
else:
    st.error("No moods are available with sufficient data.")
