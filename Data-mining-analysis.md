# Data Mining and Preprocessing Analysis

## Data Preprocessing Steps

### 1. Restaurant Data (Zomato Dataset)
#### Geographic Filtering
```python
res_data = res_data.loc[(res_data['Country Code'] == 1) & (res_data['City'] == 'New Delhi'), :]
```
- Filtered restaurants specifically for New Delhi (Country Code = 1)
- Removed invalid coordinate entries:
  - Eliminated zero longitude/latitude values
  - Removed entries with latitude above 29° (invalid for Delhi region)
- Excluded unrated restaurants
- Standardized cuisine data type to string

#### Feature Engineering
- Created numerical rating categories mapping:
  ```python
  rating_map = {
      'Not rated': -1, 
      'Poor': 0, 
      'Average': 2, 
      'Good': 3, 
      'Very Good': 4, 
      'Excellent': 5
  }
  ```

### 2. Food Choice Data
- Handled missing values by filling with empty strings
- Text preprocessing for comfort food analysis:
  - Removed stopwords
  - Applied lemmatization to standardize terms
  - Created food-to-cuisine mapping dictionary

## Data Mining Techniques Implemented

### 1. Clustering Analysis
- **K-Means Clustering**
  - Applied to restaurant locations (Longitude, Latitude)
  - Used 7 clusters to segment restaurants geographically
  - Implementation:
    ```python
    kmeans = KMeans(n_clusters=7, random_state=0).fit(res_data[['Longitude', 'Latitude']])
    ```

### 2. Text Mining
- **Cuisine Analysis**
  - Extracted and counted unique cuisines
  - Created frequency distribution of cuisines
  - Identified top 10 most common cuisines

### 3. Pattern Recognition
- **Comfort Food Analysis**
  ```python
  def search_comfort(mood):
      foodcount = {}
      for i in range(len(food_data)):
          reasons = str(food_data['comfort_food_reasons'][i]).lower()
          reasons = [word for word in reasons.split() if word not in stop_words]
          if mood in reasons:
              foods = str(food_data['comfort_food'][i]).lower()
              foods = [lemmatizer.lemmatize(food) for food in foods.split(',')]
              for food in foods:
                  foodcount[food] = foodcount.get(food, 0) + 1
      return sorted(foodcount, key=foodcount.get, reverse=True)
  ```
- Implements mood-based pattern recognition:
  - Analyzes comfort food reasons
  - Maps moods to food preferences
  - Ranks foods based on frequency

### 4. Association Analysis
- **Food-Cuisine Mapping**
  - Created associations between comfort foods and cuisine types
  - Used for restaurant recommendations
  - Example mappings:
    ```python
    food_to_cuisine_map = {
        "pizza": "pizza",
        "ice cream": "ice cream",
        "chicken wings": "mughlai",
        "chinese": "chinese",
        "chocolate": "bakery"
    }
    ```

## Visualization Techniques

### 1. Geographic Clustering Visualization
- Scatter plot of restaurant clusters
- Color-coded by cluster assignment
- Shows spatial distribution of restaurants

### 2. Cuisine Distribution Analysis
- Bar plot of top 10 cuisines
- Visualizes cuisine frequency distribution
- Uses seaborn for enhanced visualization

## Recommendation System Implementation

### 1. Mood-Based Food Recommendation
- Uses text mining on comfort food reasons
- Implements frequency-based ranking
- Returns top 3 recommended foods

### 2. Restaurant Recommendation
- Filters restaurants based on cuisine mapping
- Ranks by aggregate rating
- Returns top 3 restaurants for each recommended food

## Data Mining Outcomes

1. **Geographic Insights**
   - Restaurant distribution patterns in Delhi
   - Cluster-based neighborhood analysis

2. **Cuisine Patterns**
   - Popular cuisine types
   - Cuisine availability by area

3. **Mood-Food Correlations**
   - Common comfort foods for different moods
   - Food preference patterns

4. **Restaurant Rankings**
   - Rating-based restaurant classification
   - Cuisine-specific top performers
