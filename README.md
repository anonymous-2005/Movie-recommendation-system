# Movie Recommendation System

This project implements a **content-based movie recommendation system** using Python. The system recommends movies based on their similarity to a given movie, considering attributes such as cast, crew, keywords, and genres. The core logic is built using natural language processing (NLP) and cosine similarity.

---

## Prerequisites

Make sure the following libraries are installed before running the code:

- pandas
- numpy
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

---

## Code Walkthrough

### Data Preparation

1. **String Parsing with ************`literal_eval`************:**

   ```python
   from ast import literal_eval
   features = ["cast", "crew", "keywords", "genres"]
   for feature in features:
       movies_df[feature] = movies_df[feature].apply(literal_eval)
   ```

   This step converts stringified list data in the DataFrame into usable Python objects (e.g., lists and dictionaries).

2. **Extracting Useful Information:**

   - **Director Name:**

     ```python
     def get_director(x):
         for i in x:
             if i["job"] == "Director":
                 return i["name"]
         return np.nan
     movies_df["director"] = movies_df["crew"].apply(get_director)
     ```

     This function extracts the director's name from the `crew` field.

   - **Top 3 Names from Lists:**

     ```python
     def get_list(x):
         if isinstance(x, list):
             names = [i["name"] for i in x]
             if len(names) > 3:
                 names = names[:3]
             return names
         return []
     ```

     This function limits the size of lists (e.g., cast, keywords, genres) to the top 3 items.

3. **Data Cleaning:**

   ```python
   def clean_data(row):
       if isinstance(row, list):
           return [str.lower(i.replace(" ", "")) for i in row]
       elif isinstance(row, str):
           return str.lower(row.replace(" ", ""))
       else:
           return ""
   ```

   Removes spaces and converts all text to lowercase for uniformity.

### Feature Engineering

1. **Creating the "Soup":**

   ```python
   def create_soup(features):
       return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])
   movies_df["soup"] = movies_df.apply(create_soup, axis=1)
   ```

   Combines key features into a single string for each movie.

2. **Vectorization and Similarity Calculation:**

   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.metrics.pairwise import cosine_similarity

   count_vectorizer = CountVectorizer(stop_words="english")
   count_matrix = count_vectorizer.fit_transform(movies_df["soup"])
   similarity = cosine_similarity(count_matrix, count_matrix)
   ```

   Transforms the "soup" into a count matrix and computes cosine similarity between all movies.

### Recommendation Function

1. **Index Mapping:**

   ```python
   indices = pd.Series(movies_df.index, index=movies_df["original_title"]).drop_duplicates()
   ```

   Creates a mapping from movie titles to their indices for easy lookup.

2. **Recommendation Logic:**

   ```python
   def movies_recommendation(title, cosine_similarity):
       idx = indices[title]
       similarity_scores = list(enumerate(cosine_similarity[idx]))
       similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
       similarity_scores = similarity_scores[1:11]  # Exclude the movie itself
       movies_indices = [ind[0] for ind in similarity_scores]
       movies = movies_df["original_title"].iloc[movies_indices]
       return movies
   ```

   Retrieves the top 10 most similar movies to the given title based on cosine similarity.

---

## Usage

1. Prepare the dataset (`movies_df`) with the required fields: `original_title`, `cast`, `crew`, `keywords`, and `genres`.
2. Run the code.
3. Get recommendations by calling the function:
   ```python
   recommended_movies = movies_recommendation("The Avengers", similarity)
   print(recommended_movies)
   ```

---

## Output Examples

- Input: "The Avengers"
- Output: A list of 10 recommended movies similar to "The Avengers."

---

## Key Concepts

1. **Literal Evaluation:** Converts strings into Python objects for structured data.
2. **Feature Engineering:** Combines multiple attributes into a single representation ("soup") for each movie.
3. **Count Vectorization:** Represents text data as numerical feature vectors.
4. **Cosine Similarity:** Measures the similarity between two vectors in a high-dimensional space.

---

## Notes

- Ensure the dataset is preprocessed correctly before running the recommendation system.
- The `movies_recommendation` function assumes that all movie titles in the dataset are unique.
- Customizable features: You can modify the "soup" creation logic to include or exclude specific attributes.

---

## Limitations

- This is a content-based recommendation system. It does not consider user preferences or collaborative filtering.
- Works best with clean and well-structured datasets.

