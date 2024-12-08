from tkinter import END, FLAT, WORD, Button, Entry, Label, Text, Tk, Frame, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from difflib import get_close_matches

# Load the movies data
try:
    movies = pd.read_csv(r'E:\Movies Recommendation System Semester\movies.csv')
except FileNotFoundError:
    messagebox.showerror("File Error", "The file 'movies.csv' was not found.")
    exit()

# Check if required columns are present
required_columns = ['title', 'genres']
if not all(column in movies.columns for column in required_columns):
    messagebox.showerror("Data Error", "Required columns are missing in the dataset.")
    exit()

# Preprocess: Convert genres to lowercase and fill NaN values
movies['genres'] = movies['genres'].fillna('').str.lower()
movies['title_lower'] = movies['title'].str.lower()  # Create a lowercase column for case-insensitive matching

# Use TF-IDF Vectorizer to convert genres into vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Reduce dimensions with Truncated SVD
n_features = tfidf_matrix.shape[1]
n_components = min(20, n_features)  # Use 20 components or fewer if features are limited
svd = TruncatedSVD(n_components=n_components)
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Function to recommend movies with reasoning
def recommend_movies(title, tfidf_matrix_reduced=tfidf_matrix_reduced, movies=movies):
    title = title.strip().lower()
    if title not in movies['title_lower'].values:
        # Suggest closest matches
        similar_titles = get_close_matches(title, movies['title_lower'].tolist(), n=3, cutoff=0.6)
        return [], similar_titles
    idx = movies[movies['title_lower'] == title].index[0]
    # Compute cosine similarity
    cosine_similarities = linear_kernel(tfidf_matrix_reduced[idx:idx+1], tfidf_matrix_reduced).flatten()
    similar_indices = cosine_similarities.argsort()[::-1][1:11]  # Top 10 recommendations
    
    # Add reasoning
    recommendations = []
    reasons = []
    input_genres = set(movies['genres'].iloc[idx].split('|'))
    for i in similar_indices:
        recommended_title = movies['title'].iloc[i]
        recommended_genres = set(movies['genres'].iloc[i].split('|'))
        common_genres = input_genres.intersection(recommended_genres)
        recommendations.append(recommended_title)
        reasons.append(f"Common genres: {', '.join(common_genres)}")
    return recommendations, reasons

# Function to show recommendations
def show_recommendations():
    title = entry.get().strip()
    if not title:
        messagebox.showwarning("Input Error", "Please enter a movie title.")
        return
    recommendations, reasons_or_suggestions = recommend_movies(title)
    recommendations_text.delete(1.0, END)  # Clear previous recommendations
    if not recommendations:
        if reasons_or_suggestions:  # Display suggestions if no exact match
            recommendations_text.insert(END, f"No exact match found for '{title}'.\nDid you mean:\n", 'header')
            for suggestion in reasons_or_suggestions:
                recommendations_text.insert(END, f"• {suggestion}\n", 'normal')
        else:
            recommendations_text.insert(END, f"No recommendations found for '{title}'.", 'bold')
    else:
        recommendations_text.insert(END, f"Recommendations for '{title}':\n\n", 'header')
        for rec, reason in zip(recommendations, reasons_or_suggestions):
            recommendations_text.insert(END, f"• {rec}\n{reason}\n\n", 'normal')

# Create the main window
root = Tk()
root.title("Professional Movie Recommendation System")

# Configure window to maximize
root.state('zoomed')  # Maximizes the window to full screen
root.configure(bg='#2e2e2e')

# Set the font and styles
font_family = "Segoe UI"
header_font = (font_family, 18, 'bold italic')
subheader_font = (font_family, 14, 'italic')
normal_font = (font_family, 12)
btn_font = (font_family, 12, 'bold')
bg_color = '#2e2e2e'
text_color = '#ffffff'
button_color = '#0052cc'
button_hover_color = '#003d99'
entry_bg_color = '#404040'
entry_text_color = '#ffffff'
text_bg_color = '#404040'
text_text_color = '#ffffff'
border_color = '#ffffff'

# Frame for the main title with border
title_frame = Frame(root, bg=border_color, pady=2, padx=2)
title_frame.pack(pady=20)

Label(title_frame, text="Movie Recommendation System", font=header_font, bg=bg_color, fg=text_color, pady=10, padx=20).pack()

# Frame for the input section heading
input_frame = Frame(root, bg=border_color, pady=2, padx=2)
input_frame.pack(pady=10)

Label(input_frame, text="Enter Movie Title", font=subheader_font, bg=bg_color, fg=text_color, pady=5, padx=15).pack()

# Input Entry
entry = Entry(root, width=50, font=normal_font, bg=entry_bg_color, fg=entry_text_color, insertbackground=text_color)
entry.pack(pady=10)

# Button to get recommendations
def on_enter(e):
    e.widget['bg'] = button_hover_color

def on_leave(e):
    e.widget['bg'] = button_color

recommend_button = Button(
    root, text="Get Recommendations", font=btn_font, command=show_recommendations,
    bg=button_color, fg='#ffffff', relief=FLAT
)
recommend_button.pack(pady=15)
recommend_button.bind("<Enter>", on_enter)
recommend_button.bind("<Leave>", on_leave)

# Frame for recommendations heading with border
recommend_frame = Frame(root, bg=border_color, pady=2, padx=2)
recommend_frame.pack(pady=10)

Label(recommend_frame, text="Recommendations", font=subheader_font, bg=bg_color, fg=text_color, pady=5, padx=15).pack()

# Text widget to display recommendations
recommendations_text = Text(root, width=100, height=20, font=normal_font, bg=text_bg_color, fg=text_text_color, wrap=WORD)
recommendations_text.pack(pady=10)
recommendations_text.tag_configure('bold', font=(font_family, 12, 'bold'))
recommendations_text.tag_configure('header', font=(font_family, 14, 'bold'))
recommendations_text.tag_configure('normal', font=(font_family, 12))

# Start the GUI loop
root.mainloop()