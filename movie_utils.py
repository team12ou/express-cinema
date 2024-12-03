from random import choice

def get_most_shown_emotion(emotion_counter, labels_dict):
    max_key = max(emotion_counter, key=lambda x: emotion_counter[x])
    return labels_dict[max_key]

def get_genre(emotion, selected_lang):
        genre_dict = {
            "Angry": ["comedy", "family", "romance"],
            "Happy": ["horror", "sci-fi", "mystery"],
            "Neutral": ["action", "adventure", "drama"],
            "Sad": ["sport", "thriller", "crime"]
        }

        if str(selected_lang) == "English":
            genre_dict["Neutral"].append("western")
            genre_dict["Sad"].append("animation")

        result_list = genre_dict[emotion]
        result_genre = choice(result_list)
        result = "&genres=" + result_genre
        return result

def get_lang(selected_lang):
    lang_dict = {"English": "en", "Telugu": "te", "Malayalam": "ml", "Tamil": "ta", "Hindi": "hi"}
    result = "&primary_language=" + lang_dict[str(selected_lang)]
    return result

def open_url_based_on_emotion(emotion, selected_lang):
    base_url = "https://www.imdb.com/search/title/?title_type=feature&user_rating=7,10"
    genre = get_genre(emotion, selected_lang)
    lang = get_lang(selected_lang)
    sort_results = "&sort=num_votes,desc"
    result_url = base_url + genre + lang + sort_results
    return result_url

