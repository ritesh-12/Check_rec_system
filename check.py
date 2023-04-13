from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


def pred_rating(model, user_id, anime_ids):
    user_input = np.repeat(user_id, len(anime_ids))
    anime_input = np.array(anime_ids)
    rating_probs = model.predict([user_input, anime_input])
    ratings = {}
    for i in range(len(anime_ids)):
        ratings[anime_ids[i]] = np.argmax(rating_probs[i]) + 1
    return ratings


def pred_top_n_anime(model, user_id, n):
    datasize = 5000000
    df = pd.read_csv('rating_complete.csv')[:datasize]
    df1 = pd.read_csv('anime.csv')
    anime_seen = {}
    anime_ids = set()
    for i in range(len(df)):
        user = df['user_id'].iloc[i]
        anime = df['anime_id'].iloc[i]
        if user not in anime_seen:
            anime_seen[user] = []
        anime_seen[user].append(anime)
        anime_ids.add(anime)

    map_animeId_to_name = {}
    for i in range(len(df1)):
        ids = df1['MAL_ID'].iloc[i]
        name = df1['English name'].iloc[i]
        if name == 'Unknown':
            name = df1['Name'].iloc[i]
        map_animeId_to_name[ids] = name
    anime_ids = list(anime_ids)

    if user_id not in anime_seen:
        print('User NOT FOUND')
        return
    not_seen = []
    for ai in anime_ids:
        if ai not in anime_seen[user_id]:
            not_seen.append(ai)

    ratings = pred_rating(model, user_id, not_seen)
    ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    top_anime_ids = list(map(lambda x: x[0], ratings[:n]))
    anime_names = []
    for curr in top_anime_ids:
        anime_names.append(map_animeId_to_name[curr])

    return anime_names


def analysis(name):
    name = name.replace(" ", "%20")
    search_link_left = "https://myanimelist.net/anime.php?cat=anime&q="
    search_link_right = "&type=0&score=0&status=0&p=0&r=0&sm=0&sd=0&sy=0&em=0&ed=0&ey=0&c%5B%5D=a&c%5B%5D=b&c%5B%5D=c&c%5B%5D=f"
    search_link = search_link_left + name + search_link_right
    # print(search_link)

    import requests
    from bs4 import BeautifulSoup

    # Make a GET request to the webpage
    url = search_link
    response = requests.get(url)

    # Create a Beautiful Soup object from the HTML content of the webpage
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the <strong> tags in the HTML and extract their text content
    strong_tags = soup.find_all('strong')
    strong_texts = [tag.string for tag in strong_tags if tag.string]

    # Print the extracted texts
    # print(strong_texts)

    series_dict = {}
    for i, series in enumerate(strong_texts):
        series_dict[i + 1] = series

    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
        return dp[m][n]

    def find_series_name(name, series_dict):
        for key, value in series_dict.items():
            if edit_distance(value, name) < 0:
                return value
        return None

    matched_series = find_series_name(name, series_dict)

    if matched_series is not None:
        # print(f"The first matched series is: {matched_series}")
        value = {matched_series}
    else:
        print("Top matched series that matches input:")
        for i, (k, v) in enumerate(series_dict.items()):
            print(f"{k}: {v}")
            if i == 4:
                break

    print("6: None")
    print("Enter the number of the respective series you want ")
    key1 = int(input())
    if key1 == 6:
        print("Try Again")
        return
    else:
        value = series_dict[key1]
        print("Fetching Reviews...")

    import requests
    from bs4 import BeautifulSoup

    url = search_link
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    series_dict2 = {}

    series_tags = soup.find_all('a', class_='hoverinfo_trigger fw-b fl-l')[:5]
    # print(series_tags)
    for series_tag in series_tags:
        series_name = series_tag.strong.string
        #  print(series_name)
        series_link = series_tag['href']
        series_dict2[series_name] = series_link
    # <a class="hoverinfo_trigger fw-b fl-l" href="https://myanimelist.net/anime/966/Crayon_Shin-chan" id="sinfo966" rel="#sinfo966" style="position: relative;"><strong>Crayon Shin-chan</strong></a>
    # print(series_dict2)

    final_link = series_dict2[value]
    # print(final_link)

    list1 = []
    curr_link = final_link + "/reviews?sort=suggested&filter_check=&filter_hide=&preliminary=on&spoiler=off&p={}"
    # print(curr_link)
    page_num = 1

    while True:
        page_link = curr_link.format(page_num)
        response = requests.get(page_link)

        soup = BeautifulSoup(response.content, 'html.parser')
        review_elements = soup.find_all('div', {'class': 'text'})
        reviews = [review_element.get_text() for review_element in review_elements]
        for review in reviews:
            list1.append(review)
        next_button = soup.find('a', {'data-ga-click-type': 'review-more-reviews'})
        if next_button:
            page_num += 1
        else:
            print("Number of review:", len(list1))
            # for i in range(len(list1)):
            #    print({list1[i]})
            break
    else:
        print("No reviews are there for this series")

    story_list = []
    animation_list = []
    sound_list = []
    character_list = []
    music_list = []

    analyzer = SentimentIntensityAnalyzer()

    def find_reviews_with_word(reviews_list, word):
        """
        This function takes a list of reviews and a word as input and returns a list
        of reviews that contain the given word.
        """
        matching_reviews = []
        for review in reviews_list:
            if word in review:
                matching_reviews.append(review)
        return matching_reviews

    def find_reviews_with_word_all():
        p = 0
        n = 0
        reviews = list1
        print("total reviews related to story:", len(reviews))
        # print(reviews)
        if len(reviews) == 0:
            print("No reviews for this Series")
            return
        for review in reviews:
            score = analyzer.polarity_scores(review)
            # print(score)
        i = 1
        for review in reviews:
            score = analyzer.polarity_scores(review)['compound']

            if score > 0.5:
                p = p + 1
                # print("review",i,".","Positive")
            elif score < -0.5:
                n = n + 1
                # print("review",i,".","Negative")
            else:
                # print("review",i,".","Neutral")
                i = i + 1
        positivity = p / len(reviews)
        negativity = n / len(reviews)
        print("positivity in all reviews:", round(positivity * 100, 2), "%")
        print("negativity in all reviews:", round(negativity * 100, 2), "%")

    def find_reviews_with_word_story():
        p = 0
        n = 0
        word = "story"
        reviews = list1
        story_reviews = find_reviews_with_word(reviews, word)
        print("total reviews related to story:", len(story_reviews))
        # print(story_reviews)
        if len(story_reviews) == 0:
            print("No reviews for this category")
            return
        for review in story_reviews:
            score = analyzer.polarity_scores(review)
            # print(score)
        i = 1
        for review in story_reviews:
            score = analyzer.polarity_scores(review)['compound']

            if score > 0.5:
                p = p + 1
                # print("review",i,".","Positive")
            elif score < -0.5:
                n = n + 1
                # print("review",i,".","Negative")
            else:
                # print("review",i,".","Neutral")
                i = i + 1
        positivity = p / len(story_reviews)
        negativity = n / len(story_reviews)
        print("positivity in reviews related to story:", round(positivity * 100, 2), "%")
        print("negativity in reviews related to story:", round(negativity * 100, 2), "%")

    def find_reviews_with_word_Animation():
        p = 0
        n = 0
        word = "Animation"
        reviews = list1
        Animation_reviews = find_reviews_with_word(reviews, word)
        print("total reviews related to Animation:", len(Animation_reviews))
        # print(Animation_reviews)
        if len(Animation_reviews) == 0:
            print("No reviews for this category")
            return
        else:
            for review in Animation_reviews:
                score = analyzer.polarity_scores(review)
                # print(score)
            i = 1
            for review in Animation_reviews:
                score = analyzer.polarity_scores(review)['compound']

                if score > 0.5:
                    p = p + 1
                    # print("review",i,".","Positive")
                elif score < -0.5:
                    n = n + 1
                    # print("review",i,".","Negative")
                else:
                    # print("review",i,".","Neutral")
                    i = i + 1
        positivity = p / len(Animation_reviews)
        negativity = n / len(Animation_reviews)
        print("positivity in reviews related to Animation:", round(positivity * 100, 2), "%")
        print("negativity in reviews related to Animation:", round(negativity * 100, 2), "%")

    def find_reviews_with_word_Sound():
        p = 0
        n = 0
        word = "Sound"
        reviews = list1
        Sound_reviews = find_reviews_with_word(reviews, word)
        print("total reviews related to Animation:", len(Sound_reviews))
        # print(Sound_reviews)
        if len(Sound_reviews) == 0:
            print("No reviews for this category")
            return
        for review in Sound_reviews:
            score = analyzer.polarity_scores(review)
            # print(score)
        i = 1
        for review in Sound_reviews:
            score = analyzer.polarity_scores(review)['compound']

            if score > 0.5:
                p = p + 1
                # print("review",i,".","Positive")
            elif score < -0.5:
                n = n + 1
                # print("review",i,".","Negative")
            else:
                # print("review",i,".","Neutral")
                i = i + 1
        positivity = p / len(Sound_reviews)
        negativity = n / len(Sound_reviews)
        print("positivity in reviews related to Sound:", round(positivity * 100, 2), "%")
        print("negativity in reviews related to Sound:", round(negativity * 100, 2), "%")

    def find_reviews_with_word_Character():
        p = 0
        n = 0
        word = "Character"
        reviews = list1
        Character_reviews = find_reviews_with_word(reviews, word)
        print("total reviews related to Animation:", len(Character_reviews))
        # print(Character_reviews)
        if len(Character_reviews) == 0:
            print("No reviews for this category")
            return
        for review in Character_reviews:
            score = analyzer.polarity_scores(review)
            # print(score)
        i = 1
        for review in Character_reviews:
            score = analyzer.polarity_scores(review)['compound']

            if score > 0.5:
                p = p + 1
                # print("review",i,".","Positive")
            elif score < -0.5:
                n = n + 1
                # print("review",i,".","Negative")
            else:
                # print("review",i,".","Neutral")
                i = i + 1
        positivity = p / len(Character_reviews)
        negativity = n / len(Character_reviews)
        print("positivity in reviews related to Character:", round(positivity * 100, 2), "%")
        print("negativity in reviews related to Character:", round(negativity * 100, 2), "%")

    def find_reviews_with_word_Music():
        p = 0
        n = 0
        word = "Music"
        reviews = list1
        Music_reviews = find_reviews_with_word(reviews, word)
        print("total reviews related to Animation:", len(Music_reviews))
        # print(Music_reviews)
        if len(Music_reviews) == 0:
            print("No reviews for this category")
            return
        for review in Music_reviews:
            score = analyzer.polarity_scores(review)
            # print(score)
        i = 1
        for review in Music_reviews:
            score = analyzer.polarity_scores(review)['compound']

            if score > 0.5:
                p = p + 1
                # print("review",i,".","Positive")
            elif score < -0.5:
                n = n + 1
                # print("review",i,".","Negative")
            else:
                # print("review",i,".","Neutral")
                i = i + 1
        positivity = p / len(Music_reviews)
        negativity = n / len(Music_reviews)
        print("positivity in reviews related to Music:", round(positivity * 100, 2), "%")
        print("negativity in reviews related to Music:", round(negativity * 100, 2), "%")

    print("On what basis you want to review series")
    print("1. Overall review")
    print("2. Story")
    print("3. Animation")
    print("4. Sound")
    print("5. Character")
    print("6. Music")
    print("Enter the number from above list")
    n = input()

    if n == "1":
        find_reviews_with_word_all()
    elif n == "2":
        find_reviews_with_word_story()
    elif n == "3":
        find_reviews_with_word_Animation()
    elif n == "4":
        find_reviews_with_word_Sound()
    elif n == "5":
        find_reviews_with_word_Character()
    elif n == "6":
        find_reviews_with_word_Music()
    else:
        print(1)

#if __name__ == '__main__
def main():
    # loading model
    model1_load = keras.models.load_model('model1')
    print("Please choose the option from following:")
    print("Enter 1 for anime recommendation")
    print("Enter 2 if you want review of particular anime series ")
    print("Enter 0 for quit")
    choice = int(input())

    while (choice):
        if (choice == 1):
            print("Enter 1 for old user")
            print("Enter 2 for new user")

            curr_choice = int(input())
            if (curr_choice == 1):
                print("Please enter your UserID(Numeric): ")
                user_id = int(input())
                print("How many recommnedations you want?")
                n = int(input())  # top n anime
                print("Fetching best anime for you...")
                anime = pred_top_n_anime(model1_load, user_id, n)
                print("Here is top {} anime for you".format(n))
                for curr in anime:
                    print(curr)
            elif (curr_choice == 2):
                print("Cold start problem")

        elif (choice == 2):
            print("Please enter series name:")
            name = input()
            analysis(name)

        else:
            break

        print("Enter 1 for anime recommendation")
        print("Enter 2 if you want review of particular anime series ")
        print("Enter 0 for quit")
        choice = int(input())

main()




