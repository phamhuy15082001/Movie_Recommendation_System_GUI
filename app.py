from datetime import datetime
import streamlit as st
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
import requests
from config import API_KEY
import os 
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure this is the first Streamlit command
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## A content based movie recommendation system build using cosine similarity"
    }
)

# Function to load data
# @st.cache(persist=True, allow_output_mutation=True)
def load_movies_data(file_path):
    print('loading movies data')
    df = pd.read_csv(file_path)
    pickle.dump(df.to_dict(), open('./Dataset/movies.pkl', 'wb'))

# function to convert all strings to lower case and strip namse of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ",""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast'])+ ' '+ x['director']+' '+' '.join(x['genres'])

def load_similary_data(file_path):
    print('loading similary data')
    df = pd.read_csv(file_path)
    features = ['cast', 'crew', 'keywords', 'genres']

    for feature in features:
        df[feature] = df[feature].apply(literal_eval)
    
    # define new director, cast, genres and keywords features that are in  a suiable form
    df['director'] = df['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(get_list)
    
    for feature in features:
        df[feature] = df[feature].apply(clean_data)
    
    df['soup'] = df.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    pickle.dump(cosine_sim2, open('../Dataset/similarity.pkl', 'wb'))


# Function to reload data
def reload_data():
    global movies_dict, movies, similarity
    print('reloading')
    load_movies_data(data_file)
    # load_similary_data(data_file)
    movies_dict = pickle.load(open('./Dataset/movies.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('./Dataset/similarity.pkl', 'rb'))
    last_modified = os.path.getmtime(data_file)
    write_last_modified(last_modified)


# DB management
conn = sqlite3.connect('./database/data.db')
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

def add_userdata(username, password):
    hashed_password = generate_password_hash(password, method='sha256')
    
    c.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()


def check_authenticated():
    menu = ["Login", "Sign Up"]
    
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        st.subheader("Login Section")

        username = st.text_input("User Name")
        password = st.text_input("Password", type='password', key="login_pass")
        
        if st.button("Login"):
            if len(username) == 0 and len(password) == 0:
                    st.error("You haven't typed anything to log in!", icon="🚨")
            elif len(username) < 3:
                st.error('Your username has to be more than or equal 3 characters', icon="🚨")
            elif len(password) < 6:
                st.error('Your password has to be more then or equal 6 characters', icon="🚨")
            else:   
                create_usertable()
                c.execute('SELECT * FROM userstable WHERE username=?', (username,))
                data = c.fetchone()
                if data:
                    if check_password_hash(data[1], password):
                        st.success("Logged In as {}".format(username))
                        st.session_state["page"] = "main"
                        st.balloons()
                        st.experimental_rerun()
                else:
                    st.error('Account does not exist!', icon="🚨")
    
    elif choice == "Sign Up":
        st.subheader("Create New Account")

        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password', key="signup_pass")
        
        if st.button("Sign Up"):
            create_usertable()
            c.execute('SELECT * FROM userstable WHERE username=?', (new_user,))
            data = c.fetchone()
            
            if data:
                st.error('Username already exist!', icon="🚨")
            else:
                if len(new_user) == 0 and len(new_password) == 0:
                    st.error("You haven't typed anything to sign up!", icon="🚨")
                elif len(new_user) < 3:
                    st.error('Your username has to be more than or equal 3 characters', icon="🚨")
                elif len(new_password) < 6:
                    st.error('Your password has to be more then or equal 6 characters', icon="🚨")
                
                else:
                    add_userdata(new_user, new_password)
                    st.success("You have successfully created an account!")
                    st.info("Go to Login Menu to login")
    return False


# create poster function
# Create poster function
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(movie_id, API_KEY)
    response = requests.get(url)
    data = response.json()
    # print(data)
    if response.status_code == 200 and 'poster_path' in data:
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    else:
        return None

def recommend(movie):
    try:
        if 'last_check' not in st.session_state:
            st.session_state['last_check'] = datetime.now() 
        last_modified = read_last_modified()
        current_modified = os.path.getmtime(data_file)
        if current_modified != last_modified:
            with st.spinner('Please wait to update new dataset...'):
                reload_data()
        st.session_state['last_check'] = datetime.now()

        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        
        recommended_movies = []
        recommended_movies_posters = []
        info_list = []
        for i in movies_list:
            # fetch the movie poster
            movie_id = movies.iloc[i[0]].id
            recommended_movies.append(movies.iloc[i[0]].title)
            info = [movies.iloc[i[0]].release_date, movies.iloc[i[0]].runtime, movies.iloc[i[0]].vote_average, movies.iloc[i[0]].vote_count]
            info_list.append(info)
            recommended_movies_posters.append(fetch_poster(movie_id))
        return recommended_movies, info_list, recommended_movies_posters
    except Exception as e:
        st.error(f"Error recommending movies: {e}")
        return None, None


def main_screen():
    last_modified = read_last_modified()
    # Display last modified time
    last_modified_time = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')
    st.success(f"Dataset last modified: {last_modified_time}")
    if st.button('Check update'):
        if 'last_check' not in st.session_state:
            st.session_state['last_check'] = datetime.now() 
        last_modified = read_last_modified()
        current_modified = os.path.getmtime(data_file)
        print(f'{last_modified = }')
        print(f'{current_modified = }')
        if current_modified != last_modified:
            with st.spinner('Please wait to update new dataset...'):
                reload_data()
            st.success('Dataset is up-to-date!')
        else:
            st.info('Dataset has not changed!')
        st.session_state['last_check'] = datetime.now()

    
    selected_movie_name = st.selectbox(
        "Type or select a movie from the dropdown",
        movies['title'].values
    )

    if st.button('Show Recommendation'):
        names, info_list, posters = recommend(selected_movie_name)
        print(f'{names}=')
        print(f'{info_list = }')
        if names:
            cols = st.columns(3)
            for i in range(len(names)):
                with cols[i % 3]:
                    st.success(names[i])
                    st.code(f'Released Date: {info_list[i][0]}\nRuntime: {info_list[i][1]} minutes\nVote Average: {info_list[i][2]}/10.0\nVote Count: {info_list[i][3]}', language='python')

                    if posters[i]:
                        st.image(posters[i])
                    else:
                        st.error("Poster not available")
                # Add spacer after each row
                if (i + 1) % 3 == 0:
                    st.write("")  # Adds a blank line
        else:
            st.error("No recommendations found")
    
    if st.button("Log out"):
        st.session_state["page"] = "login"
        st.experimental_rerun()


def main():
    st.title('Movie Recommendation System')


    if "page" not in st.session_state:
        st.session_state["page"] = "login"
    
    if st.session_state["page"] == "login":
        check_authenticated()
    elif st.session_state["page"] == "main":
        main_screen()

def write_last_modified(last_modified):
    with open('last_modified.txt', 'w') as file:
        file.write(str(last_modified))

def read_last_modified():
    try:
        with open('last_modified.txt', 'r') as file:
            return float(file.read())
    except FileNotFoundError:
        return 0  # Return 0 if the file doesn't exist
    
if __name__ == '__main__':
    # Set file path
    data_file = './Dataset/tmdb_simpled_v3.csv'

    # Global variable to store the data
    last_modified = read_last_modified()
    # write_last_modified(last_modified)
    # print(f'frist check: {last_modified =}')
    movies_dict = pickle.load(open('./Dataset/movies.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('./Dataset/similarity.pkl', 'rb'))




    # Initial data load
    main()

    # # Check for changes every 10 seconds
    # if 'last_check' not in st.session_state:
    #     st.session_state['last_check'] = datetime.now()

    # if (datetime.now() - st.session_state['last_check']).total_seconds() > 10:
    #     current_modified = os.path.getmtime(data_file)

    #     if current_modified != last_modified:
    #         with st.spinner('Please wait to update new dataset...'):
    #             reload_data()
    #     # last_modified = os.path.getmtime(data_file)
    #     st.session_state['last_check'] = datetime.now()

