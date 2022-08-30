import os
import sys
import pickle
import streamlit as st
import requests
from movies_recommender_app_logger.logger import logging
from movies_recommender_app_configuration.configuration import AppConfiguration
from movies_recommender_app_pipeline.training_pipeline import TrainingPipeline
from movies_recommender_app_exception.exception_handler import AppException

class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config= app_config.get_recommendation_config()
            self.movies =  pickle.load(open(self.recommendation_config.serialized_objects_file_path,'rb'))
            self.similarity = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
        except Exception as e:
            raise AppException(e, sys) from e


    def fetch_poster(self,movie_id):
        try:
            poster_api = self.recommendation_config.poster_api
            url = poster_api.format(movie_id)
            data = requests.get(url)
            data = data.json()
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
        except Exception as e:
            raise AppException(e, sys) from e
        


    def recommend(self,movie):
        try:
            index = self.movies[self.movies['title'] == movie].index[0]
            distances = sorted(list(enumerate(self.similarity[index])), reverse=True, key=lambda x: x[1])
            recommended_movie_names = []
            recommended_movie_posters = []
            for i in distances[1:6]:
                # fetch the movie poster
                movie_id = self.movies.iloc[i[0]].movie_id
                recommended_movie_posters.append(self.fetch_poster(movie_id))
                recommended_movie_names.append(self.movies.iloc[i[0]].title)

            return recommended_movie_names,recommended_movie_posters
        except Exception as e:
            raise AppException(e, sys) from e


    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e

    
    def recommendations_engine(self,selected_movie):
        try:
            recommended_movie_names,recommended_movie_posters = self.recommend(selected_movie)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
            with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])

            with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
            with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
            with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4])
        except Exception as e:
            raise AppException(e, sys) from e



if __name__ == "__main__":
    st.header('ML Based Movies Recommender System')
    st.text("This is a content based recommendation system!")

    obj = Recommendation()

    movie_list = obj.movies['title'].values
    selected_movie = st.selectbox(
            "Type or select a movie from the dropdown",
            movie_list
        )

    #Training
    if st.button('Train Recommender System'):
        obj.train_engine()
    
    #recommendation
    if st.button('Show Recommendation'):
        obj.recommendations_engine(selected_movie)
