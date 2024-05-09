import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
merged_df = pd.read_csv('merged_df.csv')

def get_recommended_asanas(health_problems, merged_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(merged_df['Benefits'])
    tfidf_health_problems = vectorizer.transform(health_problems)
    cosine_similarities = cosine_similarity(tfidf_health_problems, tfidf_matrix)
    recommended_asanas = []
    for idx in range(len(health_problems)):
        similar_indices = cosine_similarities[idx].argsort()[:-6:-1]
        recommended_asanas.extend(merged_df['Asana'].iloc[similar_indices].values)
    return list(set(recommended_asanas))[:5]

def main():
    st.title('Asana Recommendation for Health Issues')

    health_problem_1 = st.text_input('Enter your first health problem', '')
    health_problem_2 = st.text_input('Enter your second health problem', '')
    health_problem_3 = st.text_input('Enter your third health problem', '')
    health_problem_4 = st.text_input('Enter your fourth health problem', '')

    if st.button('Recommend Asanas'):
        health_problems = [health_problem_1, health_problem_2, health_problem_3, health_problem_4]
        health_problems = [hp for hp in health_problems if hp]
        if health_problems:
            recommended_asanas = get_recommended_asanas(health_problems, merged_df)
            st.write('Based on your health problems, we recommend the following asanas:')
            for asana in recommended_asanas:
                st.write(asana)
        else:
            st.write('Please enter at least one health problem.')

if __name__ == '__main__':
    main()