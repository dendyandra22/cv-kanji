import streamlit as st

home_page = st.Page("app_pages/home.py", title="Home", default=True)
learn_page = st.Page("app_pages/learning_page.py", title="Practice Kanji Writing")
learn_n5_page = st.Page("app_pages/learn_n5.py", title="Practice Kanji N5")
exercise_page = st.Page("app_pages/exercise_page.py", title="Exercise Kanji Writing")
test_n5_page = st.Page("app_pages/test_n5.py", title="Test N5")
score_page = st.Page("app_pages/score_page.py", title="Result")

pages = {
        "Get Started": [home_page],
        "Learn Kanji": [learn_page,learn_n5_page],
        "Exercise": [exercise_page, test_n5_page],
        "Hidden": [score_page]
    }

pg = st.navigation(pages, position="hidden")

pg.run()