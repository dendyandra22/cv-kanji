import streamlit as st

def get_sidebar():
    st.sidebar.markdown("## Get Started")
    st.sidebar.page_link("pages/home.py")

    st.sidebar.markdown("## Learn Kanji")
    st.sidebar.page_link("pages/learning_page.py")

    st.sidebar.markdown("## Exercise")
    st.sidebar.page_link("pages/exercise_page.py")

def get_credit_footer():
    st.divider()

    st.markdown(
        """
        <div style="text-align: center; font-size: 0.85rem;">
            <p><strong>Kakikata</strong> • Learn Kanji Stroke by Stroke</p>
            <p>
                <strong>Credits & Acknowledgments:</strong><br>
                Illustrations by <a href="https://www.magnific.com/author/jemastock" target="_blank">Jemastock</a> & <a href="https://www.magnific.com/author/pikisuperstar" target="_blank">pikisuperstar</a> • 
                Kanji Vectors by <a href="https://kanjivg.tagaini.net/" target="_blank">KanjiVG</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Credit Section
