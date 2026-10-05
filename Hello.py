import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

pages = {
    "Get Started": [
        st.Page("pagest/home.py", title="Home")
    ],

    "Learn Kanji": [
        st.Page("pagest/learn_n5.py", title="Practice Kanji N5"),
    ],
    "Exercise": [
        st.Page("pagest/test_n5.py", title="Test N5"),
    ]
}
score_page = st.Page("pagest/score_page.py", title="Test Result")

pg = st.navigation(pages)

# Register the hidden page separately or pass all pages to st.navigation
# Passing both visible and hidden pages inside a dict + hidden parameter:
pg = st.navigation(
    {
        **pages,
        "": [score_page]  # Placing score_page outside standard sections
    },
    position="sidebar"
)
pg.run()

# st.write("# Welcome to Streamlit! 👋")
#
# st.sidebar.success("Select a demo above.")
#
# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#         Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )