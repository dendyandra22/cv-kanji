import streamlit as st
import pandas as pd
from torchvision.datasets import FakeData


def menu_card(label, image, content, btn_text, page_ref):
    st.markdown(
        f"""
        #### {label}
        """
    , text_alignment="center")
    # st.image(f"static/img/{image}")
    st.image(image)
    st.markdown(content, text_alignment="center")

    if st.button(btn_text, type="primary"):
        st.switch_page(f"pages/{page_ref}")

def lesson_card(label, image="", content="", expander_data=None, btn_text="", page_ref=""):
    st.markdown(
        f"""
        #### {label}
        """
    , text_alignment="center")
    if image:
        # st.image(f"static/img/{image}", width=280)
        st.image(image)
    if content:
        st.markdown(content, text_alignment="center")

    if st.button(btn_text, type="primary"):
        if page_ref:
            st.switch_page(f"pages/{page_ref}")

    with st.expander("See Kanji List"):
        if isinstance(expander_data, list):
            with st.container(height=200, border=False):
                tmp_df = pd.DataFrame(expander_data)
                tmp_df = tmp_df[["kanji", "meaning"]]
                tmp_df = tmp_df.set_index("kanji")
                st.dataframe(tmp_df)
        else:
            st.info("No further detail available")

def batch_card(label, expander_data=None):
    st.markdown(
        f"""
        #### {label}
        """
    , text_alignment="center")

    # with st.expander("See Kanji List"):
    if isinstance(expander_data, list):
        with st.container(height=150, border=False):
            tmp_df = pd.DataFrame(expander_data)
            tmp_df = tmp_df[["kanji", "meaning"]]
            tmp_df = tmp_df.set_index("kanji")
            st.dataframe(tmp_df)
    else:
        st.info("No further detail available")

    selected_status = False
    if st.checkbox(f"Use {label}"):
        selected_status = True

    return selected_status