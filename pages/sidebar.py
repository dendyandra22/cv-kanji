import streamlit as st
from .cards import batch_card, lesson_card

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

def get_kanji_level_card(kanji_metadata, practice_route=True):
    col1, col2 = st.columns(2)
    page_prefix = "learn" if practice_route else "test"

    with col1.container(width=350, horizontal_alignment="center", border=True):
        lesson_card(
            label="JLPT N5",
            image="https://img.magnific.com/free-vector/studying-woman-with-book-isolated_18591-83807.jpg",
            content="Beginner level • Essential Basic Kanji",
            expander_data=kanji_metadata,
            btn_text="Start N5 Practice",
            page_ref=f"{page_prefix}_n5.py"
        )

    with col2.container(width=350, horizontal_alignment="center", border=True):
        lesson_card(
            label="JLPT N4 (Coming Soon)",
            image="https://img.magnific.com/free-vector/studying-woman-writing-isolated_18591-83802.jpg",
            content="Elementary level • Everyday Kanji",
            btn_text="Start N4 Practice",
            page_ref=""
        )

    with col1.container(width=350, horizontal_alignment="center", border=True):
        lesson_card(
            label="JLPT N3 (Coming Soon)",
            image="https://img.magnific.com/free-vector/doodle-hand-drawn-cartoon-cute-girl-student-with-correct-ignore-symbols-no-tag-answer_40876-3282.jpg",
            content="Intermediate level • Pre-advanced Kanji",
            btn_text="Start N3 Practice",
            page_ref=""
        )

    with col2.container(width=350, horizontal_alignment="center", border=True):
        lesson_card(
            label="JLPT N2 (Coming Soon)",
            image="https://img.magnific.com/free-vector/doodle-hand-drawn-cartoon-cute-girl-student-with-correct-ignore-symbols-no-tag-answer_40876-3282.jpg",
            content="Upper-Intermediate level • Business Kanji",
            btn_text="Start N2 Practice",
            page_ref=""
        )

    with col1.container(width=350, horizontal_alignment="center", border=True):
        lesson_card(
            label="JLPT N1 (Coming Soon)",
            image="https://img.magnific.com/free-vector/doodle-hand-drawn-cartoon-cute-girl-student-with-correct-ignore-symbols-no-tag-answer_40876-3282.jpg",
            content="Advanced level • Fluent & native-level Kanji",
            btn_text="Start N1 Practice",
            page_ref=""
        )

def get_kanji_batch_n5(kanji_metadata, idx_batch):
    batch_opt = []

    with st.container(horizontal_alignment="center", vertical_alignment="center", border=False):
        col1, col2, col3 = st.columns([1,1,1])

        with col1.container(width=350, horizontal_alignment="center", border=True):
            expander_data = [kanji_metadata[i] for i in idx_batch["kanji 1"]]
            sel_group_1 = batch_card(
                label="Kanji Group 1",
                expander_data=expander_data,
            )

        if sel_group_1:
            batch_opt.append("kanji 1")

        with col2.container(width=350, horizontal_alignment="center", border=True):
            expander_data = [kanji_metadata[i] for i in idx_batch["kanji 2"]]
            sel_group_2 = batch_card(
                label="Kanji Group 2",
                expander_data=expander_data,
            )

        if sel_group_2:
            batch_opt.append("kanji 2")

        with col3.container(width=350, horizontal_alignment="center", border=True):
            expander_data = [kanji_metadata[i] for i in idx_batch["kanji 3"]]
            sel_group_3 = batch_card(
                label="Kanji Group 3",
                expander_data=expander_data,
            )

        if sel_group_3:
            batch_opt.append("kanji 3")

        with col1.container(width=350, horizontal_alignment="center", border=True):
            expander_data = [kanji_metadata[i] for i in idx_batch["kanji 4"]]
            sel_group_4 = batch_card(
                label="Kanji Group 4",
                expander_data=expander_data,
            )

        if sel_group_4:
            batch_opt.append("kanji 4")

        with col2.container(width=350, horizontal_alignment="center", border=True):
            expander_data = [kanji_metadata[i] for i in idx_batch["kanji 5"]]
            sel_group_5 = batch_card(
                label="Kanji Group 5",
                expander_data=expander_data,
            )

        if sel_group_5:
            batch_opt.append("kanji 5")

        with col3.container(width=350, horizontal_alignment="center", border=True):
            expander_data = [kanji_metadata[i] for i in idx_batch["kanji 6"]]
            sel_group_6 = batch_card(
                label="Kanji Group 6",
                expander_data=expander_data,
            )

        if sel_group_6:
            batch_opt.append("kanji 6")

        with col1.container(width=350, horizontal_alignment="center", border=True):
            expander_data = [kanji_metadata[i] for i in idx_batch["kanji 7"]]
            sel_group_7 = batch_card(
                label="Kanji Group 7",
                expander_data=expander_data,
            )

        if sel_group_7:
            batch_opt.append("kanji 7")

    return batch_opt
