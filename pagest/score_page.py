import streamlit as st
import pandas as pd

st.set_page_config(page_title="Result")

# Pass/Fail Threshold Constants
PASSING_FINAL_SCORE = 70.0
MIN_DTW_SCORE = 0.0
MIN_DIRECTION_SCORE = 70.0


def evaluate_kanji_attempt(item):
    """
    Returns True if the drawing meets overall and hard-gate thresholds.
    """
    is_overall_pass = item["final_score"] >= PASSING_FINAL_SCORE
    is_stroke_order_valid = (
            item["dtw_score"] >= MIN_DTW_SCORE and
            item["direction_score"] >= MIN_DIRECTION_SCORE
    )

    return is_overall_pass and is_stroke_order_valid

def get_final_grade(accuracy_pct):
    print("accuracy_pct",accuracy_pct)
    grade = "E"
    grade_color = "c32f27"
    if accuracy_pct >= 90:
        grade = "S"
        grade_color = "fb8500"
    elif accuracy_pct >= 80:
        grade = "A"
        grade_color = "ffb703"
    elif accuracy_pct >= 70:
        grade = "B"
        grade_color = "219ebc"
    elif accuracy_pct >= 60:
        grade = "C"
        grade_color = "6a994e"
    elif accuracy_pct >= 50:
        grade = "D"
        grade_color = "386641"

    # st.markdown(
    #     f"""
    #         <div style="
    #             border-radius: 10px;
    #             padding: 0px;
    #             text-align: center;
    #         ">
    #             <div style="font-size: 20px; color: #555555; font-weight: 500; margin-bottom: 2px;">
    #                 Grade
    #             </div>
    #             <div style="margin: 0px 0;">
    #                 <span style="color: #{grade_color}; font-size: 42px; font-weight: bold; font-family: monospace;">
    #                     {grade}
    #                 </span>
    #             </div>
    #         </div>
    #         """,
    #     unsafe_allow_html=True
    # )

    return grade



st.title("📊 Test Results")

test_data = st.session_state.get("test_score", [])

if not test_data:
    st.warning("No test results found. Please complete an exercise first.")
    st.stop()

# 1. Calculate Pass/Fail for each Kanji
evaluated_data = []
passed_count = 0

for item in test_data:
    is_correct = evaluate_kanji_attempt(item)
    if is_correct:
        passed_count += 1

    evaluated_data.append({
        **item,
        "is_correct": is_correct,
        "status": "PASS ✅" if is_correct else "FAIL ❌"
    })

total_kanji = len(evaluated_data)
accuracy_pct = (4 / total_kanji) * 100
grade = get_final_grade(accuracy_pct)

# 2. Display Top Banner Metrics
with st.container(width="stretch"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Correct Kanji", f"{passed_count} / {total_kanji}")
    m2.metric("Accuracy Rate", f"{accuracy_pct:.1f}%")
    m3.metric("Grade", f"{grade}")
    # with m1:
    #     st.markdown(
    #         f"""
    #                 <div style="
    #                     border-radius: 10px;
    #                     padding: 0px;
    #                     text-align: center;
    #                 ">
    #                     <div style="font-size: 20px; color: #555555; font-weight: 500; margin-bottom: 2px;">
    #                         Correct Kanji
    #                     </div>
    #                     <div style="margin: 0px 0;">
    #                         <span style= font-size: 42px; font-weight: bold; font-family: monospace;">
    #                             {passed_count}/{total_kanji}
    #                         </span>
    #                     </div>
    #                 </div>
    #                 """,
    #         unsafe_allow_html=True
    #     )
    #
    # with m2:
    #     st.markdown(
    #         f"""
    #                         <div style="
    #                             border-radius: 10px;
    #                             padding: 0px;
    #                             text-align: center;
    #                         ">
    #                             <div style="font-size: 20px; color: #555555; font-weight: 500; margin-bottom: 2px;">
    #                                 Accuracy Rate
    #                             </div>
    #                             <div style="margin: 0px 0;">
    #                                 <span style= font-size: 42px; font-weight: bold; font-family: monospace;">
    #                                     {accuracy_pct:.1f}%
    #                                 </span>
    #                             </div>
    #                         </div>
    #                         """,
    #         unsafe_allow_html=True
    #     )
    #
    # with m3:
    #     get_final_grade(accuracy_pct)

st.divider()

# 3. Categorized Breakdown (Mastered vs Needs Practice)
col_pass, col_fail = st.columns(2)

with col_pass:
    st.subheader("✅ Mastered Kanji")
    passed_list = [d["kanji"] for d in evaluated_data if d["is_correct"]]
    if passed_list:
        st.write(" ".join([f"`{k}`" for k in passed_list]))
    else:
        st.info("None in this session.")

with col_fail:
    st.subheader("⚠️ Needs Practice")
    failed_list = [d["kanji"] for d in evaluated_data if not d["is_correct"]]
    if failed_list:
        st.write(" ".join([f"`{k}`" for k in failed_list]))
    else:
        st.success("Great job! All Kanji were drawn correctly.")

st.divider()

# 4. Detailed Breakdown Table
st.subheader("Detailed Breakdown")
df = pd.DataFrame(evaluated_data)

# Format table for clean display
df_display = df[
    ["status", "kanji", "final_score", "dtw_score", "direction_score", "ssim_score", "cnn_similarity"]].copy()
df_display.columns = ["Status", "Kanji", "Final Score (%)", "Stroke Order (DTW)", "Direction", "SSIM", "CNN Similarity"]

# Format floats to 1 decimal place
for col in df_display.columns[2:]:
    df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")

st.dataframe(df_display, use_container_width=True)

st.divider()
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
with nav_col2:
    if st.button("← Back to Menu", type="primary"):
        st.session_state.kanji_index = 0
        st.session_state.in_practice = False
        _ = st.session_state.pop("sett_n_kanji", None)
        _ = st.session_state.pop("sett_random_kanji", None)
        _ = st.session_state.pop("sett_random_seed", None)
        _ = st.session_state.pop("test_score", None)
        st.switch_page("pagest/test_n5.py")