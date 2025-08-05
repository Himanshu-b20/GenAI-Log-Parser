import streamlit as st
from rag import process_data, generate_answer

st.set_page_config(page_title="Log Parser", layout="wide")

# Sidebar section
st.sidebar.title("Upload Text Files")
uploaded_files = st.sidebar.file_uploader(
    "Choose one or more .txt files",
    type=["txt"],
    accept_multiple_files=True
)
process_clicked = st.sidebar.button("Process")

# Main page
st.title("Log Parsing Tool")

# Initialize combined_text
combined_text = ""
placeholder = st.empty()

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one text file.")
    else:
        for file in uploaded_files:
            content = file.read().decode("utf-8")
            combined_text += content + "\n"  # Add newline between files

    with open("temp_file.txt", "w") as f:
        f.write(combined_text)

    for status in process_data("temp_file.txt"):
        placeholder.text(status)

query = placeholder.text_input('Question')

if query:
    try:
        answer, source = generate_answer(query)
        st.header('Answer')
        st.write(answer)
    except RuntimeError as e:
        placeholder.text('Exception Occurred')