import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag_logic import load_vector_db, run_rag
from tempfile import NamedTemporaryFile
# temp to sync
import datetime
st.sidebar.write("Reload time:", datetime.datetime.now())



# loads all keys needed
load_dotenv()


# right-align arabic text
st.markdown("""
    <style>
        * {
            direction: rtl;
            text-align: right;
            font-family: 'Segoe UI', sans-serif;
        }
        textarea, input, .stTextInput > div > div > input {
            direction: rtl !important;
            text-align: right !important;
        }
    </style>
""", unsafe_allow_html=True)




if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# page config and sidebar

st.set_page_config(page_title="تصفح ملفاتك", layout="centered", page_icon="😎")
with st.sidebar:
    UploadFile = st.button("**ارفع الملف**")
    ChatWithPdf = st.button("1\. إسأل عن البنود")
    SearchByNum = st.button("2\. ابحث برقم البند")
    FieldSummary = st.button("3\. لخص البند")



# 1st page
st.title("إرفع ملفك بصيغة PDF")
uploaded_file = st.file_uploader("اختر ملف من جهازك", type=("pdf"))



if uploaded_file and not st.session_state.db_loaded:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    st.session_state.temp_path = temp_path
    load_vector_db(st.session_state.temp_path)
    st.session_state.db_loaded = True
    st.success("تم تحميل الملف بنجاح")


for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(msg.content)
    else:
        with st.chat_message("AI"):
            st.markdown(msg.content)

query = st.chat_input("اكتب سؤالك هنا")

if query:
    st.session_state.chat_history.append(HumanMessage(query))
    with st.chat_message("Human"):
        st.markdown(query)

    with st.chat_message("AI"):
        response, unique_pages = run_rag(query)

        if unique_pages:
            pages_text = ", ".join(map(str, unique_pages))
            full_response = f"{response}\n\nراجع الصفحات: {pages_text}"
        else:
            full_response = response

        st.markdown(full_response)
        st.session_state.chat_history.append(AIMessage(full_response))
