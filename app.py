import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit_mic_recorder import speech_to_text  # استيراد وظيفة تحويل الصوت إلى نص

# تهيئة مفاتيح API
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# تهيئة الشريط الجانبي
with st.sidebar:
    # التحقق من مفاتيح API وتهيئة المكونات إذا كانت صالحة
    if groq_api_key and google_api_key:
        # تعيين مفتاح Google API كمتغير بيئي
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # تهيئة ChatGroq بمفتاح API المقدم
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # تعريف قالب المحادثة
        prompt = ChatPromptTemplate.from_template(
            """
            أجب على الأسئلة بناءً على السياق المقدم فقط.
            يرجى تقديم الإجابة الأكثر دقة بناءً على السؤال.
            <context>
            {context}
            <context>
            الأسئلة: {input}
            """
        )

        # تحميل التضميدات الموجودة من الملفات
        if "vectors" not in st.session_state:
            with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار."):
                # تهيئة التضميدات
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # تحميل فهرس FAISS الموجود مع إزالة التسلسل الآمن
                embeddings_path = "embeddings"  # المسار إلى مجلد التضميدات
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True  # استخدم فقط إذا كنت تثق في مصدر التضميدات
                    )
                    st.sidebar.write("تم تحميل التضميدات بنجاح 🎉")
                except Exception as e:
                    st.error(f"خطأ في تحميل التضميدات: {str(e)}")
                    st.session_state.vectors = None
    else:
        st.error("الرجاء إدخال مفتاحي API للمتابعة.")

# منطقة الواجهة الرئيسية
st.title("الدردشة مع PDF 🗨️")

# تهيئة حالة الجلسة للرسائل إذا لم تكن موجودة بالفعل
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض سجل المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# زر المايكروفون بجانب مكان كتابة السؤال
st.markdown(
    """
    <style>
    .stButton button {
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }
    .stButton button:hover {
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# إنشاء حاوية للإدخال وزر المايكروفون
col1, col2 = st.columns([6, 1])

with col1:
    # حقل إدخال النص
    human_input = st.chat_input("اكتب سؤالك هنا...")

with col2:
    # زر المايكروفون
    voice_input = speech_to_text(
        start_prompt="🎤 اضغط للتحدث",
        stop_prompt="⏹️ توقف",
        language="ar",  # اللغة (ar للعربية، en للإنجليزية)
        use_container_width=True,
        just_once=True,
    )

# إذا تم اكتشاف إدخال صوتي، قم بمعالجته
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
    with st.chat_message("user"):
        st.markdown(voice_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # إنشاء وتكوين سلسلة المستندات والمسترجعات
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # الحصول على الرد من المساعد
        response = retrieval_chain.invoke({"input": voice_input})
        assistant_response = response["answer"]

        # إضافة وعرض رد المساعد
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # عرض المعلومات الداعمة (أرقام الصفحات فقط)
        with st.expander("المعلومات الداعمة"):
            if "context" in response:
                # استخراج أرقام الصفحات الفريدة من السياق
                page_numbers = set()
                for doc in response["context"]:
                    page_number = doc.metadata.get("page", "غير معروف")
                    if page_number != "غير معروف" and str(page_number).isdigit():  # التحقق مما إذا كان رقم الصفحة صالحًا
                        page_numbers.add(int(page_number))  # التحويل إلى عدد صحيح للفرز

                # عرض أرقام الصفحات
                if page_numbers:
                    page_numbers_str = ", ".join(map(str, sorted(page_numbers)))  # فرز الصفحات رقميًا وتحويلها إلى نص
                    st.write(f"هذه الإجابة مأخوذة من الصفحات: {page_numbers_str}")
                else:
                    st.write("لا توجد أرقام صفحات صالحة في السياق.")
            else:
                st.write("لا يوجد سياق متاح.")
    else:
        # تنبيه المستخدم بضرورة تحميل التضميدات
        assistant_response = (
            "لم يتم تحميل التضميدات. الرجاء التحقق من مسار التضميدات."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# إذا تم إدخال نص، قم بمعالجته
if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # إنشاء وتكوين سلسلة المستندات والمسترجعات
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # الحصول على الرد من المساعد
        response = retrieval_chain.invoke({"input": human_input})
        assistant_response = response["answer"]

        # إضافة وعرض رد المساعد
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # عرض المعلومات الداعمة (أرقام الصفحات فقط)
        with st.expander("المعلومات الداعمة"):
            if "context" in response:
                # استخراج أرقام الصفحات الفريدة من السياق
                page_numbers = set()
                for doc in response["context"]:
                    page_number = doc.metadata.get("page", "غير معروف")
                    if page_number != "غير معروف" and str(page_number).isdigit():  # التحقق مما إذا كان رقم الصفحة صالحًا
                        page_numbers.add(int(page_number))  # التحويل إلى عدد صحيح للفرز

                # عرض أرقام الصفحات
                if page_numbers:
                    page_numbers_str = ", ".join(map(str, sorted(page_numbers)))  # فرز الصفحات رقميًا وتحويلها إلى نص
                    st.write(f"هذه الإجابة مأخوذة من الصفحات: {page_numbers_str}")
                else:
                    st.write("لا توجد أرقام صفحات صالحة في السياق.")
            else:
                st.write("لا يوجد سياق متاح.")
    else:
        # تنبيه المستخدم بضرورة تحميل التضميدات
        assistant_response = (
            "لم يتم تحميل التضميدات. الرجاء التحقق من مسار التضميدات."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
