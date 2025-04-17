import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai
import shutil
import difflib

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# -------------------- API Key 로딩 --------------------
def API_key():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------- 음식 리스트 로딩 --------------------
@st.cache_data
def get_food_name_list():
    file_path = '44.음식분류_AI_데이터_영양DB.xlsx'
    df = pd.read_excel(file_path)
    return df['음 식 명'].dropna().unique().tolist()

# -------------------- 음식 문서 로딩 --------------------
def load_file_as_documents():
    file_path = '44.음식분류_AI_데이터_영양DB.xlsx'
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['음 식 명', '에너지(kcal)'])
    df = df.drop_duplicates(subset=['음 식 명'])

    food_documents = []
    for _, row in df.iterrows():
        food_name = row['음 식 명']
        nutrients = {
            '중량': f"{row['중량(g)']}g",
            '에너지': f"{round(row['에너지(kcal)'], 2)} kcal",
            '탄수화물': f"{row['탄수화물(g)']}g",
            '당류': f"{row['당류(g)']}g",
            '지방': f"{row['지방(g)']}g",
            '단백질': f"{row['단백질(g)']}g",
            '칼슘': f"{row['칼슘(mg)']}mg",
            '인': f"{row['인(mg)']}mg",
            '나트륨': f"{row['나트륨(mg)']}mg",
            '칼륨': f"{row['칼륨(mg)']}mg",
            '마그네슘': f"{row['마그네슘(mg)']}mg",
            '철': f"{row['철(mg)']}mg",
            '아연': f"{row['아연(mg)']}mg",
            '콜레스테롤': f"{row['콜레스테롤(mg)']}mg",
            '트랜스지방': f"{row['트랜스지방(g)']}g",
        }
        content = f"{food_name}:\n" + "\n".join([f"- {key}: {val}" for key, val in nutrients.items()])
        food_documents.append(Document(page_content=content))

    return food_documents

# -------------------- 벡터 DB 구성 --------------------
def create_retriever(food_documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        food_documents,
        embedding=embeddings,
        persist_directory="./Vector_DB"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------- 음식명 추출 --------------------
def clean_text(text):
    return text.replace(" ", "").lower()

def extract_foods_robust(question, food_list):
    q = clean_text(question)
    exact = [food for food in food_list if clean_text(food) in q]
    if exact:
        return exact
    close = difflib.get_close_matches(q, [clean_text(f) for f in food_list], n=2, cutoff=0.6)
    recovered = [food for food in food_list if clean_text(food) in close]
    return recovered

# -------------------- 목표 선택 UI --------------------
def choose_goal():
    goals = {
        "체중 감량": "체중 감량을 하고 싶으시군요! 아래에 체중 감량과 관련된 자세한 질문을 작성해주세요.",
        "체중 증가": "체중 증가를 하고 싶으시군요! 아래에 체중 증가와 관련된 자세한 질문을 작성해주세요.",
        "피부 관리": "피부 관리를 하고 싶으시군요! 아래에 피부 관리와 관련된 자세한 질문을 작성해주세요.",
        "근육 증강": "근육량을 늘리고 싶으시군요! 아래에 근육 증강과 관련된 자세한 질문을 작성해주세요"
    }

    st.markdown("### 🎯 목표를 선택하세요:")
    left, right = st.columns(2)

    with left:
        for goal in goals:
            if st.button(goal):
                st.session_state.selected_goal = goal
        if st.session_state.selected_goal:
            st.success(goals[st.session_state.selected_goal])

    with right:
        goal = st.session_state.selected_goal
        if goal:
            st.image(f"streamlit_img/{goal.replace(' ', '')}.png")

    if st.session_state.selected_goal:
        st.session_state.question = st.text_input(f"{st.session_state.selected_goal}과 관련된 질문을 입력하세요:")

# -------------------- 프롬프트 생성 --------------------
def generate_prompt(goal, question):
    base = f"회원님의 건강 목표는 '{goal}'입니다.\n"
    instruction_map = {
        "체중 감량": "에너지(kcal), 지방, 당류, 나트륨을 중점적으로 확인하고 칼로리 제한과 당/지방 조절 관점에서 평가해주세요. 너무 고열량 음식은 대체 식품도 추천해주세요.",
        "체중 증가": "에너지(kcal), 탄수화물, 단백질 중심으로 분석해주세요. 체중 증가에 도움이 되는 영양소가 충분한지 판단하고 부족할 경우 보충 식품을 추천해주세요.",
        "피부 관리": "당류, 나트륨은 낮을수록 좋고 아연, 철, 칼륨, 단백질은 피부에 도움이 됩니다. 이런 항목들을 바탕으로 피부 건강에 얼마나 적절한 식단인지 평가해주세요.",
        "근육 증강": "단백질(g), 에너지(kcal), 마그네슘, 철 성분 위주로 분석해주세요. 근육 합성에 도움이 되는 음식인지 확인하고, 부족한 경우 대체 식품을 추천해주세요."
    }
    instructions = instruction_map.get(goal, "")
    return (
        base +
        "아래 질문을 분석하여 음식 이름을 찾아 관련 영양 정보를 활용하고,\n"
        f"목표에 맞는 식단 조언을 작성해주세요.\n\n질문: \"{question}\"\n\n{instructions}\n"
        "식단 조언 말고도 추가적인 질문이 있다면 답변해주세요.\n"
        "답변은 한국어로, 항상 맨 앞에 '회원님!'으로 시작해주세요."
    )

# -------------------- 초기화 --------------------
@st.cache_resource
def initialize_retriever():
    API_key()
    if os.path.exists("./Vector_DB"):
        shutil.rmtree("./Vector_DB")
    food_documents = load_file_as_documents()
    return create_retriever(food_documents)

# -------------------- 실행 --------------------
retriever = initialize_retriever()
food_name_list = get_food_name_list()

st.title("식단 관리 목표")

if "selected_goal" not in st.session_state:
    st.session_state.selected_goal = None
if "question" not in st.session_state:
    st.session_state.question = None

choose_goal()

if st.session_state.question:
    matched_foods = extract_foods_robust(st.session_state.question, food_name_list)

    if matched_foods:
        query = " ".join(matched_foods)
        docs = retriever.get_relevant_documents(query=query)
    else:
        st.warning("질문에서 정확한 음식명을 찾지 못했습니다. 전체 질문으로 검색합니다.")
        docs = retriever.get_relevant_documents(query=st.session_state.question)

    unique_docs = list({doc.page_content: doc for doc in docs}.values())
    prompt = generate_prompt(goal=st.session_state.selected_goal, question=st.session_state.question)
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    full_prompt = f"{prompt}\n\n아래는 참고할 음식 정보입니다:\n{context}"

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    response = llm.invoke(full_prompt)
    st.write(response.content)
