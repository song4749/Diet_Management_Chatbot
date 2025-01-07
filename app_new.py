import streamlit as st
import pandas as pd
import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import openai


def API_key():
    # 본인의 API KEY 등록
    load_dotenv()  # .env 파일에서 환경 변수 로드

    openai.api_key = os.getenv("OPENAI_API_KEY")


# def load_file_as_documents():
#     # 엑셀 파일 로드 (칼로리 데이터셋)
#     file_path = '44.음식분류_AI_데이터_영양DB.xlsx'
#     df = pd.read_excel(file_path)

#     # 음식 이름과 칼로리를 문자열로 변환하여 문서로 저장
#     food_documents = []
#     for _, row in df.iterrows():
#         food_name = row['음 식 명']
#         calories = row['에너지(kcal)']
#         rounded_calories = round(calories, 2)  # 소수점 2자리로 반올림
#         food_documents.append(Document(page_content=f"{food_name}: {rounded_calories} kcal"))

#     return food_documents


def load_file_as_txt():
    # 엑셀 파일 로드 (칼로리 데이터셋)
    file_path = '44.음식분류_AI_데이터_영양DB.xlsx'
    df = pd.read_excel(file_path)

    # 음식 이름과 칼로리를 문자열로 변환하여 문서로 저장
    food_txt = []
    for _, row in df.iterrows():
        food_name = row['음 식 명']
        calories = row['에너지(kcal)']
        rounded_calories = round(calories, 2)  # 소수점 2자리로 반올림
        food_txt.append(f"{food_name}: {rounded_calories} kcal")  # 리스트에 추가

    return food_txt


def ready_openai(food_file):
    # OpenAIEmbeddings를 사용해 텍스트를 벡터화
    embeddings = OpenAIEmbeddings()

    # Chroma: 문서를 벡터 DB에 저장
    # vectorstore = Chroma.from_documents(food_file, embeddings)
    vectorstore = Chroma.from_texts(food_file, embeddings)

    # 벡터 DB를 검색할 retriever 설정
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # k=2: 가장 관련성 높은 4개의 문서 검색


    # RetrievalQA 체인 생성 (검색된 문서로 답변 생성)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model = 'gpt-4', temperature=0.7),  # OpenAI 모델 사용
        chain_type="stuff",  # 검색된 문서를 모두 연결하여 답변 생성
        retriever=retriever
    )

    return qa_chain


# 초기화 작업을 캐시하여 한 번만 실행
@st.cache_resource
def initialize_qa_chain():
    API_key()
    food_file = load_file_as_txt()
    qa_chain = ready_openai(food_file)
    return qa_chain


def choose_goal():
    # 목표 목록
    goals = {
        "체중 감량": "체중 감량을 하고 싶으시군요! 아래에 체중 감량과 관련된 자세한 질문을 작성해주세요.",
        "체중 증가": "체중 증가를 하고 싶으시군요! 아래에 체중 증가와 관련된 자세한 질문을 작성해주세요.",
        "피부 관리": "피부 관리를 하고 싶으시군요! 아래에 피부 관리와 관련된 자세한 질문을 작성해주세요.",
        "근육 증강": "근육량을 늘리고 싶으시군요! 아래에 근육 증강과 관련된 자세한 질문을 작성해주세요"
    }

    # 목표 선택 버튼
    if st.session_state.selected_goal is None:
        for goal in goals:
            if st.button(goal):
                st.session_state.selected_goal = goal
                st.write(goals[goal])

    # 질문창 생성
    if st.session_state.selected_goal:
        st.session_state.question = st.text_input(
            f"{st.session_state.selected_goal}과 관련된 질문을 입력하세요:"
        )
    

# QA Chain 초기화
qa_chain = initialize_qa_chain()

st.title("식단 관리 목표")

# 선택된 목표와 질문 상태 관리
if "selected_goal" not in st.session_state:
    st.session_state.selected_goal = None
if "question" not in st.session_state:
    st.session_state.question = None

choose_goal()

if st.session_state.question:
    prompt = f"""
                목표에 따른 식단관리 질문입니다.
                질문: '{st.session_state.question}'에서 음식 이름을 정확히 찾아 그 칼로리를 알려주세요.
                또한 목표{st.session_state.selected_goal}와 관련된 조언을 해주세요.
                답변을 한국어로, 항상 답변 맨 앞에 '회원님!'으로 시작해 주세요.
            """
    
    answer = qa_chain.run(prompt)
    st.write(answer)