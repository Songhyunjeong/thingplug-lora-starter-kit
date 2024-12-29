# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:03:38 2024

@author: postp
"""
import os
import json
import pdfplumber
import pytesseract
from PIL import Image, UnidentifiedImageError
import pandas as pd
import openai
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API 설정
openai.api_key = ""

# 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    """
    PDF에서 텍스트를 추출하는 함수.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 이미지 추출 및 OCR 함수
def extract_images_and_text_from_pdf(pdf_path):
    """
    PDF에서 이미지 텍스트를 추출하는 함수.
    """
    images_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            if page.images:
                for img_meta in page.images:
                    print(f"Page {page_num + 1}: Found image metadata: {img_meta}")
                    img_stream = img_meta.get("stream")
                    if img_stream:
                        try:
                            raw_data = img_stream.get_data()
                            if len(raw_data) > 0:
                                pil_image = Image.open(io.BytesIO(raw_data))
                                images_text += pytesseract.image_to_string(pil_image)
                        except UnidentifiedImageError as e:
                            print(f"Unidentified image on page {page_num + 1}: {e}")
    return images_text

# 표 추출 함수
def extract_tables_from_pdf(pdf_path):
    """
    PDF에서 표 데이터를 추출하는 함수.
    """
    tables_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables_text += df.to_string() + "\n"
    return tables_text

# 임베딩 생성 함수
def generate_embedding(text):
    """
    주어진 텍스트에 대한 임베딩을 생성.
    """
    response = openai.Embedding.create(input=text[:8192], model="text-embedding-3-small")
    return response["data"][0]["embedding"]

# 컨텍스트 캐시 함수 (임베딩 포함)
def generate_or_load_embedding(pdf_path, embedding_cache_path="embedding_cache.json"):
    """
    PDF에서 텍스트를 추출하여 임베딩을 생성하거나 캐시된 임베딩을 로드.
    """
    if os.path.exists(embedding_cache_path):
        with open(embedding_cache_path, "r", encoding="utf-8") as cache_file:
            cached_embeddings = json.load(cache_file)
        if pdf_path in cached_embeddings:
            print("Cached embedding found. Using existing embedding.")
            return cached_embeddings[pdf_path]
    else:
        cached_embeddings = {}

    print("Generating new embedding.")
    text = extract_text_from_pdf(pdf_path)
    images_text = extract_images_and_text_from_pdf(pdf_path)
    tables_text = extract_tables_from_pdf(pdf_path)
    context = text + "\n\n" + images_text + "\n\n" + tables_text

    # Trim context to fit model token limit
    trimmed_context = context[:8192]
    embedding = generate_embedding(trimmed_context)
    cached_embeddings[pdf_path] = {"embedding": embedding, "context": trimmed_context}

    with open(embedding_cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(cached_embeddings, cache_file, ensure_ascii=False, indent=4)

    return cached_embeddings[pdf_path]

# 가장 유사한 컨텍스트 검색 함수
def find_most_similar_context(question, cached_embeddings):
    """
    질문과 가장 유사한 컨텍스트를 검색.
    """
    question_embedding = generate_embedding(question)
    similarities = []

    for pdf_path, data in cached_embeddings.items():
        pdf_embedding = np.array(data["embedding"]).reshape(1, -1)
        question_embedding = np.array(question_embedding).reshape(1, -1)
        similarity = cosine_similarity(pdf_embedding, question_embedding)[0][0]
        similarities.append((pdf_path, similarity))

    most_similar = max(similarities, key=lambda x: x[1])
    return cached_embeddings[most_similar[0]]["context"]

# OpenAI API를 사용한 질문-답변 처리 함수
def answer_question_with_openai(context, question):
    """
    OpenAI API를 사용하여 질문에 답변.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ],
            temperature=0.5,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "답변을 생성할 수 없습니다."

# 실행 코드
if __name__ == "__main__":
    pdf_path = "C:/Users/postp/Downloads/003.딥페이크_변조영상_데이터_구축_가이드라인.pdf"
    question = "딥페이크 변조영상 데이터셋 데이터 종류에는 어떤것이 있나요?"

    # 캐싱된 임베딩 로드 또는 생성
    embedding_cache_path = "embedding_cache.json"
    context_data = generate_or_load_embedding(pdf_path, embedding_cache_path)

    # 유사한 컨텍스트 검색
    with open(embedding_cache_path, "r", encoding="utf-8") as cache_file:
        cached_embeddings = json.load(cache_file)

    context = find_most_similar_context(question, cached_embeddings)

    # 질문에 답변 생성
    answer = answer_question_with_openai(context, question)

    print("질문:", question)
    print("답변:", answer)
