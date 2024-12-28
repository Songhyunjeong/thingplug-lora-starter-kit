# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:32:18 2024

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

# 컨텍스트 캐시 함수
def generate_or_load_context(pdf_path, cache_path="context_cache.json"):
    """
    PDF에서 텍스트, 이미지, 표를 추출하여 컨텍스트를 생성하거나 캐시된 컨텍스트를 로드.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            cached_contexts = json.load(cache_file)
        if pdf_path in cached_contexts:
            print("Cached context found. Using existing context.")
            return cached_contexts[pdf_path]
    else:
        cached_contexts = {}

    print("Generating new context.")
    text = extract_text_from_pdf(pdf_path)
    images_text = extract_images_and_text_from_pdf(pdf_path)
    tables_text = extract_tables_from_pdf(pdf_path)
    context = text + "\n\n" + images_text + "\n\n" + tables_text

    cached_contexts[pdf_path] = context
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(cached_contexts, cache_file, ensure_ascii=False, indent=4)

    return context

# OpenAI API를 사용한 질문-답변 처리 함수
def answer_question_with_openai(context, question):
    """
    OpenAI API를 사용하여 질문에 답변.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":"You are a helpful assistant. first,answer based only on the provided context. second, you can use external knowledge and answer specify if when the answer is not based on provided context.'"},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ],
            temperature=0.5,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "답변을 생성할 수 없습니다."

# 긴 컨텍스트 처리 함수
def process_long_context_with_openai(context, question, max_length=1500):
    """
    긴 컨텍스트를 분할하여 OpenAI API로 질문에 답변.
    """
    words = context.split()
    chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    answers = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}: {len(chunk.split())} tokens")
        try:
            answer = answer_question_with_openai(chunk, question)
            answers.append(answer)
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
    return "\n".join(answers)

# 실행 코드
if __name__ == "__main__":
    pdf_path = "C:/Users/postp/Downloads/003.딥페이크_변조영상_데이터_구축_가이드라인.pdf"
    question = "딥페이크 변조영상 확인 웹사이트를 만들기 위한 제안서 작성해줘"

    context = generate_or_load_context(pdf_path)
    if len(context.split()) > 1500:
        print("Context is too long. Splitting for processing.")
        answer = process_long_context_with_openai(context, question)
    else:
        answer = answer_question_with_openai(context, question)

    print("질문:", question)
    print("답변:", answer)

