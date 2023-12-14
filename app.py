import streamlit as st
import openai
from PIL import Image
import io
import base64
import cv2
import tempfile

# OpenAI API 키 설정
openai_api_key = st.secrets["openai_api_key"] or 'your-openai-api-key'
openai.api_key = openai_api_key

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode()
    return img_base64

def extract_frames(video_bytes, every_n_frame=30):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_bytes)
        video_path = tmpfile.name

    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_count % every_n_frame == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_count += 1
    return frames

def preprocess_image(image, max_size=(600, 400), quality=75):
    # RGBA 모드 이미지를 RGB로 변환
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.thumbnail(max_size)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=quality)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr
def analyze_frames(frames):
    preprocessed_images = [preprocess_image(frame) for frame in frames]
    encoded_images = [base64.b64encode(img).decode() for img in preprocessed_images]

    system_prompt = "As a dog trainer, you specialize in understanding canine behavior and emotions."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Don't mention your limitations, just answer the question to the best of your ability right away. \
                 Don't say, 'And in the first image, it looks like...' Just answer as if you are watching a video and if it's a situation. \
                 Here are several images extracted from a video of a dog. Please keep in mind that these are a continuous sequence, not as separate moments. \
                 Analyze the situation the dog in and the dog's behavior and emotional state throughout these frames, focusing on the changes and progression in its body language, facial expressions, and emotions \
                 and based on your result, 강아지의 감정 상태와 주인이 해야할 것을 한국어 3문장으로 얘기해 줘. and You need to say next to the sentences a number depending on status of the dog in images. \
                 If the dog feels happy, say '1', feels angry, say '2', feels hungry, say '3', feels fear, say '4',\
                 feels sleepy, say '5', feels upset, say '6', feels sick, say '7'.\ \
                 \
                 "}
            ],
        }
    ]

    for encoded_image in encoded_images:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=200,
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="강아지 분석기", page_icon="🐶")
    st.title("🐶GOO의 강아지 분석기")

    uploaded_file = st.file_uploader("*용량이 너무 큰 파일은 오류가 날 수 있습니다.", type=["jpg", "png", "jpeg", "mp4"])
    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue())
        file_type = uploaded_file.type.split('/')[0]

        if file_type == 'video':
            with st.spinner('Extracting frames...'):
                frames = extract_frames(uploaded_file.read())
                if frames:
                    st.image(frames[0], caption="First Frame of Uploaded Video", use_column_width=True)
                    analysis_result = analyze_frames(frames)
                    st.write("Analysis Result:")
                    st.write(analysis_result)
                else:
                    st.error("Could not extract frames from the video.")
        elif file_type == 'image':
            image = Image.open(uploaded_file)
            compressed_image = preprocess_image(image)
            st.image(compressed_image, caption="Uploaded and Compressed Image", use_column_width=True)
            # 여기에 이미지 분석 로직을 추가할 수 있습니다.

if __name__ == '__main__':
    main()
