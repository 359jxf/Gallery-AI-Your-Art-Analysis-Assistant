from htbuilder.units import rem
from htbuilder import div, styles
import uuid
import os 
from dotenv import load_dotenv 

import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from PIL import Image
from embedding import process_embbeding,get_similar_file
from querygraph import queryGraph,queryImage
from vllm import call_vllm

# 配置neo4j
url="neo4j://localhost:7687"
username="neo4j"
password="apropos-sphere-violin-texas-strong-2496"
graph = Neo4jGraph(
    url=url,  
    username=username,       
    password=password,
    # enhanced_schema=True,
)

# 模型配置, 加载 API Key
load_dotenv()

deepseek_llm = ChatOpenAI(
    model_name="deepseek-chat", 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  
    openai_api_base="https://api.deepseek.com/v1", 
    streaming=True
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  
    base_url='https://openai.api2d.net/v1'
)
GPT_MODEL = "gpt-4o-mini" 

# 标签页名
st.set_page_config(page_title="Gallery AI", page_icon="🌼")

# 纯文本问答，调用图谱QA
def get_response_languageOnly(prompt):
    return queryGraph(deepseek_llm,graph,prompt,10)

# 多模态问答，查找相似图片+图谱QA+调用多模态模型 
def get_response_forImage(image_path,prompt):
    # clip编码
    emb=process_embbeding(image_path)
    # 查找图谱类似图片
    filenames=get_similar_file(url,username,password,emb,num=1)
    # 在图谱内搜集他们的信息
    kg=queryImage(deepseek_llm,graph,top_k=20,image_filenames=filenames)
    # 调用多模态大模型分析
    return call_vllm(openai_client,GPT_MODEL,kg,prompt,image_path,filenames)

def save_uploaded_image(uploaded_file):
    """保存上传的图片到本地，返回唯一文件路径"""
    # 生成唯一文件名（避免同名文件覆盖）
    save_path = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    
    # 保存图片到本地
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path

def clear_uploaded_image():
    """清空已上传的图片（重置会话状态，不删除本地文件）"""
    st.session_state.uploaded_image_path = None

# -----------------------------------------------------------------------------
# UI 绘制逻辑
# -----------------------------------------------------------------------------
st.html(div(style=styles(font_size=rem(5), line_height=1))["❀"])

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

with title_row:
    st.title(
        "Hi! I'm Gallery AI",
        anchor=False,
        width="stretch",
    )

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initial_question" not in st.session_state:
    st.session_state.initial_question = None
if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None  # 存储本地图片路径

user_just_asked_initial_question = (
    st.session_state.initial_question is not None
)

has_message_history = len(st.session_state.messages) > 0

col1, col2 = st.columns([0.4, 0.6])

with col1:
    # 图像上传区域
    uploaded_file = st.file_uploader(
        "Upload an image(JPG/PNG)", 
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    # 如果上传了新图片，自动保存并覆盖旧图片
    if uploaded_file:
        if "uploaded_image_path" in st.session_state and st.session_state.uploaded_image_path:
            clear_uploaded_image()
        save_path = save_uploaded_image(uploaded_file)
        st.session_state.uploaded_image_path = save_path

    if st.session_state.uploaded_image_path and uploaded_file:
        # 读取并预览图片
        image = Image.open(uploaded_file)
        st.image(
            image, 
            caption=f"Uploaded Image: {uploaded_file.name}", 
            output_format="auto"
        )

    # 聊天输入
    user_message = st.chat_input("Ask a question...")
    if not user_message:
        if user_just_asked_initial_question:
            user_message = st.session_state.initial_question
            st.session_state.initial_question = None  # 重置初始问题

with title_row:
    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        if "uploaded_image_path" in st.session_state:
            clear_uploaded_image()
        st.session_state.uploaded_image_path = None
        if "image_uploader" in st.session_state:
            del st.session_state["image_uploader"]
    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

with col2:
    # 显示聊天历史
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.container()  # 修复幽灵消息bug
            if message["role"] == "user" and "image_path" in message:
                st.markdown(message["content"])
                st.image(message["image_path"], width=200)
            else:
                st.markdown(message["content"])

    # 处理用户提问
    if user_message:
        # 修复 LaTeX 符号冲突
        user_message = user_message.replace("$", r"\$")

        # 显示用户消息
        with st.chat_message("user"):
            if st.session_state.uploaded_image_path:
                st.markdown(user_message)
                st.image(st.session_state.uploaded_image_path, width=200)
            else:
                st.text(user_message)
        
        # 将用户消息添加到历史记录
        user_msg = {"role": "user", "content": user_message}
        if st.session_state.uploaded_image_path:
            user_msg["image_path"] = st.session_state.uploaded_image_path
        st.session_state.messages.append(user_msg)

        # 显示助手回复
        with st.chat_message("assistant"):
            with st.spinner("Waiting..."):
                if "image_path" in user_msg:
                    # 🔥 多模态问答
                    response = get_response_forImage(
                        image_path=user_msg["image_path"], 
                        prompt=user_message
                    )
                else:
                    # 文本问答
                    response = get_response_languageOnly(user_message)

                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
        # ✅ 清空图片缓存（用户发送后立即清空上传区）
        clear_uploaded_image()