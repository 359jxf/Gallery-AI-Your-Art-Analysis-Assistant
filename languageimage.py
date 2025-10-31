import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph, Neo4jVector
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv 
from openai import OpenAI
from PIL import Image
import base64
import os
import io
import time

# 加载环境变量
load_dotenv()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  
    base_url='https://openai.api2d.net/v1'
)

# 初始化CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cpu"
model.to(device)
model.eval()  

# Neo4j连接信息
url="neo4j://localhost:7687"
username="neo4j"
password="apropos-sphere-violin-texas-strong-2496"

# 自定义CLIP嵌入类，neo4j向量存储需要
class CLIPEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        self.test_dim = 512  # 改成的embedding维度,clip是512维
        self.test_vector = [0.0] * self.test_dim

    def embed_query(self, vector):
        if isinstance(vector, str):
            return self.test_vector
        return vector.tolist()

    def embed_documents(self, vectors):
        return [self.embed_query(v) for v in vectors]

user_content = []
# 图片转base64并存入消息内容
def optimize_image_for_api(image_path, max_size=(2048, 2048), quality=85):
    """优化图片以减少token消耗"""
    img = Image.open(image_path)
    
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = rgb_img
    
    buffer = io.BytesIO()
    img_format = image_path.split('.')[-1].upper()
    if img_format == 'JPG':
        img_format = 'JPEG' 
    img.save(buffer, format=img_format, quality=quality, optimize=True)
    buffer.seek(0)

    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{img_format};base64,{base64_image}" 
        }
    })

# 1. 处理目标图片，生成base64
target_image_path = "test.jpg"
optimize_image_for_api(target_image_path)

# 2. 生成目标图片的embedding
try:
    image = Image.open(target_image_path).convert("RGB") 
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():  
        img_embedding = model.get_image_features(**inputs)
    
    img_embedding = img_embedding / img_embedding.norm(p=2, dim=1, keepdim=True)
    
    emb=img_embedding.cpu().numpy().flatten().tolist()
    print(f"成功处理图片: {target_image_path}")

except Exception as e:
    print(f"处理图片失败 {target_image_path}: {str(e)}")
    emb=None 

# 3. 相似度检索
clip_embedding = CLIPEmbeddings(model=None)
vectorestore = Neo4jVector.from_existing_graph(
    embedding=clip_embedding,
    url=url,
    username=username,
    password=password,
    node_label="Artwork",
    embedding_node_property="embedding",
    text_node_properties=["filename"]
)

similar_docs = vectorestore.similarity_search_by_vector(emb, k=2,query="")

# 4. 提取相似图片的文件名并转换为base64加入消息
image_filenames=[]
for doc in similar_docs:
    filename = doc.page_content.strip().split('filename:')[1].strip()
    if filename: 
        # print(filename)
        image_filenames.append(filename)

note_name="Sequence of uploaded images: the filename of the No.1 image is "+target_image_path
for idx, filename in enumerate(image_filenames):
    note_name+=", the filename of the No."+str(idx+2)+" image is "+filename
    image_path = os.path.join("images", filename)
    optimize_image_for_api(image_path)

# 5. 对这些节点做图谱关系查询（deepseek）
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password,
    # enhanced_schema=True,
)

graph.refresh_schema()

deepseek_llm = ChatOpenAI(
    model_name="deepseek-chat", 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  
    openai_api_base="https://api.deepseek.com/v1", 
)

chain = GraphCypherQAChain.from_llm(
    llm=deepseek_llm,
    graph=graph,
    verbose=True, 
    top_k=20,
    allow_dangerous_requests=True,
)

if image_filenames:
    filename_str = ", ".join(image_filenames)
    query = f"""
    Below are the filenames of the relevant works I want to query: {filename_str}. 
    You need to query the scores of dimensions and the reasons in all HAS_LEVEL relationships they are involved in, 
    and then return the results in the following JSON format:
    [ 
        {{
            "filename":"xxxx",
            "dimension": "overall", 
            "level": "Good", 
            "reason": "xxx"
        }},
        {{
            "filename":"xxxx",
            "dimension": "color", 
            "level": "Good", 
            "reason": "xxx"
        }},
        ...
    ]
    Note: You must only return JSON (do not include any extra text, comments, or formatting).
    """
    res = chain.invoke({"query": query})
    kg=res['result']
else:
    print("No similar artworks found.Closed...")
    exit()

# 6. 构建最终prompt，调用GPT-4o进行分析
user_instruction="How do you think of my artwork and your suggestions?"

prompt = f"""
You are an expert art critic and visual composition analyst.

The user uploaded one artwork: **{target_image_path}**.  
Your task is to provide a detailed, *image-grounded* critique and improvement suggestions based on what you visually observe in it.

You are also given a set of *reference evaluations* from previous similar artworks with known visual issues and quality assessments to help you understand how to evaluate, but **do not mention or reference them in your answer.**
Here are the internal references:
{kg}

Your response should have the following structure:

**Visual Observation:**  
(A concrete description of what you see in the image)

**Evaluation:**  
(A precise critique reflecting the technical and expressive strengths and weaknesses.)

**Improvement Suggestions:**  
(Detailed, actionable advice for improvement)

---

User’s question: “{user_instruction}”
Additional Note: {note_name}
"""
# print(prompt)

user_content.append({
        "type": "text",
        "text": prompt
    })

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": user_content  
        }
    ],
    max_tokens=400,
)

print(response.choices[0].message.content)