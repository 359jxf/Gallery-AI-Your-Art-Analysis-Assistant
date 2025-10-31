from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_neo4j import Neo4jVector
from langchain.embeddings.base import Embeddings

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
device = "cpu"
model.to(device)
model.eval()  

# 为clip定义嵌入类便于进行图谱查询
class CLIPEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        # 新增：预定义测试向量（维度必须和你的图片embedding一致，如CLIP的512维）
        self.test_dim = 512  # 改成你实际的embedding维度（如512、2048）
        self.test_vector = [0.0] * self.test_dim

    def embed_query(self, vector):
        # 新增：处理LangChain初始化时的测试字符串（如"foo"）
        if isinstance(vector, str):
            return self.test_vector
        # 原有逻辑：处理图片embedding（numpy数组）
        return vector.tolist()

    def embed_documents(self, vectors):
        # 批量处理时，同样先判断类型，再调用embed_query
        return [self.embed_query(v) for v in vectors]

def process_embbeding(img_path):
    try:
        image = Image.open(img_path).convert("RGB") 
        
        # 预处理图片（归一化、resize等）
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # 生成嵌入向量（CLIP的图片编码器输出）
        with torch.no_grad():  # 关闭梯度计算，加速推理
            img_embedding = model.get_image_features(**inputs)
        
        # 归一化向量（可选，通常推荐）
        img_embedding = img_embedding / img_embedding.norm(p=2, dim=1, keepdim=True)
        
        # 转换为Python列表，便于存储
        emb=img_embedding.cpu().numpy().flatten().tolist()
        print(f"成功处理图片: {img_path}")

    except Exception as e:
        print(f"处理图片失败 {img_path}: {str(e)}")
        emb=None 
    return emb

def get_similar_file(url,username,password,emb,num=2):
    # 从已有节点表中初始化 vector store
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

    # 相似度检索
    similar_docs = vectorestore.similarity_search_by_vector(emb, k=num,query="")

    filenames=[]
    for doc in similar_docs:
        filename = doc.page_content.strip().split('filename:')[1].strip()
        if filename: 
            # print(filename)
            filenames.append(filename)
    return filenames

if __name__=='__main__':
    url="neo4j://localhost:7687"
    username="neo4j"
    password="apropos-sphere-violin-texas-strong-2496"

    emb=process_embbeding("test.jpg")
    num=2
    filenames=get_similar_file(url,username,password,emb,num)
    print(filenames)