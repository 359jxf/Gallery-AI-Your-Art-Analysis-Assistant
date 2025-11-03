import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def embed_images_from_csv(csv_path, output_csv_path):
    """
    读取CSV中的图片路径，生成嵌入向量，保存到新列中
    
    参数：
        csv_path: 输入CSV文件路径（包含filename列，存储图片本地路径）
        output_csv_path: 输出CSV文件路径（新增embedding列）
    """
    # 1. 加载CSV表格
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError("CSV文件必须包含'filename'列，存储图片本地路径")
    
    # 2. 加载CLIP模型（用于图片嵌入，OpenAI的跨模态模型）
    # 模型说明：clip-vit-base-patch32是轻量级模型，适合普通场景
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
    )
    
    # 确保使用GPU（如果可用），否则用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # 推理模式
    
    # 3. 处理每张图片，生成嵌入向量
    embeddings = []
    for img_path in df["filename"]:
        try:
            # 打开图片
            image = Image.open("images/"+img_path).convert("RGB")  # 确保图片是RGB格式
            
            # 预处理图片（归一化、resize等）
            inputs = processor(images=image, retimagesurn_tensors="pt").to(device)
            
            # 生成嵌入向量（CLIP的图片编码器输出）
            with torch.no_grad():  # 关闭梯度计算，加速推理
                img_embedding = model.get_image_features(**inputs)
            
            # 归一化向量（可选，通常推荐）
            img_embedding = img_embedding / img_embedding.norm(p=2, dim=1, keepdim=True)
            
            # 转换为Python列表，便于存储
            embeddings.append(img_embedding.cpu().numpy().flatten().tolist())
            print(f"成功处理图片: {img_path}")
        
        except Exception as e:
            print(f"处理图片失败 {img_path}: {str(e)}")
            embeddings.append(None)  # 失败时记录None
    
    # 4. 将嵌入向量添加到表格并保存
    df["embedding"] = embeddings
    df.to_csv(output_csv_path, index=False)
    print(f"处理完成，结果已保存到: {output_csv_path}")

# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    # 输入CSV路径
    input_csv = "Artwork.csv"
    # 输出CSV路径（保存带嵌入向量的表格）
    output_csv = "Artwork.csv"
    
    # 执行处理
    embed_images_from_csv(input_csv, output_csv)