from langchain_openai import ChatOpenAI
from openai import OpenAI
from PIL import Image
import base64
import os
import io

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

def call_vllm(client,GPT_MODEL,kg,user_instruction,target_image_path,image_filenames):
    # 将图片转换为base64
    optimize_image_for_api(target_image_path)
    note_name="Sequence of uploaded images: the filename of the No.1 image is "+target_image_path
    for idx, filename in enumerate(image_filenames):
        note_name+=", the filename of the No."+str(idx+2)+" image is "+filename
        image_path = os.path.join("images", filename)
        optimize_image_for_api(image_path)

    # 构造prompt
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

    # 调用模型
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": user_content  
            }
        ],
        max_tokens=400,
    )

    res=response.choices[0].message.content
    return res