from langchain_openai import ChatOpenAI
from openai import OpenAI
from PIL import Image
import base64
import os
import io

import requests

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

    return img_format, base64_image


def call_vllm(client,GPT_MODEL,kg,user_instruction,target_image_path,image_filenames):
    user_content=[]
    # 将图片转换为base64
    target_img_format, target_base64_image = optimize_image_for_api(target_image_path)
    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{target_img_format};base64,{target_base64_image}" 
        }
    })
    note_name="Sequence of uploaded images: the filename of the No.1 image is "+target_image_path
    for idx, filename in enumerate(image_filenames):
        note_name+=", the filename of the No."+str(idx+2)+" image is "+filename
        image_path = os.path.join("images", filename)
        img_format, base64_image = optimize_image_for_api(image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{img_format};base64,{base64_image}"
            }
        })

    # 构造prompt
    prompt = f"""
    You are an expert art critic and visual composition analyst.
    Your task is to provide a detailed, *image-grounded* answer for user's question based on what you visually observe in it.

    You are also given a set of *reference evaluations* from previous similar artworks with known visual issues and quality assessments to help you understand how to evaluate, but **do not mention or reference them in your answer.**
    Here are the internal references:
    {kg}

    The user uploaded one artwork: **{target_image_path}**.  
    User’s question: **{user_instruction}**.
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

def call_gallerygpt(api_url,kg,user_instruction,target_image_path,image_filenames):
    IMAGE_PATH = [target_image_path]+[os.path.join("images", filename) for filename in image_filenames]

    note_name="Sequence of uploaded images: the filename of the No.1 image is "+target_image_path
    for idx, filename in enumerate(image_filenames):
        note_name+=", the filename of the No."+str(idx+2)+" image is "+filename

    # 构造prompt
    prompt = f"""
    You are an expert art critic and visual composition analyst.
    Your task is to provide a detailed, *image-grounded* answer for user's question based on what you visually observe in it.

    You are also given a set of *reference evaluations* from previous similar artworks with known visual issues and quality assessments to help you understand how to evaluate, but **do not mention or reference them in your answer.**
    Here are the internal references:
    {kg}

    The user uploaded one artwork: **{target_image_path}**.  
    User’s question: **{user_instruction}**.
    Additional Note: {note_name}
    """
    # print(prompt)

    file_objs = [open(path, "rb") for path in IMAGE_PATH]
    files = [("image", f) for f in file_objs]  # 多文件上传格式
    data = {
        "query": prompt
    }

    # 发送POST请求并获取结果
    try:
        response = requests.post(api_url, files=files, data=data)
        response.raise_for_status()  # 检查请求是否成功（非200状态码抛异常）
        
        result = response.json()
        res = result["response"]

    except FileNotFoundError:
        print(f"错误：图片文件 {IMAGE_PATH} 不存在")
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务，请检查服务是否启动")
    except requests.exceptions.HTTPError as e:
        print(f"错误：请求失败，状态码 {response.status_code}，详情：{e}")
    except KeyError:
        print("错误：服务返回结果格式异常，未找到 'response' 字段")
    except Exception as e:
        print(f"未知错误：{e}")
    finally:
        # 关闭图片文件句柄（避免资源泄漏）
        for f in file_objs:
            f.close()
    
    return res
