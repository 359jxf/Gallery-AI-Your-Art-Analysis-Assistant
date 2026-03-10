import requests

# 1. 配置服务地址和请求参数
API_URL = "http://127.0.0.1:8001/process"  
IMAGE_PATH = ['test2.jpg', 'images/de6c7f952d564077a5e713653fc7db60.jpg', 'images/2f983ebf846541ceabdf2145133d2e8d.png'] 
QUERY = """
    You are an expert art critic and visual composition analyst.
    Your task is to provide a detailed, *image-grounded* answer for user's question based on what you visually observe in it.

    You are also given a set of *reference evaluations* from previous similar artworks with known visual issues and quality assessments to help you understand how to evaluate, but **do not mention or reference them in your answer.**
    Here are the internal references:
[
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "theme_and_logic",
        "level": "Good",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "sense_of_order",
        "level": "Average",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "overall",
        "level": "Average",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "mood",
        "level": "Average",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "light_and_shadow",
        "level": "Average",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "space_and_perspective",
        "level": "Average",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "layout_and_composition",
        "level": "Average",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "details_and_texture",
        "level": "Good",
        "reason": ""
    },
    {
        "filename": "2f983ebf846541ceabdf2145133d2e8d.png",
        "dimension": "color",
        "level": "Good",
        "reason": ""
    },
    {
        "filename": "de6c7f952d564077a5e713653fc7db60.jpg",
        "dimension": "layout_and_composition",
        "level": "Below Average",
        "reason": "The overall layout of the screen is too average"
    },
    {
        "filename": "de6c7f952d564077a5e713653fc7db60.jpg",
        "dimension": "overall",
        "level": "Below Average",
        "reason": "The screen is very simple"
    },
    {
        "filename": "de6c7f952d564077a5e713653fc7db60.jpg",
        "dimension": "sense_of_order",
        "level": "Below Average",
        "reason": ""
    },
    {
        "filename": "de6c7f952d564077a5e713653fc7db60.jpg",
        "dimension": "details_and_texture",
        "level": "Below Average",
        "reason": "stiff texture, slightly insufficient attention to detail"
    },
    {
        "filename": "de6c7f952d564077a5e713653fc7db60.jpg",
        "dimension": "mood",
        "level": "Below Average",
        "reason": ""
    },
    {
        "filename": "de6c7f952d564077a5e713653fc7db60.jpg",
        "dimension": "theme_and_logic",
        "level": "Average",
        "reason": ""
    },
    {
        "filename": "de6c7f952d564077a5e713653fc7db60.jpg",
        "dimension": "creativity",
        "level": "Below Average",
        "reason": ""
    }
]

    The user uploaded one artwork: test2.jpg.  
    User’s question: Is the subject of this painting clearly defined?.
    Additional Note: Sequence of uploaded images: the filename of the No.1 image is test2.jpg, the filename of the No.2 image is de6c7f952d564077a5e713653fc7db60.jpg, the filename of the No.3 image is 2f983ebf846541ceabdf2145133d2e8d.png
    """


# 2. 构造请求（包含图片文件和文本参数）
file_objs = [open(path, "rb") for path in IMAGE_PATH]
files = [("image", f) for f in file_objs]  # 多文件上传格式
data = {
    "query": QUERY
}

# 3. 发送POST请求并获取结果
try:
    response = requests.post(API_URL, files=files, data=data)
    response.raise_for_status()  # 检查请求是否成功（非200状态码抛异常）
    
    # 4. 解析返回的JSON结果
    result = response.json()
    print("模型回答：", result["response"])

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