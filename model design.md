# 模型设计
对于名画，如果只是用知识图谱或者微调保留艺术信息，那每幅画真正保存的信息很少，实际的效果不如直接让大模型联网搜索这幅画相关艺术评论来的细致和多元，针对性强。

知识图谱可以换一种思路使用，在立项的时候有位老师说：这个模型可以教人怎么画画吗？所以我们可以设计一种提供审美性评论和修改建议的模型，让知识图谱成为一个美术老师。这样的模型也可以对名作和业余作品发表艺术评论，还能提出新的建议。

**总结:** 联网搜索api负责品鉴艺术品，知识图谱负责教人怎么画和评论个人作品。

## 能力维度
| 审美技能类别                            | 作用               | 对模型的输出能力影响             |
| --------------------------------- | ---------------- | ---------------------- |
| **1.风格（Style）**                   | 识别是谁的风格、属于何种流派   | 给出文化背景与风格方向建议          |
| **2.构图（Composition）**             | 决定画面结构是否稳定、舒服    | 给出改动画面元素布局建议           |
| **3.色彩（Color Theory）**            | 决定气氛、情绪、节奏与统一性   | 可给出“饱和降低/明度拉开/冷暖对比”建议  |
| **4.光影（Lighting & Contrast）**     | 决定空间感、立体感、戏剧性    | 给出“增强暗部对比/突出焦点”建议      |
| **5.笔触与材质（Brushwork / Texture）**  | 决定质感与观看节奏        | 给出“笔触更自由/加干湿对比/纹理分层”建议 |
| **6.空间与透视（Depth & Perspective）**  | 决定是否真实、有纵深、有沉浸感  | 给出“前中后景分明/消失点校正”建议     |
| **7.主题与叙事（Theme & Storytelling）** | 决定画是否有情感、意图、信息表达 | 输出“主题不突出/叙事断裂/情感增强策略”  |
| **8.视觉节奏与统一性（Rhythm & Harmony）**  | 决定画面是否耐看、松弛、有韵律  | 可给出“重复元素弱化/呼应关系增强”建议   |

## 示例回答流程（Prompt）
🧩 1. 总体审美评价（Harmony / Mood / Theme）

🧩 2. 分维度打分（构图：7/10，色彩：9/10，光影：5/10…）

🧩 3. 找问题（左下角抢戏、色彩噪点、笔触节奏割裂）

🧩 4. 给修改策略（移动主体 / 降低亮度 / 加透视引导线）

## 知识图谱设计细节
### 数据集作品选择

| 类型                       | 内容                                                 | 建议占比    |
| ------------------------ | -------------------------------------------------- | ------- |
| **大师作品 / 经典艺术品**         | 结构严谨、构图成熟、色彩经典、具有明确审美标准的作品                         | **30%** |
| **优秀但非大师的作品**            | 获奖作品 / 画师高质量作品 / stable diffusion / midjourney 的佳作 | **30%** |
| **普通 / 业余作品** | 比较混乱、风格不成熟、有明显缺陷                                   | **40%** |

✅ **为什么不是大师作品过多？**

如果“大师作品”比例太高（例如 70%），模型会出现这些问题：

| 现象         | 问题                 |
| ---------- | ------------------ |
| 点评趋向精英化    | 任何作品都说“不够经典”       |
| 过度模板化      | 只推色彩三角构图、黄金分割等固定范式 |
| 对真实用户作品不友好 | 输出高高在上、不具可操作性      |

✅ **为什么要保留 30% 大师作品？**

* 模型要有一个较高的审美上限（benchmark）
* 大师作品能作为 **知识锚点（anchor）**，让模型懂“什么叫好”
* 以后才能在点评时说出：“这幅画的问题是A，而大师常用的解决方式是B”

✅ **中间那 30% 又为何重要？**

它承担了一个至关重要的作用：**避免审美二极化**。

如果只有经典（完美），业余（混乱），那模型会变得非黑即白，而现代审美很多时候是灰度和多样的。例如：

* 单点色彩冲突 ≠ 不好的画
* 表达强烈而失稳 ≠ 缺陷，有时是风格

优质但非大师的作品可以把模型的审美曲线变得**连续**而非**断层式**。

✅ **真实数据 vs ai生成**

ai生成艺术评论的话，基本都是鼓励型，打分都高，无法看出区别。真实评论才能学习人类偏好

### 知识图谱应该有哪些点和边
**节点（Nodes）建议 4 大类：**

| 类别           | 示例                            |
| ------------ | ----------------------------- |
| **审美维度节点**   | Composition, Color, Lighting… |
| **问题模式节点**   | “拥挤”“焦点不清”“冷暖混乱”“对比不足”        |
| **修改策略节点**   | “增加留白”“提高饱和度”“增强对比”“调整透视”     |
| **风格节点（可选）** | 梵高、莫奈、印象派…                    |

**边（Edges）建议：**

| 边类型                             | 示例                    | 用途       |
| ------------------------------- | --------------------- | -------- |
| *dimension_has_problem*         | Composition → “元素拥挤”  | 定位问题来源   |
| *problem_solved_by*             | “元素拥挤” → “增加留白”       | 让图谱能反推建议 |
| *dimension_style_related*       | Color → Impressionism | 风格推理     |
| *strategy_belongs_to_dimension* | “增强对比” → Lighting     | 让建议有维度依据 |

> 目标是：模型遇到问题节点 → 沿图走到“建议节点” → 得到合理修改建议

### 问答数据集如何转成知识图谱

你生成的数据 JSON 已经是结构化的，只需：

| 步骤     | 动作                                      |
| ------ | --------------------------------------- |
| Step 1 | 解析每条问答 JSON                             |
| Step 2 | 抽取每个维度的 `problem`、`advice` 字段           |
| Step 3 | 建 “dimension → problem → advice” 的链式三元组 |
| Step 4 | 写入 Neo4j / GraphDB / or PyG HeteroData  |

三元组示例（最终图谱用的就是这些）：

```
(Composition) -[dimension_has_problem]-> (元素拥挤)
(元素拥挤) -[problem_solved_by]-> (增加留白)
(增加留白) -[strategy_belongs_to_dimension]-> (Composition)
```

### 在这个领域里，知识图谱相比直接微调的意义是什么

| 方案             | 优势              | 缺点            |
| -------------- | --------------- | ------------- |
| **直接微调（SFT）**  | 语言自然、收敛快        | 逻辑不稳定、理由缺乏一致性 |
| **加入知识图谱**     | 结构化推理、点评一致、风格可控 | 成本高但结果更像“老师”  |
| **两者结合（推荐路线）** | 既懂规则又会说话        | ✅ 最符合你的目标     |

核心结论：

> ⚠️如果你要的是“审美逻辑稳定、可控、讲道理、有依据”的点评 → **知识图谱很有意义，并远强于纯 SFT**

---

### 怎么用知识图谱辅助大模型生成

| 路线                             | 原理                                        | 适合你吗    |
| ------------------------------ | ----------------------------------------- | ------- |
| **Retrieval方式（推荐）**            | LLM点评前 → 根据检测到的问题 → 从图谱检索到相关建议 → 拼进prompt | ✅简单稳健   |
| **Graph Neural Network方式（进阶）** | 用 GNN 学图 → 输出 embedding → 作为 LLM输入        | ✅更强，但复杂 |

## 

## Cypher语句

创建节点

```cypher
LOAD CSV WITH HEADERS FROM 'file:///Artwork.csv' AS row CREATE (Artwork:Artwork {id: row.id, filename: row.filename, embedding: apoc.convert.fromJsonList(row.embedding)}) RETURN Artwork;
LOAD CSV WITH HEADERS FROM 'file:///Artstyle.csv' AS row CREATE (Artstyle:Artstyle {id: row.id}) RETURN Artstyle;
LOAD CSV WITH HEADERS FROM 'file:///Category.csv' AS row CREATE (Category:Category {id: row.id}) RETURN Category;
LOAD CSV WITH HEADERS FROM 'file:///Subject.csv' AS row CREATE (Subject:Subject {id: row.id}) RETURN Subject;
LOAD CSV WITH HEADERS FROM 'file:///AestheticDimension.csv' AS row CREATE (Dimension:Dimension {id: row.id}) RETURN Dimension;
```

查询节点种类及个数

```
MATCH (n)
UNWIND labels(n) AS label  
RETURN label AS node_type, count(n) AS node_count
ORDER BY node_count DESC;
```

可视化图谱

```
MATCH (n) RETURN n
```

删除所有节点

```
match (n) detach delete n
```

删除所有关系

```
MATCH ()-[r]->()
DELETE r;
```

查询关系种类

```
MATCH ()-[r]-()
RETURN DISTINCT type(r) AS relationship_types
ORDER BY relationship_types;
```

查询关系属性

```
MATCH (a:Artwork {id: "1"})-[r:HAS_LEVEL]->(b:Dimension {id: "theme_and_logic"})
RETURN 
  a.id AS artwork_id,   
  type(r) AS rel_type,   
  b.id AS dimension_id,   
  properties(r) AS rel_properties; 
```

创建关系

```cypher
LOAD CSV WITH HEADERS FROM 'file:///Artwork_CATEGORY.csv' AS row
MATCH (a:Artwork {id: row.artwork})
MATCH (b:Category {id: row.category})
MERGE (a)-[:BELONGS_TO_CATEGORY]->(b);
LOAD CSV WITH HEADERS FROM 'file:///Artwork_STYLE.csv' AS row
MATCH (a:Artwork {id: row.artwork})
MATCH (b:Artstyle {id: row.style})
MERGE (a)-[:BELONGS_TO_STYLE]->(b);
LOAD CSV WITH HEADERS FROM 'file:///Artwork_SUBJECT.csv' AS row
MATCH (a:Artwork {id: row.artwork})
MATCH (b:Subject {id: row.subject})
MERGE (a)-[:BELONGS_TO_SUBJECT]->(b);
LOAD CSV WITH HEADERS FROM 'file:///Artwork_DIMENSION.csv' AS row
MATCH (a:Artwork {id: row.artwork}) 
MATCH (b:Dimension {id: row.dimension})
MERGE (a)-[:HAS_LEVEL {
  level: row.level,
  reason: coalesce(row.reason, "")  
}]->(b);
```

## 链路

评价一幅作品：用户给出query和image——向量编码image查询到类似的作品——cypher找到他们的属性——然后把query、image、得到类似图片的属性传递给大模型（不通过QAchain），给一个好的prompt让大模型回答

QAchain反面教材，因为他基于文本查询，不能传图片：

```
Based on the data from the similar artworks you provided, here is an analysis for "test.jpg".\n\n**Analysis of Similar Artworks:**\n\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Sense of Order\n    *   **Level:** Very Good\n    *   **Reason:** Overall dots, lines, and surfaces\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Mood\n    *   **Level:** Very Good\n    *   **Reason:** (No specific reason provided)\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Overall\n    *   **Level:** Very Good\n    *   **Reason:** (No specific reason provided)\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Creativity\n    *   **Level:** Good\n    *   **Reason:** Very innovative traditional Chinese painting techniques\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Theme and Logic\n    *   **Level:** Very Good\n    *   **Reason:** (No specific reason provided)\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Layout and Composition\n    *   **Level:** Very Good\n    *   **Reason:** Complete composition\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Color\n    *   **Level:** Very Good\n    *   **Reason:** Elegant colors, Bold colors\n*   **Artwork: 354e90d1a7e14209b37b0485889fd7d6.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Details and Texture\n    *   **Level:** Very Good\n    *   **Reason:** Delicate shapes\n*   **Artwork: 6946ed4dcf74458ca41d314f372d9dc7.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Layout and Composition\n    *   **Level:** Average\n    *   **Reason:** (No specific reason provided)\n*   **Artwork: 6946ed4dcf74458ca41d314f372d9dc7.png**\n    *   **Category:** Traditional Chinese Painting\n    *   **Artstyle:** Freehand\n    *   **Subject:** Mountains and Water\n    *   **Dimension:** Overall\n    *   **Level:** Average\n    *   **Reason:** (No specific reason provided)\n\n**Summary for "test.jpg":**\n\nThe similar artworks are all traditional Chinese freehand paintings with a mountains and water theme. The evaluation for "test.jpg" can be summarized by comparing it to these works.\n\nThe artwork "354e90d1a7e14209b37b0485889fd7d6.png" consistently received "Very Good" ratings across most dimensions, including its sense of order, mood, composition, color, and details, with specific praise for its use of dots, lines, surfaces, and elegant yet bold colors. Its creativity was also noted as "Good" for being innovative.\n\nIn contrast, the artwork "6946ed4dcf74458ca41d314f372d9dc7.png" received "Average" ratings for its overall quality and layout.\n\nTherefore, "test.jpg" appears to be in the style of a traditional Chinese freehand landscape. To achieve a high evaluation similar to the top-rated example, it should demonstrate a strong command of composition, elegant color use, delicate details, and a clear, well-ordered structure.
```

## 细节

提示词：

在call vllm的提示词中，如果不仔细考虑，那么自定义的gallery模型就会返回一些明显报错的回答：

```
(gallery) D:\desktop\gallerysitp\Gallery-AI-Your-Art-Analysis-Assistant>python client.py
模型回答： ############################################################################################################################################################

(gallery) D:\desktop\gallerysitp\Gallery-AI-Your-Art-Analysis-Assistant>python client.py
模型回答： 46541ceabdf2145133d2e8d.png
```

这是因为这一步的prompt很长（主要是图谱里的参考图片信息），不仅要传给模型多张图片，还要在提示词中清楚的告诉他图片和文件名的对应关系。而且不是简单的看图说话任务，是要模型对比多张图片得出结论。

一个典型的bug就是由于模型的输入token有限，只能控制在图谱中查询图片维度评分的数量，否则prompt就会过长。这样就可能导致实际发给模型的参考图片列表，一些图片的评分还来不及查，没有出现在kg里。但是prompt里又出现了这张图片，他可能觉得对不上就开始胡言乱语。所以我对最终传给模型的参考图片列表又做了一遍筛查，剔除不在kg的图片。