from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

#纯文本问答，直接查询图谱
def queryGraph(llm,graph,query,top_k=20):
    graph.refresh_schema()

    # 初始化Cypher QA链
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        top_k=top_k,
        allow_dangerous_requests=True
    )
    res = chain.invoke({"query": query})  
    return res['result']


# 查询图片维度得分信息，格式化返回
def queryImage(llm,graph,top_k=20,image_filenames=[]):
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
        graph.refresh_schema()
        
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            top_k=top_k,
            allow_dangerous_requests=True
        )
        res = chain.invoke({"query": query})
        kg=res['result']
        return kg
    else:
        print("No similar artworks found.Closed...")