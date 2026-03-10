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
        You need to :
        1. generate Cypher to query the scores of dimensions and the reasons in all HAS_LEVEL relationships they are involved in, 
        2. return the results from knowledge graph in the following JSON format:
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
        Note: you only need to return the json results without any explanation.
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

        # 假设kg是json数组
        unique_filenames = list({item["filename"] for item in kg if "filename" in item})

        return unique_filenames,kg
    else:
        print("No similar artworks found.Closed...")