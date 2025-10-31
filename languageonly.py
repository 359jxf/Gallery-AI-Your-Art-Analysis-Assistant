from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import os

def queryGraph(llm,graph,query,top_k=20):
    # 初始化Cypher QA链
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        top_k=top_k,
        allow_dangerous_requests=True
    )
    res = chain.invoke({"query": query})  
    print(res['result'])

if __name__ == '__main__':
    # 连接Neo4j
    graph = Neo4jGraph(
        url="neo4j://localhost:7687",  
        username="neo4j",       
        password="apropos-sphere-violin-texas-strong-2496",
        enhanced_schema=True,
    )
    
    # 加载.env文件的环境变量
    load_dotenv()

    deepseek_llm = ChatOpenAI(
        model_name="deepseek-chat", 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  
        openai_api_base="https://api.deepseek.com/v1", 
    )

    queryGraph(deepseek_llm,graph,query="how to evaluate color",top_k=10)