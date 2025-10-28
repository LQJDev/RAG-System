from langchain_openai import ChatOpenAI

from langchain_openai import ChatOpenAI

def test_deepseek():
    llm = ChatOpenAI(
        model="deepseek-chat",  # DeepSeek 官方推荐模型名
        api_key="sk-133c8a8e18bd4049bddeda8397868174",  # ✅ 换成你的实际 key
        base_url="https://api.deepseek.com/v1",  # DeepSeek 官方 API 地址
        temperature=0.3
    )

    response = llm.invoke("你好，用 LangChain 调用 DeepSeek 成功了吗？")

    print("✅ 模型响应：", response.content)

if __name__ == "__main__":
   test_deepseek()