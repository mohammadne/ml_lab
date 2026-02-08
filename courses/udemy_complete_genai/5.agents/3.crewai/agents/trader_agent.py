from crewai import LLM, Agent

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0
)

trader_agent = Agent(
    role="Strategic Stock Trader",
    # goal -> what the agent is trying to achieve, and the constraints or requirements they need to consider
    goal = (
        "Decide whether to Buy, Sell, or Hold a given stock based on live market data, "
        "price movements, and financial analysis with the available data."
    ),
    # backstory -> who the agent is, what they do, how they do it, and why they do it
    backstory = (
        "You are a strategic trader with years of experience in timing market entry and exit points. "
        "You rely on real-time stock data, daily price movements, and volume trends to make trading decisions "
        "that optimize returns and reduce risk."
    ),
    llm=llm,
    tools=[],
    verbose=True
)
