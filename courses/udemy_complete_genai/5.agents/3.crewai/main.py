from agents.analyst_agent import analyst_agent
from agents.trader_agent import trader_agent
from crewai import Crew
from dotenv import load_dotenv
from tasks.analyse_task import get_stock_analysis
from tasks.trade_task import trade_decision

load_dotenv()

stock_crew = Crew(
    agents=[analyst_agent, trader_agent],
    tasks=[get_stock_analysis, trade_decision],
    verbose=True
)


def run(stock: str):
    result = stock_crew.kickoff(inputs={"stock": stock})
    print(result)


if __name__ == "__main__":
    # run("TESLA")
    run("APPLE")
