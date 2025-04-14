# cookbook/agents/finance_agents.py
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.thinking import ThinkingTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.financial_datasets import FinancialDatasetsTools

def get_finance_agent():
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[ThinkingTools(add_instructions=True), YFinanceTools(enable_all=True)],
        name="Finance Analyst",
        role="Provide deep analysis of market trends and stocks.",
        show_tool_calls=True,
        markdown=True,
    )

def get_financial_datasets_agent():
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[FinancialDatasetsTools()],
        name="Datasets Agent",
        role="Query financial statements, trends, and ownership structures.",
        show_tool_calls=True,
        markdown=True,
    )
