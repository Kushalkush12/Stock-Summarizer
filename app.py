__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import requests
import streamlit as st
from crewai import Crew, LLM, Task, Agent, Process
from crewai.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

googleapi = os.getenv('GOOGLE_API_KEY')
fin_api_key = os.getenv('FIN_DATA_API')

# Initialize LLM
llm = LLM(
    model='gemini/gemini-2.0-flash',
    api_key=googleapi
)

st.set_page_config(page_title="Stock Market Analysis Dashboard", page_icon="📊", layout="centered")

# Title and Input UI
st.markdown("# 📈 Stock Market Analysis Dashboard")
st.write("Enter the stock symbol (e.g., HAL, IBM):")

symbol = st.text_input("", placeholder="Enter stock symbol", max_chars=30).strip().upper()

if st.button("Analyze Stock") and symbol:
    with st.status("Fetching stock data..."):
        # Define API Tool
        class APIRequestTool(BaseTool):
            name: str = "API tool"
            description: str = "Fetches stock data from Alpha Vantage API"
            url: str

            def _run(self):
                response = requests.get(self.url)
                return response.json()


        # Create API request tools
        weekly_data_tool = APIRequestTool(
            url=f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={fin_api_key}')
        monthly_data_tool = APIRequestTool(
            url=f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={fin_api_key}')

    with st.status("Analyzing weekly and monthly trends..."):
        # Define Agents
        weekly_agent = Agent(
            role="Stock Market Weekly Data Analyst",
            goal="Analyze the weekly stock data and generate insights.",
            backstory="Experienced financial analyst with expertise in technical analysis.",
            llm=llm,
            tools=[weekly_data_tool]
        )

        monthly_agent = Agent(
            role="Stock Market Monthly Data Analyst",
            goal="Analyze long-term stock trends and performance.",
            backstory="Veteran financial consultant specializing in market trends.",
            llm=llm,
            tools=[monthly_data_tool]
        )

        summarizer = Agent(
            role="Stock Market Insight Generator",
            goal="Summarize key insights from weekly and monthly trends.",
            backstory="Expert strategist distilling market data into actionable insights.",
            llm=llm
        )

        # Define Tasks
        weekly_task = Task(
            description="Generate a report for weekly stock trends.",
            expected_output="Weekly stock performance insights.",
            agent=weekly_agent
        )

        monthly_task = Task(
            description="Generate a report for monthly stock trends.",
            expected_output="Monthly stock performance insights.",
            agent=monthly_agent
        )

        summary_task = Task(
            description="Summarize weekly and monthly reports into one investment insight.",
            expected_output="Comprehensive stock analysis.",
            agent=summarizer,
            context=[weekly_task, monthly_task]
        )

    with st.status("Generating final report..."):
        # Create Crew
        crew = Crew(
            agents=[weekly_agent, monthly_agent, summarizer],
            tasks=[weekly_task, monthly_task, summary_task],
            process=Process.sequential,
            verbose=True
        )

        res = crew.kickoff()

    # Display the markdown report directly
    st.markdown(res.raw, unsafe_allow_html=True)
