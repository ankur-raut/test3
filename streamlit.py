import streamlit as st
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType,create_csv_agent
from langchain.llms import OpenAI, Cohere
import os
import requests
import json
import yfinance as yf
from yahooquery import Ticker
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.agents import Tool, AgentExecutor
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Cohere
from langchain.evaluation.loading import load_dataset
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain.agents import initialize_agent, load_tools
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain import LLMMathChain, SerpAPIWrapper,LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from fpdf import FPDF
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

st.header('FinAI')

mod = None

with st.sidebar:
    with st.form('Cohere/OpenAI'): 
            model = st.radio('Choose OpenAI/Cohere',('OpenAI','Cohere')) 
            api_key = st.text_input('Enter API key',             
                                    type="password",) 
            serpAI_key = st.text_input('Enter SERPAIAPI key',             
                            type="password",)  

            submitted = st.form_submit_button("Submit")
if api_key:
    if(model == 'OpenAI'):
        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI(temperature=0.3)
        mod = 'OpenAI'
        os.environ["SERPAPI_API_KEY"] = serpAI_key
    if(model == 'Cohere'):
        os.environ["Cohere_API_KEY"] = api_key
        llm = Cohere(cohere_api_key=api_key)
        mod = 'Cohere'
        os.environ["SERPAPI_API_KEY"] = serpAI_key

# agent = initialize_agent(tools,
#                          llm,
#                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                          verbose=True)

# agent.run("Analyze apple stock and craft investment recommendations")

def get_company_news(company_name):
    params = {
        "engine": "google",
        "tbm": "nws",
        "q": company_name,
        "api_key": os.environ["SERPAPI_API_KEY"],
    }

    response = requests.get('https://serpapi.com/search', params=params)
    data = response.json()

    return data.get('news_results')


def write_news_to_file(news, filename):
    with open(filename, 'w') as file:
        for news_item in news:
            if news_item is not None:
                title = news_item.get('title', 'No title')
                link = news_item.get('link', 'No link')
                date = news_item.get('date', 'No date')
                file.write(f"Title: {title}\n")
                file.write(f"Link: {link}\n")
                file.write(f"Date: {date}\n\n")


# company_name = "Microsoft"
# news = get_company_news(company_name)
# if news:
#     write_news_to_file(news, "investment.txt")
# else:
#     print("No news found.")


def get_stock_evolution(company_name, period="1y"):
    # Get the stock information
    stock = yf.Ticker(company_name)

    # Get historical market data
    hist = stock.history(period=period)

    # Convert the DataFrame to a string with a specific format
    data_string = hist.to_string()

    hist.to_csv('stocks_data.csv')

    fin = stock.get_financials()
    fin.to_csv('fin_data.csv')

    # Append the string to the "investment.txt" file
    with open("investment.txt", "a") as file:
        file.write(f"\nStock Evolution for {company_name}:\n")
        file.write(data_string)
        file.write("\n")

# get_stock_evolution("MSFT")  # replace "MSFT" with the ticker symbol of the company you are interested in


def get_financial_statements(ticker):
    # Create a Ticker object
    company = Ticker(ticker)

    # Get financial data
    balance_sheet = company.balance_sheet().to_string()
    cash_flow = company.cash_flow(trailing=False).to_string()
    income_statement = company.income_statement().to_string()
    valuation_measures = str(company.valuation_measures)  # This one might already be a dictionary or string

    # Write data to file
    with open("investment.txt", "a") as file:
        file.write("\nBalance Sheet\n")
        file.write(balance_sheet)
        file.write("\nCash Flow\n")
        file.write(cash_flow)
        file.write("\nIncome Statement\n")
        file.write(income_statement)
        file.write("\nValuation Measures\n")
        file.write(valuation_measures)

def get_data(company_name, company_ticker, period="1y", filename="investment.txt"):
     news = get_company_news(company_name)
     if news:
         write_news_to_file(news, filename)
     else:
         print("No news found.")

     get_stock_evolution(company_ticker)

     get_financial_statements(company_ticker)

import openai
def financial_analyst(request):
    print(f"Received request: {request}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{
            "role": "user",
            "content": f"Given the user request, what is the comapany name and the company stock ticker ?: {request}?"
        }],
        functions=[{
            "name": "get_data",
            "description":
            "Get financial data on a specific company for investment purposes",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type":
                        "string",
                        "description":
                        "The name of the company",
                    },
                    "company_ticker": {
                        "type":
                        "string",
                        "description":
                        "the ticker of the stock of the company"
                    },
                    "period": {
                        "type": "string",
                        "description": "The period of analysis"
                    },
                    "filename": {
                        "type": "string",
                        "description": "the filename to store data"
                    }
                },
                "required": ["company_name", "company_ticker"],
            },
        }],
        function_call={"name": "get_data"},
    )

    message = response["choices"][0]["message"]

    if message.get("function_call"):
        # Parse the arguments from a JSON string to a Python dictionary
        arguments = json.loads(message["function_call"]["arguments"])
        company_name = arguments["company_name"]
        company_ticker = arguments["company_ticker"]

        # Parse the return value from a JSON string to a Python dictionary
        get_data(company_name, company_ticker)

        with open("investment.txt", "r") as file:
            content = file.read()[:14000]

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": request
                },
                message,
                {
                    "role": "system",
                    "content": """write a detailed investment thesis to answer
                      the user request. Provide numbers to justify
                      your assertions, a lot ideally. Never mention
                      something like this:
                      However, it is essential to consider your own risk
                      tolerance, financial goals, and time horizon before
                      making any investment decisions. It is recommended
                      to consult with a financial advisor or do further
                      research to gain more insights into the company's f
                      undamentals and market trends. The user
                      already knows that"""
                },
                {
                    "role": "assistant",
                    "content": content,
                },
            ],
        )

        return second_response["choices"][0]["message"]["content"]

def generate_pdf(text,image_paths):
    # Save FPDF() class into a variable pdf
    pdf = FPDF()

    # Add a page
    pdf.add_page()
    # Set style and size of font that you want in the PDF
    pdf.set_font("Arial", size=12)

    # Set left margin and right margin
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)

    # Add multi-cell with line break
    pdf.multi_cell(0, 10, text)

    # Move to the next line after the text
    pdf.ln()

    # Add a page
    pdf.add_page()

    # Add the first image to the PDF
    pdf.image(image_paths[0], x=20, y=pdf.get_y(), w=175)

    # Calculate the y-coordinate for the second image
    second_image_y = pdf.get_y() + 150

    # Add the second image to the PDF
    pdf.image(image_paths[1], x=20, y=second_image_y, w=175)

    # Save the PDF with the given file name
    pdf.output("output.pdf")

def graphs(path,prompt):
    agent = create_csv_agent(
        OpenAI(temperature=0),
        path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    agent.run(prompt)

# Category 1: Revenue and Sales
revenue_sales_questions = [
    "What was the total revenue generated by the company during the fiscal year?",
    "How does the revenue of the current year compare to the previous year?",
    "Which product/service contributed the most to the company's revenue?",
    "Did the company experience any significant changes in sales volume or pricing?",
    "Are there any notable trends or patterns in the revenue growth of the company over the past few years?",
    "What were the geographical regions or markets where the company generated the highest revenue?",
    "Has the company introduced any new revenue streams or business lines?",
    "Were there any extraordinary events or factors that affected the company's revenue performance?",
    "How does the revenue composition of the company compare to its competitors in the industry?",
    "Are there any forecasts or projections for future revenue growth provided in the report?"
]

# Category 2: Expenses and Costs
expenses_costs_questions = [
    "What were the major expense categories for the company during the fiscal year?",
    "How do the expenses of the current year compare to the previous year?",
    "Did the company implement any cost-saving measures or efficiency improvements?",
    "Were there any significant changes in the cost of raw materials or production inputs?",
    "How does the company's expense ratio compare to industry benchmarks?",
    "Did the company incur any one-time or non-recurring expenses during the year?",
    "Are there any trends or patterns in the company's cost structure over the past few years?",
    "Has the company invested in research and development (R&D) or capital expenditures?",
    "What were the employee-related costs and benefits provided by the company?",
    "Are there any forecasts or projections for future cost management initiatives provided in the report?"
]

# Category 3: Profitability and Financial Ratios
profitability_ratios_questions = [
    "What was the net profit or net income generated by the company during the fiscal year?",
    "How does the profitability of the current year compare to the previous year?",
    "What is the company's gross profit margin and how has it changed over time?",
    "Did the company experience any changes in operating profit or operating margin?",
    "What is the return on assets (ROA) and return on equity (ROE) for the company?",
    "Has the company improved its profitability compared to its competitors in the industry?",
    "Are there any trends or patterns in the company's profitability ratios over the past few years?",
    "Did the company face any challenges or risks that impacted its profitability?",
    "How does the company's profitability ratios compare to industry benchmarks?",
    "Are there any forecasts or projections for future profitability provided in the report?"
]

# Category 4: Cash Flow and Liquidity
cash_flow_liquidity_questions = [
    "What was the operating cash flow generated by the company during the fiscal year?",
    "How does the cash flow from operations of the current year compare to the previous year?",
    "Did the company experience any significant changes in its working capital management?",
    "What were the major sources and uses of cash for the company during the year?",
    "Has the company made any significant investments or divestments during the year?",
    "How does the company's cash conversion cycle compare to industry benchmarks?",
    "Are there any trends or patterns in the company's cash flow statement over the past few years?",
    "What is the company's current ratio and quick ratio for assessing liquidity?",
    "Did the company undertake any debt financing or equity financing activities?",
    "Are there any forecasts or projections for future cash flow or liquidity provided in the report?"
]
# Define the options for the dropdown menu
categories = [
    "Revenue and Sales",
    "Expenses and Costs",
    "Profitability and Financial Ratios",
    "Cash Flow and Liquidity"
]
# Create a textbox to enter company's name
company_name = st.text_input("Enter the company's name:")

uploaded_file = st.file_uploader(f"Upload an Annual Report of {company_name} if available (PDF).", type=['pdf'])
toolkit = None
if uploaded_file is not None: 
        st.write("File uploaded successfully!")
        file_contents = uploaded_file.read()
        save_path = uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(file_contents)
        print(save_path)
        loader = PyPDFLoader(save_path) #Step 1.1
        documents = loader.load()
        #1.2
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0) #Splitting the text and creating chunks
        docs = text_splitter.split_documents(documents)
        if(mod=="OpenAI"):
            embeddings = OpenAIEmbeddings()
        if(mod=="Cohere"):
            embeddings = CohereEmbeddings(cohere_api_key=api_key)
        store = Chroma.from_documents(docs,embeddings)
        vectorstore_info = VectorStoreInfo(
            name="starbucks",
            description="Starbucks financials",
            vectorstore=store,
        )
        # llm = OpenAI(temperature=0.3)
        toolkit = VectorStoreToolkit(llm=llm,vectorstore_info=vectorstore_info)


# Create a dropdown using the `selectbox` function
selected_category = st.selectbox("Select a category:", categories)

if(selected_category=="Revenue and Sales"):
    selected_ques= st.selectbox("Select a category:", revenue_sales_questions)
if(selected_category=="Expenses and Costs"):
    selected_ques= st.selectbox("Select a category:", expenses_costs_questions)
if(selected_category=="Profitability and Financial Ratios"):
    selected_ques= st.selectbox("Select a category:", profitability_ratios_questions)
if(selected_category=="Cash Flow and Liquidity"):
    selected_ques= st.selectbox("Select a category:", cash_flow_liquidity_questions)

# st.write(selected_ques)

ans=[]
if (st.button("Submit")):
    output_res=[]

    st.write("Company Name: " + company_name)

    # llm = OpenAI(temperature=0.3)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(llm = llm,
                    toolkit = toolkit,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    tools = tools,
                    verbose=True)
    if(mod=='OpenAI'):
        ans = financial_analyst(company_name)
        st.write("Question asked by the user is " + selected_ques)
        response = agent.run(selected_ques+f" Consider {ans}")
        st.write(response)
        st.write("Report")
        st.write(ans)
        prompt2="Make a line graph with Date as the x label and Closing Value as the y label and save the graph as an image file name of file as 'img2.png'"
        prompt1="Make a line graph, x axis lables should be rotation = 90, save the graph as an image file and name of file as 'img1.png'"
        graphs('stocks_data.csv',prompt1)
        graphs('fin_data.csv',prompt2)
        image_path = ['img1.png','img2.png']
        generate_pdf(ans,image_path)
        st.write("PDF generated successfully! Click below to download.")
        # Download link
        with open("output.pdf", "rb") as f:
            st.download_button("Download PDF", f.read(), file_name="output.pdf", mime="application/pdf")
    else:
        try:
            response = agent.run(f"As a financial data analyst, your task is to thoroughly analyze \
            the annual financial report of a company and provide accurate answers based solely on the data presented \
            in the document. It is important to strictly adhere to the information provided in the report and \
            refrain from making any assumptions or speculations. \
            If necessary, you may utilize appropriate tools and formulas to derive the required answers.\
            prompt = {selected_ques}")
            print(response)
            # response = agent.run(selected_ques)
            st.write(response)
        except:
            st.write("Cohere Key Cannot give out the desired outputs. Pls provide OpenAI key for better results or try again!")