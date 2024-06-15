import os
from dotenv import load_dotenv
from langchain import LLMMathChain, PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain_core.tools import tool, Tool, StructuredTool
from tools.agents.text_summarizer import TextSummarizer
from utilities.scraper import Scraper
from pydantic import BaseModel, Field
from enum import Enum
import json
import requests
import re
import getpass
import logging

load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")

#init model as a global variable - no need to keep more than one instance of the model
model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_math_chain = LLMMathChain(llm=model, verbose=True)

class ScrapingWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="URL of the website to scrape")

def scrape_and_summarize(objective: str, url: str) -> str:
    """
    Scrape a website and summarize its content based on a given objective.
    Args:
        input (ScrapingWebsiteInput): The input containing the objective and the URL to scrape.
    Returns:
        str: The summarized content.
    """
    scraper = Scraper()
    text_summarizer = TextSummarizer(model=model)

    scraped = scraper.scrape(url)
    summarized = text_summarizer.create_summary(objective=objective, content=scraped.get_text())

    return summarized

class GetPriceInPLNInput(BaseModel):
    """Inputs to convert price to pln tool"""
    input_price_eur: float

def convert_price_to_pln(input_price_eur: float):
    """
    Having an input amount in EUR, convert it into the corresponding amount in Polish zloty.
    Calculation is based on the current exchange rate between currencies.
    """
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    url = f'https://v6.exchangerate-api.com/v6/{api_key}/latest/EUR'

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Error fetching exchange rate data: {response.status_code}")

    data = response.json()
    pln_rate = data.get('conversion_rates', {}).get('PLN')

    if pln_rate is None:
        raise ValueError("PLN exchange rate not found in the API response")

    print(f"Current EUR price in PLN: {pln_rate}")

    price_pln = input_price_eur * pln_rate
    print(f"Car price in PLN: {price_pln}")

    return price_pln

class FuelType(str, Enum):
    Pb95 = "Pb 95"
    Pb98 = "Pb 98"
    ON = "ON"
    LPG = "LPG"

class GetCurrentFuelPriceInput(BaseModel):
    """Inputs to parsing fuel price tool"""
    fuel_type: FuelType = Field(description="Type of fuel: Pb 95, Pb 98, ON, LPG")

def get_fuel_price(fuel_type: FuelType):
    """
    scrape specific website containing average fuel prices for particular country
    """
    url = 'https://www.reflex.com.pl/'
    scraper = Scraper()
    soup = scraper.scrape(url)
    prices_block = soup.select('div.nboxes-prices-info ul li')
    fuel_prices = {}

    for li in prices_block:
        label = li.select_one('div.nboxes-prices-list-label').get_text(strip=True)
        value = li.select_one('div.nboxes-prices-list-value').get_text(strip=True)
        try:
            numeric_value = re.findall(r"[-+]?\d*\.\d+|\d+", value)[0]
            fuel_prices[label] = float(numeric_value)
        except ValueError:
            print(f"Could not convert {value} to float")

    print(f"Fuel prices keys: {list(fuel_prices.keys())}")
    print(f"Requested fuel type: {fuel_type}")

    if fuel_type.value not in fuel_prices:
        raise KeyError(f"Fuel type {fuel_type.value} not found in the scraped data")
        
    return fuel_prices[fuel_type.value]

class CalculateFuelCostsInput(BaseModel):
    """Inputs to fuel costs calculation"""
    avg_fuel_consumption_per_100_km: float = Field(description="Average fuel consumption of the car per 100km")
    fuel_price: float = Field(description="Price of the car fuel in polish zloty")

def calculate_fuel_costs(avg_fuel_consumption_per_100_km: float, fuel_price: float):
    """
    Calculate monthly fuel costs based on average fuel consumption and fuel type.
    Args:
        input (CalculateFuelCostsInput): The input containing average fuel consumption and fuel type.
    Returns:
        dict: A dictionary containing the fuel type, fuel price per liter, monthly fuel consumption in liters, and the total monthly fuel cost.
    """
    # Assuming 1000 kilometers driven monthly
    monthly_distance = 1000
    monthly_fuel_consumption = (monthly_distance / 100) * avg_fuel_consumption_per_100_km
    monthly_fuel_cost = monthly_fuel_consumption * fuel_price

    return {
        "fuel_price_per_liter": fuel_price,
        "monthly_fuel_consumption_liters": monthly_fuel_consumption,
        "monthly_fuel_cost": monthly_fuel_cost
    }

class QueryInput(BaseModel):
    """Inputs for Google search"""
    q: str = Field(description="The query that users gives to the agent")

@tool("search", args_schema=QueryInput)
def search(input: QueryInput) -> str:
    """
    Perform Google search with exact phrase and parse the results.
    Args:
        input (QueryInput): The input containing the query for the search.
    Returns:
        str: The search results.
    """
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": input
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    } 
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return response.text

 # creates langchain agent with the tools
tools = [
    Tool.from_function(
        func=search,
        name="search",
        args_schema=QueryInput,
        description="use when you need to perform google search to look up information on the internet"
    ),
    StructuredTool.from_function(
        func=scrape_and_summarize,
        name="scrape_and_summarize",
        args_schema=ScrapingWebsiteInput,
        description="use when you need to scrape the website and summarize it's content",
    ),
    StructuredTool.from_function(
        func=convert_price_to_pln,
        name="convert_price_to_pln",
        args_schema=GetPriceInPLNInput,
        description="use when you have car price in EUR, and you need to convert it to price in PLN."
    ),
    StructuredTool.from_function(
        func=get_fuel_price,
        name="get_fuel_price",
        args_schema=GetCurrentFuelPriceInput,
        description="use when you need to get the current price of a particular fuel suited for particular car model"
    ),
    StructuredTool.from_function(
        func=calculate_fuel_costs,
        name="calculate_fuel_costs",
        args_schema=CalculateFuelCostsInput,
        description="use when you need to calculate monthly fuel costs for particular car"
    ),
    Tool.from_function(
        func=llm_math_chain.run,
        name="calculator",
        description="useful to calculate costs, like monthly exploatation and payment costs, and other calculations. Useful to convert between currencies.",
    )
]

system_message = SystemMessage(
    content="""You are a world class automotive journalist and researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research.
            You have huge knowledge about cars, and you can answer any questions about cars.
            You have an ability to write engaging, informative and captivating posts and articles about cars, especially about sports and luxury cars.
            Your job is to make less knowledgable user gain immediate value from your research, and you should write in a way that is easy to understand.
            
            Make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
                """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=model, max_token_limit=1000)

agent = initialize_agent(
    tools,
    model,
    agent=AgentType.OPENAI_FUNCTIONS, # legacy agent type - https://python.langchain.com/v0.1/docs/modules/agents/agent_types/
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=100,
    max_execution_time=180
)   

class AgentRunner():
    def __init__(self) -> None:
        self.agent = initialize_agent(
            tools,
            model,
            agent=AgentType.OPENAI_FUNCTIONS, # legacy agent type - https://python.langchain.com/v0.1/docs/modules/agents/agent_types/
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
            max_iterations=100,
            max_execution_time=180
        ) 
    
    def run_agent(self, car_data):
        """
        Run the agent with the provided car data.
        """
        template = """The goal of the agent is to prepare an instagram post about a hot car for the flame_n_gasoline instagram page. This page is about hot sports cars. It aims to educate the audience about the iconic sports cars and inspire them to buy the particular car one day. Each post contains interesting facts about the car and calculation of the costs necessary to own the car. The task is to create the post about this car: {car_data}.  Include the nicely formatted information about the tech data. Perform google search for intersting facts about the car and the car history. Then create engaging summary plugging the tech data like Horsepower and torque, that will be suitable for instagram post. Add tech data provided in input to enrich the summary content. Find out what is its average price EUR. Then, convert the price to polish zloty (PLN). Then use only amounts in PLN. Find out what is average fuel consumption for this car per 100 kilometers and what fuel type it uses. Find what is the current price for the fuel type that the car uses in Polish Zloty. Calculate what would be the average monthly fuel cost, assuming we drive 1000 kilometers monthly, using current fuel price in Polish Zloty. Calculate the theoretical leasing cost for this car, assuming that buyer submits 40% (0.4) down payment which is the 40% of the total car price. The remaining total price (0.6) of the car should be splitted for 3 years (36 months) - this would be the monthly installment. Add 10% or multiply the monthly installment by 1.1 to calculate monthly installment including leasing interest. At the end, collect 4 prices: the total average price of this car, the down payment, the monthly installment cost including leasing interest rate, and the monthly fuel cost. At the end, prepare 3-4 sentences, engaging post for instagram about the car, which includes interesting fact and summary of the costs including 4 prices calculated before. The post is aimed at the automotive enthusiasts, who aim to afford particular car. This post should serve as the inspiration for them to earn money and get this car, fire them up for having this car and equipping them with knowledge about this car, so that they build up solid knowledge about various hot cars so they can make their own informed decision."""
        prompt = PromptTemplate(
            input_variables=["car_data"],
            template=template
        )
        formatted_prompt = prompt.format(car_data=car_data)
        response = self.agent({"input": formatted_prompt})
        
        return response['output']