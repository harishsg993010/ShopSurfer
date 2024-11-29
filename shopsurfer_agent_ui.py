import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool
import gradio as gr

class ShoppingCrew:
    def __init__(self):
        # Initialize tools
        self.search_tool = SerperDevTool()
        self.website_search_tool = WebsiteSearchTool()
        self.scrape_tool = ScrapeWebsiteTool()

    def create_agents(self):
        # Product Verification Agent (New!)
        verification_agent = Agent(
            role='Product Specification Verifier',
            goal='Verify exact product details and model information',
            backstory="""You are a technical specialist who verifies product specifications, 
            model numbers, and ensures accuracy of product information. You have extensive 
            experience in consumer electronics and can spot incorrect or outdated information.""",
            tools=[self.search_tool, self.website_search_tool],
            verbose=True
        )

        # Research Agent (Enhanced)
        research_agent = Agent(
            role='Product Research Specialist',
            goal='Conduct comprehensive product research including customer reviews and availability',
            backstory="""You are an expert product researcher who specializes in 
            detailed product analysis. You focus on gathering authentic customer reviews,
            checking real-time stock availability, and verifying warranty information.
            You know how to distinguish genuine reviews from fake ones and can provide
            balanced perspectives from actual users.""",
            tools=[self.search_tool, self.website_search_tool, self.scrape_tool],
            verbose=True
        )

        # Deal Finding Agent (Enhanced)
        deal_finder_agent = Agent(
            role='Deals Analysis Expert',
            goal='Find and verify current deals with specific dates and conditions',
            backstory="""You are a professional deal finder who specializes in 
            tracking promotion dates and conditions. You verify the legitimacy of deals,
            understand complex terms and conditions, and can find hidden savings
            opportunities. You're expert at comparing warranty options and total ownership costs.""",
            tools=[self.search_tool, self.website_search_tool, self.scrape_tool],
            verbose=True
        )

        # Report Generation Agent (Enhanced)
        report_agent = Agent(
            role='Shopping Report Specialist',
            goal='Create detailed, accurate shopping reports with specific links and availability info',
            backstory="""You are an experienced shopping analyst who creates 
            comprehensive reports. You ensure all information is current and accurate,
            include specific product links, and provide detailed availability information.
            You excel at presenting complex information in an easy-to-understand format
            while maintaining technical accuracy.""",
            verbose=True
        )

        return verification_agent, research_agent, deal_finder_agent, report_agent

    def create_tasks(self, query: str, verification_agent: Agent, research_agent: Agent, 
                    deal_finder_agent: Agent, report_agent: Agent):
        # Verification Task (New!)
        verification_task = Task(
            description=f"""Verify exact product specifications for: {query}
            
            Required Steps:
            1. Confirm exact model number and name
            2. Verify all technical specifications
            3. Check for any recent product updates or revisions
            4. Confirm product variants and options
            5. Validate compatibility information
            
            Return:
            - Exact model numbers and names
            - Confirmed specifications
            - Any discrepancies found
            - Latest product updates""",
            agent=verification_agent,
            expected_output="Detailed product verification findings"
        )

        # Research Task (Enhanced)
        research_task = Task(
            description=f"""Research this product in detail: {query}
            
            Required Steps:
            1. Search for the product across major retailers
            2. Collect and analyze customer reviews (minimum 10)
            3. Check current stock availability
            4. Verify warranty options and terms
            5. Analyze return policies
            6. Compare technical specifications
            
            Return findings including:
            - Verified retailer list with exact product page URLs
            - Detailed warranty information
            - Stock availability status
            - Curated customer reviews (positive and negative)
            - Technical specifications comparison
            - Return policy details""",
            agent=research_agent,
            context=[verification_task],
            expected_output="""Detailed product research findings which should 
             Verified retailer list with exact product page URLs
             Detailed warranty information
             Stock availability status
             Curated customer reviews (positive and negative)
             Technical specifications comparison
             Return policy details
            """
        )

        # Deal Finding Task (Enhanced)
        deal_finding_task = Task(
            description="""Using the research findings, analyze current deals:
            
            Required Steps:
            1. Verify current prices and promotions
            2. Document specific promotion dates
            3. Detail all terms and conditions
            4. Compare warranty options
            5. Check stock availability for deals
            6. Verify shipping timeframes
            
            Provide for each retailer:
            - Current price with promotion dates
            - Detailed terms and conditions
            - Warranty options and costs
            - Stock availability status
            - Shipping/pickup options and timeframes
            - Total cost breakdown including all fees""",
            agent=deal_finder_agent,
            context=[verification_task, research_task],
            expected_output="""Detailed deals analysis which should include:
            Current price with promotion dates
            Detailed terms and conditions
            Warranty options and costs
            Stock availability status
            Shipping/pickup options and timeframes
            Total cost breakdown including all fees
            """

        )

        # Report Generation Task (Enhanced)
        report_task = Task(
            description="""Create a comprehensive shopping report including:
            
            1. Executive Summary
               - Top recommendation with reasoning
               - Price range overview with specific dates
               - Best current deal with terms
            
            2. Product Details
               - Verified model numbers and specifications
               - Detailed pros and cons
               - Curated customer reviews (minimum 5 positive and 5 negative)
               - Warranty options comparison
            
            3. Price and Availability Comparison
               - Retailer comparison table with exact URLs
               - Current stock availability
               - Promotion dates and terms
               - Complete cost breakdown
            
            4. Buying Guide
               - Best retailer choice with reasoning
               - Stock availability alerts
               - Direct product page links
               - Warranty recommendations
            
            Format the report clearly with sections and bullet points.
            Include specific dates, model numbers, and direct links.""",
            agent=report_agent,
            context=[verification_task, research_task, deal_finding_task],
            expected_output="""Final shopping report which includes clearly with sections and bullet points.
            Include specific dates, model numbers, and direct links"""
        )

        return [verification_task, research_task, deal_finding_task, report_task]

    def run(self, query: str):
        # Create agents
        verification_agent, research_agent, deal_finder_agent, report_agent = self.create_agents()
        
        # Create tasks
        tasks = self.create_tasks(query, verification_agent, research_agent, 
                                deal_finder_agent, report_agent)
        
        # Create crew
        crew = Crew(
            agents=[verification_agent, research_agent, deal_finder_agent, report_agent],
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        # Start the crew
        result = crew.kickoff()
        
        return result


def validate_api_keys(openai_key, serper_key):
    """Validate API keys are not empty"""
    if not openai_key.strip() or not serper_key.strip():
        return "Please enter both API keys"
    return None

def search_products(openai_key, serper_key, query):
    """Handle the product search with API key validation"""
    # Validate inputs
    validation_error = validate_api_keys(openai_key, serper_key)
    if validation_error:
        return validation_error
    
    # Set API keys
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["SERPER_API_KEY"] = serper_key
    
    try:
        # Create and run the shopping crew
        shopping_crew = ShoppingCrew()
        result = shopping_crew.run(query)
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example search queries
EXAMPLE_QUERIES = [
    "High-end espresso machine under $1000",
    "4K gaming monitor with 144Hz refresh rate",
    "Noise-cancelling headphones with long battery life",
    "Robot vacuum with mapping capability",
    "Air fryer with digital controls",
    "Smart doorbell with video recording",
]

# Create the Gradio interface
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # AI Shopping Assistant
        Enter your API keys and search for products to get detailed analysis and recommendations.
        """)
        
        with gr.Row():
            with gr.Column():
                openai_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password"
                )
                serper_key = gr.Textbox(
                    label="Serper.dev API Key",
                    placeholder="Enter your Serper.dev API key",
                    type="password"
                )
        
        with gr.Row():
            query = gr.Textbox(
                label="What are you looking for?",
                placeholder="Enter your product search query...",
                lines=2
            )
        
        with gr.Row():
            examples = gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=query,
                label="Example Searches"
            )
        
        with gr.Row():
            search_button = gr.Button("Search", variant="primary")
        
        output = gr.Textbox(
            label="Results",
            lines=20
        )
        
        search_button.click(
            fn=search_products,
            inputs=[openai_key, serper_key, query],
            outputs=output
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True)
