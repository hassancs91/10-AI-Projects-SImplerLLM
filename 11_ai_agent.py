from SimplerLLM.language.llm import LLMProvider
from SimplerLLM.tools.generic_loader import load_content
from agent_class_experimental import Agent


#Define a custom tool
def load_content_from_url(url: str):
    """
    Load the page content from a given URL.
    Parameters: url (str)
    """
    content = load_content(url)
    return content.content


# Create an agent instance
agent = Agent(LLMProvider.OPENAI, model_name="gpt-4o")

#add tools
agent.add_tool(load_content_from_url)

user_query = """
generate a consise bullet point summary of the following article: 
https://learnwithhasan.com/generate-content-ideas-ai/?"""

# Generate a response
agent.generate_response(user_query)








