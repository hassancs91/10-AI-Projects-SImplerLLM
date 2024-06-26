from SimplerLLM.language.llm import LLM,LLMProvider
from SimplerLLM.tools.generic_loader import load_content

llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

url = "https://www.youtube.com/watch?v=LJeZq8MymAs"

content = load_content(url).content

summarize_prompt = f"generate a bullet point summary for the following: {content}"

generated_text = llm_instance.generate_response(prompt=summarize_prompt)

print(generated_text)