from SimplerLLM.language.llm import LLM,LLMProvider
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.tools.serp import search_with_serper_api

llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

search_query = "best microphones 2024"

search_results = search_with_serper_api(query=search_query,num_results=3)

overall_research = ""

for result in search_results:
    content = load_content(str(result.URL)).content
    main_points = llm_instance.generate_response(prompt=f"extract the key points of the following: {content}")
    overall_research = overall_research + "\n\n"  + main_points


user_prompt = f"extract the key information out of the following {overall_research}"

final_result = llm_instance.generate_response(prompt=user_prompt)

print(final_result)