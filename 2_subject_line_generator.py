from SimplerLLM.language.llm import LLM,LLMProvider
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")


subject_line_generator_prompt = """as an expert in email marketing, your task is to
generate special email subject Lines.

Based on the following input: digital mareting tips.

Output: Return ONLY list of 5 Subject Lines in a bulleted list.                                                                     
"""

generated_text = llm_instance.generate_response(prompt=subject_line_generator_prompt)


print(generated_text)