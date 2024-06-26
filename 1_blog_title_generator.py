from SimplerLLM.language.llm import LLM,LLMProvider
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")


blog_title_generator_prompt = """I want you to act as a professional blog titles generator. 
Think of titles that are seo optimized and attention-grabbing at the same time,
and will encourage people to click and read the blog post.
I want to generate 10 titles maximum in a numbered list format.
My blog post is is about: Digital Marketing tips                                                                     
"""

generated_text = llm_instance.generate_response(prompt=blog_title_generator_prompt)


print(generated_text)