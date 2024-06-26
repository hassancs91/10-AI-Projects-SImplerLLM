from SimplerLLM.language.llm import LLM,LLMProvider
from SimplerLLM.prompts.prompt_builder import create_multi_value_prompts

llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

## working with multiple value prompts
multi_value_prompt_template = """Generate 5 titles for a blog about {topic} and {style}"""

params_list = [
     {"topic": "SEO tips", "style": "catchy"},
     {"topic": "youtube growth", "style": "click baity"},
     {"topic": "email matketing tips", "style": "SEO optimized"}
]

multi_value_prompt = create_multi_value_prompts(multi_value_prompt_template)
generated_prompts = multi_value_prompt.generate_prompts(params_list)


for prompt in generated_prompts:
    response = llm_instance.generate_response(prompt=prompt)
    print(response)
