from SimplerLLM.language.llm import LLM,LLMProvider

llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")



hook_generator_prompt = """
as an expert copywriter specialized in hook generation, your task is to 
analyze the [Provided_Hook_Examples].

Use the templates that fit most to generate 3 new Hooks 
for the following topic: {user_input} and Usage in: {usage}. 

The output should be ONLY valid JSON as follows:
[
  {{
    "hook_type": "The chosen hook type",
    "hook": "the generatoed hook"
  }},
  {{
    "hook_type": "The chosen hook type",
    "hook": "the generatoed hook
  }},
  {{
    "hook_type": "The chosen hook type",
    "hook": "the generatoed hook"
  }}
]

[Provided_Hook_Examples]:
"Hook Type,Template,Use In
Strong sentence,"[Topic] won’t prepare you for [specific aspect].",Social posts, email headlines, short content
The Intriguing Question,"What’s the [adjective describing difference] difference between [Subject 1] and [Subject 2]?",Video intros, email headlines, social posts
Fact,"Did you know that [Interesting Fact about a Topic]?",Video intros, email headlines, social posts, short content
Metaphor,"[Subject] is like [Metaphor]; [Explanation of Metaphor].",Video intros, email headlines, short content
Story,"[Time Frame], [I/We/Subject] was/were [Situation]. Now, [Current Situation].",Video intros, short content
Statistical,"Nearly 70% of [Population] experience [Phenomenon] at least once in their lives.",Blog posts, reports, presentations
Quotation,"[Famous Person] once said, '[Quotation related to Topic]'.",Speeches, essays, social posts
Challenge,"Most people believe [Common Belief], but [Contradictory Statement].",Debates, persuasive content, op-eds
Visual Imagery,"Imagine [Vivid Description of a Scenario].",Creative writing, advertising, storytelling
Call-to-Action,"If you’ve ever [Experience/Desire], then [Action to take].",Marketing content, motivational speeches, campaigns
Historical Reference,"Back in [Year/Period], [Historical Event] changed the way we think about [Topic].",Educational content, documentaries, historical analyses
Anecdotal,"Once, [Short Anecdote related to Topic].",Personal blogs, speeches, narrative content
Humorous,"Why did [Topic] cross the road? To [Punchline].",Social media, entertaining content, ice-breakers
Controversial Statement,"[Controversial Statement about a Topic].",Debates, opinion pieces, discussion forums
Rhetorical Question,"Have you ever stopped to think about [Thought-Provoking Question]? ",Speeches, persuasive essays, social posts
"
The JSON object:\n\n"""



input_topic = "AI tools"
input_usage = "short video"

input_prompt = hook_generator_prompt.format(user_input=input_topic,usage=input_usage)

generated_text = llm_instance.generate_response(prompt=input_prompt)

print(generated_text)