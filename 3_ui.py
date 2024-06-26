# Install Streamlit if not already installed by running the following command in your terminal:
# pip install streamlit

# Importing necessary libraries
import streamlit as st
from SimplerLLM.language.llm import LLM, LLMProvider

# Create an instance of the LLM class with the specified provider and model
llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4o")

# Function to generate hooks based on user input
def generate_hooks(topic, usage):
    # The prompt template for the hook generation
    hook_generator_prompt = """
    as an expert copywriter specialized in hook generation, your task is to 
    analyze the [Provided_Hook_Examples].

    Use the templates that fit most to generate 3 new Hooks 
    for the following topic: {user_input} and Usage in: {usage}. 

    The output should be ONLY valid JSON as follows:
    [
      {{
        "hook_type": "The chosen hook type",
        "hook": "the generated hook"
      }},
      {{
        "hook_type": "The chosen hook type",
        "hook": "the generated hook"
      }},
      {{
        "hook_type": "The chosen hook type",
        "hook": "the generated hook"
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

    # Format the prompt with user input
    input_prompt = hook_generator_prompt.format(user_input=topic, usage=usage)

    # Generate the response using the LLM instance
    generated_text = llm_instance.generate_response(prompt=input_prompt)
    
    return generated_text

# Streamlit app
def main():
    # Set the title of the Streamlit app
    st.title("Hook Generator using GPT-4")

    # Input fields for the user to enter topic and usage
    topic = st.text_input("Enter the topic for hook generation:", "AI tools")
    usage = st.text_input("Enter the usage context:", "short video")

    # Button to trigger the hook generation process
    if st.button("Generate Hooks"):
        with st.spinner('Generating hooks...'):
            hooks = generate_hooks(topic, usage)
            st.success("Hooks generated successfully!")
            st.json(hooks)  # Display the generated hooks in JSON format

# Run the Streamlit app
if __name__ == "__main__":
    main()
