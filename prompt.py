from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template

custom_instructions = """
You are a real estate expert.
Only answer questions related to properties.
Always mention price, location and size.
Be professional.

"""
new_template = custom_instructions + template

PROMPT = PromptTemplate(
    template=new_template,
    input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)