from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

class TextSummarizer():
    def __init__(self, model) -> None:
        self.model = model
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", "\n\n"],
            chunk_size=10000,
            chunk_overlap=1000
        )
    
    def create_summary(self, objective: str, content: str):
        template = """
        %INSTRUCTIONS:
        Write a summary of the following text for {objective}:
        Respond in a highly engaging manner as a world-class car documentalist. Responses should be highly engaging and informative.
        Responses should be easy to understand.
        %TEXT:  
        {text}
        """

        num_tokens = self.model.get_num_tokens(content)
        print(f"There are {num_tokens} tokens in the content")

        docs = self.text_splitter.create_documents([content])
        print(f"There are {len(docs)} documents")



        prompt = PromptTemplate.from_template(template)
        prompt.format(objective=objective, text=content)

        summary_chain = load_summarize_chain(
            llm=self.model,
            chain_type='map_reduce',
            verbose=True,
        )

        output = summary_chain.run(docs)
        print(f"SUMMARY OUTPUT for {objective}: {output}".format(objective, output))

        return output