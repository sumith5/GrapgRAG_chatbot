"""
Primary entry point for the RAG app. Integrates other RAG functionality into a UI
"""
import os
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI


# Neo4j Client Setup
os.environ["OPENAI_API_KEY"] = ""
os.environ["NEO4J_URI"] = ""
os.environ["NEO4J_USERNAME"] = ""
os.environ["NEO4J_PASSWORD"] = ""

llm = ChatOpenAI(model="gpt-4o", temperature=0)

import os
from typing import List
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader, TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI


class GraphBuilder():
    """
    Encapsulates the core functionality requires to build a full knowledge graph 
    from multiple sources of unstructured text

    _extended_summary_
    """
    def __init__(self):
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def chunk_document_text(self, raw_docs):
        """
        Accepts raw text context extracted from source and applies a chunking 
        algorithm to it. 

        Args:
            raw_docs (str): The raw content extracted from the source

        Returns:
            List: List of document chunks
        """
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        docs = text_splitter.split_documents(raw_docs[:3])
        return docs

    def graph_document_text(self, text_chunks):
        """
        Uses experimental LLMGraphTransformer to convert unstructured text into a knowledge graph

        Args:
            text_chunks (List): List of document chunks
        """
        llm_transformer = LLMGraphTransformer(llm=self.llm)

        graph_docs = llm_transformer.convert_to_graph_documents(text_chunks)
        self.graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=True,
            include_source=True
        )

    def chunk_and_graph(self, raw_docs):
        """
        Breaks the raw text into chunks and converts into a knowledge graph

        Args:
            raw_docs (str): The raw content extracted from the source
        """
        text_chunks = self.chunk_document_text(raw_docs)
        if text_chunks is not None:
            self.graph_document_text(text_chunks)



    def scrape_jina_ai(self,url):
      """
      Uses the jina api for webscraping the prolifics about us site
      Args:
            url (str) : The website link
      """
      response = requests.get("https://r.jina.ai/" + url)
      return response

    def extract_scrape_content(self):
      url  = "website link"
      content = self.scrape_jina_ai(url)
      print(content.text)
      path = "/content/scrape.txt"
      with open(path,'w') as f:
        f.writelines(content.text)
        f.close()
      self.graph_text_content(path)


    def graph_text_content(self, path):
        """
        Provided with a text document, will extract and chunk the text
        before generating a graph

        Args:
            path (str): Text document path
        """
        text_docs = TextLoader(path).load()
        print(text_docs)
        self.chunk_and_graph(text_docs)


    def index_graph(self):
        """
        Creates an index on the populated graph tp assist with efficient searches
        """
        self.graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    def reset_graph(self):
        """
        WARNING: Will clear entire graph, use with caution
        """
        self.graph.query(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Entities(BaseModel):
    """
    Identify and capture information about entities from text
    """

    names: List[str] = Field(
        description=
            "All the objects, person, organization, or business entities that appear in the text",
    )



from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

class GraphRAG():
    """
    Class to encapsulate all methods required to query a graph for retrieval augmented generation
    """

    def __init__(self):
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def create_entity_extract_chain(self):
        """
        Creates a chain which will extract entities from the question posed by the user. 
        This allows us to search the graph for nodes which correspond to entities more efficiently

        Returns:
            Runnable: Runnable chain which uses the LLM to extract entities from the users question
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting objects, person, organization, " +
                    "or business entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )

        entity_extract_chain = prompt | self.llm.with_structured_output(Entities)
        return entity_extract_chain

    def generate_full_text_query(self, input_query: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspellings.

        Args:
            input_query (str): The extracted entity name pulled from the users question

        Returns:
            str: _description_
        """
        full_text_query = ""

        # split out words and remove any special characters reserved for cipher query
        words = [el for el in remove_lucene_chars(input_query).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, question: str) -> str:
        """
        Creates a retriever which will use entities extracted from the users query to 
        request context from the Graph and return the neighboring nodes and edges related
        to that query. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The fully formed Graph Query which will retrieve the 
                 context relevant to the users question
        """

        entity_extract_chain = self.create_entity_extract_chain()
        result = ""
        entities = entity_extract_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def create_vector_index(self) -> Neo4jVector:
        """
        Uses the existing graph to create a vector index. This vector representation
        is based off the properties specified. Using OpenAIEmbeddings since we are using 
        GPT-4o as the model. 

        Returns:
            Neo4jVector: The vector representation of the graph nodes specified in the configuration
        """
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index

    def retriever(self, question: str) -> str:
        """
        The graph RAG retriever which combines both structured and unstructured methods of retrieval 
        into a single retriever based off the users question. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The retrieved data from the graph in both forms
        """
        print(f"Search query: {question}")
        vector_index = self.create_vector_index()
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        structured_data = self.structured_retriever(question)
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data

    def create_search_query(self, chat_history: List, question: str) -> str:
        """
        Combines chat history along with the current question into a prompt that 
        can be executed by the LLM to answer the new question with history.

        Args:
            chat_history (List): List of messages captured during this conversation
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The formatted prompt that can be sent to the LLM with question & chat history
        """
        search_query = ChatPromptTemplate.from_messages([
            (
                "system",
                """Given the following conversation and a follow up question, rephrase the follow 
                up question to be a standalone question, in its original language.
                Chat History:
                {chat_history}
                Follow Up Input: {question}
                Standalone question:"""
            )
        ])
        formatted_query = search_query.format(
            chat_history=chat_history, question=question)
        return formatted_query



def graph_content(progress_bar, status_text):
    """
    Entry point to generate a new graph. Will add controls to the UI 
    to perform these actions in the future
    """
    print("Building graph from content")
    graph_builder = GraphBuilder()
    
    graph_builder.extract_scrape_content()
    status_text.text("Complete scraping and content is extracted")
    graph_builder.index_graph()

def reset_graph():
    """
    Will reset the graph by deleting all relationships and nodes
    """
    graph_builder = GraphBuilder()
    graph_builder.reset_graph()

def get_response(question: str) -> str:
    """
    For the given question will formulate a search query and use a custom GraphRAG retriever 
    to fetch related content from the knowledge graph. 

    Args:
        question (str): The question posed by the user for this graph RAG

    Returns:
        str: The results of the invoked graph based question
    """
    rag = GraphRAG()
    search_query = rag.create_search_query(st.session_state.chat_history, question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": lambda x: rag.retriever(search_query),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Using invoke method to get response
    return chain.invoke({"chat_history": st.session_state.chat_history, "question": question})

def init_ui():
    """
    Primary entry point for the app. Creates the chat interface that interacts with the LLM. 
    """
    def get_state():
      session_state = st.session_state
      if "custom_state" not in session_state:
          session_state.custom_state = {}
      return session_state.custom_state
    state = get_state()
    st.set_page_config(page_title="Visionary ChatBot", layout="wide")
    st.title("Visionary ChatBot")
    st.write("A GrapgRag based Chatbot")
    st.sidebar.image(['/content/Prolificslogo.jpg'], width=220)

    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm here to help. Ask me anything!")
        ]

    user_query = st.chat_input("Ask a question....")
    if user_query is not None and user_query != "":
        response = get_response(user_query)

         # Add the current chat to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Print the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        if isinstance(message, AIMessage):
          with st.spinner("Thinking..."):
            with st.chat_message("AI"):
                st.write(message.content)

    with st.sidebar:
        st.header("Graph Management")
        st.write("Below are options to populate and reset your graph database")

        # Create two columns for the buttons
        if st.button("Populate Graph"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            graph_content(progress_bar, status_text)

if __name__ == "__main__":
    init_ui()
