import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.utilities.sql_database import SQLDatabase


GROQ_API_KEY = "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"
# print("This is api--------", GROQ_API_KEY)

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    For any queries related to customers table use dbo.customers as a table name 
    for table names use dbo."tablename" 

    For example:
    Question: Retrieve the first names of the top 5 customers from the customers table
    SQL Query: SELECT TOP 5 first_name FROM dbo.customers;
    Question: Name 10 product name
    SQL Query: SELECT TOP 10 product_name FROM dbo.products;
    
    Make sure write a answer in a well formatted readable form. Add proper bullet points(numeric).
    Your turn:
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Please use group name along with their table name to refer to any table.

    Conversation History: {chat_history}
    MSSQL Query: <SQL>{query}</SQL>
    User question: {question}
    MSSQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=GROQ_API_KEY)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain)
        .assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()
st.set_page_config(page_title="Chat with MS SQL", page_icon=":speech_balloon:")

st.title("Chat with MS SQL")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MS SQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="1433", key="Port")  # Default MS SQL port is 1433
    st.text_input("User", value="sa", key="User")
    st.text_input("Password", type="password", value="password", key="Password")
    st.text_input("Database", value="master", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))

