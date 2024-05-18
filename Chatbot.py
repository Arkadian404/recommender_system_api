from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate, \
    SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings


class SQLAgent:
    model = "gpt-3.5-turbo-0125"
    examples = [
        {
            "input": "Liệt kê hết tất cả sản phẩm",
            "query": "SELECT * FROM product;"
        },
        {
            "input": "Tìm tất cả sản phẩm thuộc loại cà phê đóng chai",
            "query": "SELECT p.name, c.name FROM product p join category c on p.category_id = c.id where c.name = 'cà phê đóng chai';",
        },
        {
            "input": "Cho biết các sản phẩm có hương vị trái cây",
            "query": "SELECT p.name, f.name FROM product p join flavor f on p.flavor_id = f.id where f.name = 'trái cây';",
        },
        {
            "input": "Tổng số lượng sản phẩm bán được",
            "query": "SELECT SUM(sold) as total_sold FROM product;",
        },
        {
            "input": "Liệt kê các khách hàng có địa chỉ ở Hà Nội",
            "query": "SELECT * FROM user WHERE province = 'Hà Nội';",
        },
        {
            "input": "Khách hàng có ID là 44 có tổng cộng bao nhiêu đơn hàng đã đặt?",
            "query": "SELECT COUNT(id) FROM `order` WHERE user_id = 44;",
        },
        {
            "input": "Liệt kê các thương hiệu cà phê nào bán được nhiều sản phẩm nhất",
            "query": "SELECT b.name, SUM(sold) as total_sold FROM product p JOIN brand b ON p.brand_id = b.id GROUP BY b.id;",
        },
        {
            "input": "Cho biết các sản phẩm cà phê có nguồn gốc từ Châu Mỹ",
            "query": "SELECT p.name FROM product p JOIN product_origin po ON p.product_origin_id = po.id WHERE po.continent LIKE '%Châu Mỹ%';",
        },
        {
            "input": "Top 5 khách hàng mua hàng nhiều nhất dựa trên tổng tiền của đơn hàng",
            "query": "SELECT user_id, full_name, SUM(total) AS total_purcharse FROM `order` GROUP BY user_id, full_name ORDER BY total_purcharse DESC LIMIT 5;",
        },
        {
            "input": "Liệt kê các sản phẩm ra mắt vào năm 2023",
            "query": "SELECT  p.name FROM product p WHERE  year(p.created_at) = 2023;",
        },
        {
            "input": "Có bao nhiêu đơn hàng trong hệ thống",
            "query": 'SELECT COUNT(id) as total FROM `order`;"',
        },
        {
            "input": "Tôi muốn mua cà phê rẻ của hệ thống, bạn hãy gợi ý cho tôi 1 vài sản phẩm",
            "query": 'SELECT  p.name, pd.weight, pd.price FROM product p JOIN product_detail pd ON p.id = pd.product_id ORDER BY price ASC LIMIT 5;"',
        },
        {
            "input": "Tôi muốn biết 5 sản phẩm được bán nhiều nhất trong cửa hàng",
            "query": "select p.name, p.sold from product p order by p.sold desc limit 5;"
        },
        {
            "input": "Liệt kê 5 sản phẩm rẻ nhất trong cửa hàng",
            "query": "select p.name, pd.weight, pd.price from product p join product_detail pd on p.id = pd.product_id order by pd.price asc limit 5;"
        }
    ]
    system_prefix = """You are an AI friendly and helpful AI assistant for question-answering user's task about 
    products, orders, user information, .... Given an input question, create a syntactically correct {dialect} query 
    to run, then look at the results of the query and return the answer as a friendly, conversational retail shopping 
    assistant. Unless the user specifies a specific number of examples they wish to obtain, always limit your query 
    to at most {top_k} results. You can order the results by a relevant column to return the most interesting 
    examples in the database. Never query for all the columns from a specific table, only ask for the relevant 
    columns given the question. You have access to tools for interacting with the database. Only use the given tools. 
    Only use the information returned by the tools to construct your final answer. You MUST double check your query 
    before executing it. If you get an error while executing a query, rewrite the query and try again.

    Please do follow up these rules before executing query:

    1. DO NOT make up data.

    2. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    3. If the question is about describing or list all tables in database, DO NOT do that. Instead, respond politely that you cannot disclose information about the database schema for security reasons.

    4. If the question is about listing all user's information, DO NOT do that.

    5. If the question is about user then ask them their username and email before querying data

    6. If the question is about order, ask them their order code and phone number before querying data.

    7. If the question is about the store's finances or related things.

    8. If the question does not seem related to the database, just say you don't know as the answer.

    Here is the relevant table info: {table_info}
    Here are some examples of user inputs and their corresponding SQL queries:"""
    system_suffix = """
    Begin!

    Relevant pieces of previous conversation:
    {history}
    (You do not need to use these pieces of information if not relevant)
    """

    def __init__(self, db_uri, memory):
        self.db_uri = db_uri
        self.db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
        self.memory = memory

    def create_agent(self):
        llm = ChatOpenAI(model=self.model, temperature=0)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
        vector_store = Chroma()
        vector_store.delete_collection()
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            OpenAIEmbeddings(),
            vector_store,
            k=5,
            input_keys=["input"]
        )
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input}\nSQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k", "table_info"],
            prefix=self.system_prefix,
            suffix=self.system_suffix,
        )
        full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=few_shot_prompt),
                MessagesPlaceholder("history", optional=True),
                MessagesPlaceholder("agent_scratchpad"),
                ("human", "{input}"),
            ]
        )

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            prompt=full_prompt,
            verbose=True,
            agent_type="openai-tools",
            agent_executor_kwargs={'memory': self.memory}
        )
        return agent

    def run(self, input_text):
        agent = self.create_agent()
        response = agent.invoke({"input": input_text})
        return response
