import logging
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from transformers import SquadExample
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA


def get_docs():
    # Specify the dataset name and the column containing the content
    dataset_name = "databricks/databricks-dolly-15k"
    page_content_column = "context"  # or any other column you're interested in

    # Create a loader instance
    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

    # Load the data
    data = loader.load()

    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of 1000 characters each with a 150-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    docs = text_splitter.split_documents(data)
    return docs


def get_embeddings():
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': 'mps'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    return embeddings


def get_llm():
    # Specify the model name you want to use
    model_name = "Intel/dynamic_tinybert"

    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        "question-answering", 
        model=model_name, 
        tokenizer=tokenizer,
        return_tensors='pt'
    )

    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm = HuggingFacePipeline(
        pipeline=question_answerer,
        model_kwargs={"temperature": 0.7, "max_length": 512},
    )
    return llm


logging.basicConfig(level="INFO", format="[%(levelname)s] %(asctime)s.%(msecs)03d %(name)s.%(funcName)s#%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)
logger.info("started")
docs = get_docs()
embeddings = get_embeddings()
question = "Who is Thomas Jefferson?"
squad_example = SquadExample(qas_id=None, question_text=question, context_text="", title=None, answer_text=None, start_position_character=None)
logger.info("building FAISS db")
db = FAISS.from_documents(docs, embeddings)
# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

# Create a question-answering instance (qa) using the RetrievalQA class.
# It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
llm = get_llm()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

# Finally, we call this QA chain with the question we want to ask.
result = qa({"query": question})
with open("result.txt", "w") as fp:
    fp.write(result["result"])
logger.info("completed")