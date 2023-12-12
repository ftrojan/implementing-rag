import logging
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


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

logging.basicConfig(level="INFO", format="[%(levelname)s] %(asctime)s.%(msecs)03d %(name)s.%(funcName)s#%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)
logger.info("started")
docs = get_docs()
embeddings = get_embeddings()
logger.info("building FAISS db")
db = FAISS.from_documents(docs, embeddings)
question = "What is cheesemaking?"
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)
logger.info("completed")