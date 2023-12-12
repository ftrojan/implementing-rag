import logging
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline


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
llm = get_llm()
logger.info("completed")
