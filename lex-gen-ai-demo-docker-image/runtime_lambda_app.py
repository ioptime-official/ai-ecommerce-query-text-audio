import boto3
from botocore.exceptions import ClientError
import logging
import json
import os
from typing import Optional, List, Mapping, Any
# from langchain.llms.base import LLM
# from llama_index import (
#     LangchainEmbedding,
#     PromptHelper,
#     ResponseSynthesizer,
#     LLMPredictor,
#     ServiceContext,
#     Prompt,
# )
polly_client = boto3.client('polly')
import base64
voice_id = 'Joanna'

# from llama_index.response_synthesizers import get_response_synthesizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from typing import Dict
from langchain import SagemakerEndpoint
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.chains.question_answering import load_qa_chain
import json

from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
import pinecone
import os
# from llama_index.query_engine import RetrieverQueryEngine
# from llama_index.retrievers import VectorIndexRetriever
# from llama_index.vector_stores.types import VectorStoreQueryMode
# from llama_index import StorageContext, load_index_from_storage

s3_client = boto3.client('s3')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENDPOINT_NAME = "huggingface-pytorch-sagemaker-endpoint61"
OUT_OF_DOMAIN_RESPONSE = "I'm sorry, but I am only able to give responses regarding the source topic"
newout="this is message that llm not ecounter"

ACCOUNT_ID = boto3.client('sts').get_caller_identity().get('Account')
INDEX_BUCKET = "lexgenaistack-created-index-bucket-"+ACCOUNT_ID
INDEX_WRITE_LOCATION = "/tmp/index"
RETRIEVAL_THRESHOLD = 0.4        
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

# embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# pinecone.init(
# 	api_key='',
# 	environment='gcp-starter'
# )
# index = pinecone.Index('llamapdf')

# # initialize pinecone
# pinecone.init(
# 	api_key='',
# 	environment='gcp-starter'  # next to api key in console
# )
# index_name = "llamapdf" # put in the name of your pinecone index here


# docsearch = Pinecone.from_existing_index(index_name, embeddings)
# logger.info('---successfully picorn runing====')




def handler(event, context):
    initialize_cache()

    # embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # logger.info('embedding model used',embeddings)
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',cache_folder="/tmp/HF_CACHE")
    pinecone.init(
	api_key='',
	environment='gcp-starter'
    )
    index = pinecone.Index('llamapdf')

    # initialize pinecone
    pinecone.init(
        api_key='',
        environment='gcp-starter'  # next to api key in console
    )
    index_name = "llamapdf" # put in the name of your pinecone index here


    docsearch = Pinecone.from_existing_index(index_name, embed_model)
    logger.info('---successfully picorn runing====')

    logger.info("========below is the prompt template code============")
    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    """.strip()


    def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f"""
    [INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {prompt} [/INST]
    """.strip()
    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end.If you don't know the answer, just say that you don't know, don't try to make up an answer."

    template = generate_prompt(
        """
    {context}


    Question: {question}
    provide URL with answer provide URL with answer but not as clickable link., Do not mention the authenticity or working of the URLs.

    """,
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    logger.info('this is the ----- prompt value')
    logger.info(prompt)
    content_handler = ContentHandler()
    

    logger.info("----below is llm sagemaker code")
    chain = load_qa_chain(
        llm=SagemakerEndpoint(
            endpoint_name=ENDPOINT_NAME,
            # credentials_profile_name='default',
            region_name="us-east-1",
            model_kwargs={"max_new_tokens":100,
                            "do_sample":True,
                            "temperature":0.7,
                            "top_p":0.95,
                            "top_k":40,
                            "repetition_penalty":1.13,
                            "stop": ["\nUser:", "<|endoftext|>", "</s>"]},
            content_handler=content_handler,
        ),
        prompt=prompt,
    )
    query_input = event["inputTranscript"]
    logger.info("---insidee quervy input")
    logger.info(query_input)


    docs=docsearch.similarity_search(query_input,k=2)
    logger.info("---this is docs file")
    logger.info(docs)
    x=chain({"input_documents": docs, "question": query_input})
    logger.info('-------x values printed and its datatype')
    logger.info(x)
    logger.info(type(x))
    output_text_value = x.get('output_text', '')
    logger.info("--belwo is outpit_textvalue---0")
    logger.info(output_text_value)
# Find the index of [INST]
    inst_index = output_text_value.find('[/INST]')

# Extract the substring after [INST]
    if inst_index != -1:
        logger.info('--inside the inst_index---')
        result_string = output_text_value[inst_index + len('[/INST]'):].strip()
        logger.info('---result string value after cleaining---')
        logger.info(result_string)
    else:
        logger.info("[INST] not found in the 'output_text' value.")

    try:
        answer = result_string
        logger.info('in the try statement')
        logger.info(answer)
    except Exception as e:
        logger.error(f"An exception occurred: {str(e)}")
        answer = newout
        

    response = generate_lex_response(event, {}, "Fulfilled", answer)
    
    jsonified_resp = json.loads(json.dumps(response, default=str))
    logger.info("jesonfied response00====")
    logger.info(jsonified_resp)
    return jsonified_resp

def generate_lex_response(intent_request, session_attributes, fulfillment_state, message):
    intent_request['sessionState']['intent']['state'] = fulfillment_state
    return {
        'sessionState': {
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close'
            },
            'intent': intent_request['sessionState']['intent']
        },
        'messages': [
            {
                "contentType": "PlainText",
                "content": message
            }
        ],
        'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
    }

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        logger.info('---inside input function---')
        input_str = json.dumps({'inputs': prompt, 'parameters': model_kwargs})
        logger.info(input_str)
        # input_str = json.dumps({'inputs': prompt, **model_kwargs})
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        logger.info("---inside transformoutput---")
        response_json = json.loads(output.read().decode("utf-8"))
        logger.info(response_json)
        return response_json[0]["generated_text"]
    



def initialize_cache():
    logger.info("-------temp files areas code enter-----")
    if not os.path.exists("/tmp/TRANSFORMERS_CACHE"):
        os.mkdir("/tmp/TRANSFORMERS_CACHE")

    if not os.path.exists("/tmp/HF_CACHE"):
        os.mkdir("/tmp/HF_CACHE")



        ########################## below is the langchain code...



# from langchain.embeddings import HuggingFaceEmbeddings

# from langchain.vectorstores import Pinecone

# from sentence_transformers import SentenceTransformer

# import pinecone
# import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

# embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# pip install pinecone-client

# import pinecone

# pinecone.init(
# 	api_key='',
# 	environment='gcp-starter'
# )
# index = pinecone.Index('llamapdf')

# # initialize pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,  # find at app.pinecone.io
#     environment=PINECONE_API_ENV  # next to api key in console
# )
# index_name = "llamapdf" # put in the name of your pinecone index here

# docsearch = Pinecone.from_existing_index(index_name, embeddings)

# query="i want to buy ladies sweetshirt."
# docs=docsearch.similarity_search(query,k=2)

# docs

# from multiprocessing import context
# x=chain.run(input_documents=docs, question=query)

# x




# llama index code below
# import boto3
# from botocore.exceptions import ClientError
# import logging
# import json
# import os
# from typing import Optional, List, Mapping, Any
# from langchain.llms.base import LLM
# from llama_index import (
#     LangchainEmbedding,
#     PromptHelper,
#     ResponseSynthesizer,
#     LLMPredictor,
#     ServiceContext,
#     Prompt,
# )
# polly_client = boto3.client('polly')
# import base64
# voice_id = 'Joanna'
# # from llama_index.response_synthesizers import get_response_synthesizer
# from langchain.embeddings import HuggingFaceEmbeddings
# from llama_index.query_engine import RetrieverQueryEngine
# from llama_index.retrievers import VectorIndexRetriever
# from llama_index.vector_stores.types import VectorStoreQueryMode
# from llama_index import StorageContext, load_index_from_storage

# s3_client = boto3.client('s3')

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# ENDPOINT_NAME = "huggingface-pytorch-sagemaker-endpoint53"
# OUT_OF_DOMAIN_RESPONSE = "I'm sorry, but I am only able to give responses regarding the source topic"
# newout="this is message that llm not ecounter"

# ACCOUNT_ID = boto3.client('sts').get_caller_identity().get('Account')
# INDEX_BUCKET = "lexgenaistack-created-index-bucket-"+ACCOUNT_ID
# INDEX_WRITE_LOCATION = "/tmp/index"
# RETRIEVAL_THRESHOLD = 0.4        

# # define prompt helper
# max_input_size = 400  # set maximum input size
# num_output = 50  # set number of output tokens
# max_chunk_overlap = 0  # set maximum chunk overlap
# prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
# # Specify the voice ID (e.g., 'Joanna' for English (US))
# voice_id = 'Joanna'

# # Specify the S3 bucket name and object key
# s3_bucket_name = 'audioali'
# object_key = 'audio/sample.mp3'


# def handler(event, context):

#     # lamda can only write to /tmp/
#     initialize_cache()

#     # define our LLM
#     llm_predictor = LLMPredictor(llm=CustomLLM())
#     embed_model = LangchainEmbedding(HuggingFaceEmbeddings(cache_folder="/tmp/HF_CACHE"))
#     service_context = ServiceContext.from_defaults(
#         llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model,
#     )

#     ### Download index here
#     if not os.path.exists(INDEX_WRITE_LOCATION):
#         logger.info("index written directory is making")
#         os.mkdir(INDEX_WRITE_LOCATION)
#     try:
#         logger.info("Try to copying file form index to this folder")
#         s3_client.download_file(INDEX_BUCKET, "docstore.json", INDEX_WRITE_LOCATION + "/docstore.json")
#         s3_client.download_file(INDEX_BUCKET, "index_store.json", INDEX_WRITE_LOCATION + "/index_store.json")
#         s3_client.download_file(INDEX_BUCKET, "vector_store.json", INDEX_WRITE_LOCATION + "/vector_store.json")

#         # load index
#         storage_context = StorageContext.from_defaults(persist_dir=INDEX_WRITE_LOCATION)
#         index = load_index_from_storage(storage_context, service_context=service_context)
#         logger.info("Index successfully loaded")
#     except ClientError as e:
#         logger.error(e)
#         return "ERROR LOADING/READING INDEX"

#     retriever = VectorIndexRetriever(
#         service_context=service_context,
#         index=index,
#         similarity_top_k=2,
#         vector_store_query_mode=VectorStoreQueryMode.DEFAULT,  # doesn't work with simple
#         alpha=0.5,
#     )


#     synth = ResponseSynthesizer.from_args(
#         response_mode="simple_summarize",
#         service_context=service_context
#     )

#     query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
#     query_input = event["inputTranscript"]
#     logger.info("query variable")
#     logger.info(query_input)

#     try:
#         answer = query_engine.query(query_input)
#         logger.info('in the try statement')
#         logger.info(answer)
#         if answer.source_nodes[0].score < RETRIEVAL_THRESHOLD:
#             answer = OUT_OF_DOMAIN_RESPONSE
#     except Exception as e:
#         logger.error(f"An exception occurred: {str(e)}")
#         answer = newout
        
        
    

#     response = generate_lex_response(event, {}, "Fulfilled", answer)
    
#     jsonified_resp = json.loads(json.dumps(response, default=str))
#     logger.info("jesonfied response00====")
#     logger.info(jsonified_resp)
#     return jsonified_resp



# def generate_audio_from_text(text):
#     response = polly_client.synthesize_speech(
#         Text=text,
#         OutputFormat='mp3',
#         VoiceId=voice_id
#     )

#     # Save the audio file locally
#     audio_file_path = "/tmp/response_audio.mp3"
#     with open(audio_file_path, 'wb') as file:
#         file.write(response['AudioStream'].read())
#         logger.info("here is audio_file in response file")
#         logger.info(response['AudioStream'])
#     return audio_file_path
# def generate_lex_response(intent_request, session_attributes, fulfillment_state, message):
#     intent_request['sessionState']['intent']['state'] = fulfillment_state
#     return {
#         'sessionState': {
#             'sessionAttributes': session_attributes,
#             'dialogAction': {
#                 'type': 'Close'
#             },
#             'intent': intent_request['sessionState']['intent']
#         },
#         'messages': [
#             {
#                 "contentType": "PlainText",
#                 "content": message
#             }
#         ],
#         'requestAttributes': intent_request['requestAttributes'] if 'requestAttributes' in intent_request else None
#     }

# # define prompt template
# template = (
#     "We have provided context information below. \n"
#     "---------------------\n"
#     "CONTEXT1:\n"
#     "{context_str}\n\n"

#     "\n---------------------\n"
#     'Given this context, please answer the question if answerable based on on the CONTEXT1  "{query_str}"\n; '  # otherwise specify it as CANNOTANSWER
# )
# my_qa_template = Prompt(template)
# logger.info("my-qa response")
# logger.info(my_qa_template)

# def call_sagemaker(prompt, endpoint_name=ENDPOINT_NAME):
#     logger.info("inside payload prompt")
#     logger.info(prompt)
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "do_sample": True,
#             "top_p": 0.9,
#             "temperature": 0.5,
#             "max_new_tokens": 150,
#             "repetition_penalty": 1.03,
#             "stop": ["\nUser:", "<|endoftext|>", "</s>"]
#         }
#     }

#     sagemaker_client = boto3.client("sagemaker-runtime")
#     payload = json.dumps(payload)
#     response = sagemaker_client.invoke_endpoint(
#         EndpointName=endpoint_name, ContentType="application/json", Body=payload
#     )
#     response_string = response["Body"].read().decode()
#     return response_string

# def get_response_sagemaker_inference(prompt, endpoint_name=ENDPOINT_NAME):
#     resp = call_sagemaker(prompt, endpoint_name)
#     resp = json.loads(resp)[0]["generated_text"][len(prompt):]
#     return resp

# class CustomLLM(LLM):
#     model_name = "gpt2"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         logger.info("prompt value")
#         logger.info(prompt)
#         response = get_response_sagemaker_inference(prompt, ENDPOINT_NAME)
#         return response

#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         return {"name_of_model": self.model_name}

#     @property
#     def _llm_type(self) -> str:
#         return "custom"
    
# def initialize_cache():
#     if not os.path.exists("/tmp/TRANSFORMERS_CACHE"):
#         os.mkdir("/tmp/TRANSFORMERS_CACHE")

#     if not os.path.exists("/tmp/HF_CACHE"):
#         os.mkdir("/tmp/HF_CACHE")