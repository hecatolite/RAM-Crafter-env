import warnings

warnings.filterwarnings("ignore")
import os
import openai
import csv
import json
from time import time
from torch import cuda, bfloat16
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import tiktoken
import chromadb
import argparse

openai.api_key = 'sk-m3zHFtxVcENW9jaw0349Ed70C1914f98BbB5186aB67a1dD3'
openai.api_base = 'https://pro.aiskt.com/v1'

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None, help="the dataset you train")
    parser.add_argument('--result_path', type=str, default=None, help="result save path")
    parser.add_argument('--vectordb_name', type=str, default=None, help="the name of the vectordb")
    parser.add_argument('--method', type=str, default=None, help="method")
    parser.add_argument('--model_path', type=str, default=None, help="the model path")

    return parser.parse_args(args)


def GPT_assis(Q, A, P):
    args = "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. \
    Please output True or False only. If there are expressions like 'I don't know' or 'I cant find' or 'The text doesn't provide' or 'it is not provided in the text', \
    then output 'None'." + "Question: " + Q + ' ' + "groudtruth = " + A + ' ' + "predict_answer = " + P
    args = args.replace('"', '^')
    while True:
        if '\n' in args:
            args = args.strip().replace('\n', ' ')
        else:
            break
    prompt = [{"role": "user", "content": args}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=prompt,
        max_tokens=100
    )

    return response.choices[0].message.content


def get_vectordb(re_no=4, df_collection_name="wiki_mquake_2104",
                 df_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'database/chroma_db'),
                 df_model_name=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            'inference/all-MiniLM-L6-v2')):
    embeddings = SentenceTransformerEmbeddings(model_name=df_model_name)
    chroma = chromadb.PersistentClient(path=df_path)
    collection = chroma.get_collection(df_collection_name)
    vectordb = Chroma(
        client=chroma,
        collection_name=df_collection_name,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": re_no, "filter": {'source': {'$nin': ['shortmem']}}})
    return retriever, collection


def RAG(retriever, llm, q, enc=tiktoken.encoding_for_model("text-davinci-003")):
    docs = retriever.get_relevant_documents(q)
    doc_details = docs[0].to_json()['kwargs']
    text, title = doc_details['page_content'], doc_details['metadata']['title']
    if len(enc.encode(text)) > 4000:
        docs[0].page_content = text[:4000]
    print('【', title, '】', '【', docs[0].page_content[:100], '】')

    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=[docs[0]], question=q)
    return result, title


def QA(ID, q, a, llm, retriever, mtd):
    print('===========', ID, q, a)
    title, turns, step, feedback = '', '', '', ''
    answer[ID] = {}
    answer[ID]['Q'] = q
    answer[ID]['A'] = a
    answer[ID]['P'], answer[ID]['Time'], answer[ID]['R'], answer[ID]['Eval'] = [], [], [], []
    answer[ID]['Turn'], answer[ID]['Step'], answer[ID]['Feedback'] = '', '', ''
    time_1 = time()
    if mtd == "RAG-only" or mtd == "RAG-upd":
        prediction, title = RAG(retriever, llm, q)
    elif mtd == "Self-knowledge":
        prediction = llm(q)

    time_2 = time()
    timecost = round(time_2 - time_1, 3)
    while True:
        if '\n' in prediction:
            prediction = prediction.strip().replace('\n', '')
        else:
            break
    print(prediction, '\n')
    tag = GPT_assis(q, a, prediction)
    print(tag)
    answer[ID]['Eval'].append(tag)
    answer[ID]['P'].append(prediction)
    answer[ID]['Time'].append(timecost)
    answer[ID]['R'].append(title)
    answer[ID]['Turn'] = turns
    answer[ID]['Step'] = step
    answer[ID]['Feedback'] = feedback


if __name__ == '__main__':
    args = parse_args()

    model = args.model_path
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = AutoConfig.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
    )

    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        repetition_penalty=1.1,
        top_p=1,
        temperature=0,
        max_new_tokens=300,
        device_map="auto", )

    llm = HuggingFacePipeline(pipeline=query_pipeline)

    if args.method == "RAG-only" or args.method == "RAG-upd":
        retriever, collection = get_vectordb(df_collection_name='wiki_' + args.dataset + '_2104',
                                             df_path=os.path.join(
                                                 os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 'database/', args.vectordb_name))
    else:
        retriever, collection = None, None

    dataset = args.dataset

    if dataset == 'freshqa':
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                               f'{dataset}_groundtruth.txt'),
                  'r') as f:
            loader = f.readlines()
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'freshqa_IDs.txt'),
                  'r') as f:
            white = list(eval(f.read()))
    else:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                               f'{dataset}_groundtruth.json'),
                  'r') as f:
            loader = json.loads(f.read())

    filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', f'{dataset}.csv')
    path = args.result_path

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        with open(path, 'r') as f:
            answer = json.loads(f.read())
    except:
        answer = {}

    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:

            if dataset == 'freshqa':
                ID, q = int(row[0]), row[2]
                a = loader[ID].strip('\n')
                if ID >= 0 and ID in white:
                    QA(ID, q, a, llm, retriever, mtd=args.method)
                    with open(path, 'w') as g:
                        g.write(json.dumps(answer))

            elif dataset == 'mquake':
                ID, q = int(row[0]), row[1]
                a = loader[str(ID)].strip('\n')
                if ID >= 0:
                    QA(ID, q, a, llm, retriever, mtd=args.method)
                    with open(path, 'w') as g:
                        g.write(json.dumps(answer))
