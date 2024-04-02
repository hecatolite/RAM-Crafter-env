<h1 align="center">RAM with Crafter</h1>

## ✏️ Table of Contents

- [📊 **Data**](#-data)
- [⚙ **Installation**](#-installation)
- [🚀 **Quick Start**](#-quick-start)
    - [**Step 1: Prerequisites**](#step-1-prerequisites)
    - [**Step 2: Data Preparation**](#step-2-data-preparation)
    - [**Step 3: Inference**](#step-3-inference)
- [👩‍🏫 **Different types of feedback in RAM**](#-different-types-of-feedback-in-ram)
- [🔍 **Different retrieval strategies**](#-different-retrieval-strategies)
- [**Human Study**](#human-study)
    - [**Interactivate Interface**](#interactivate-interface)
    - [**Result**](#result)
- [📝 **Citation**](#-citation)
- [📣 **Contacts**](#-contacts)

## 📊 Data

We use two datasets FreshQA and MQuAKE and prepare relevant knowledge accordingly as below:

```
RAM/data
├─freshqa_2104 # old knowledge of FreshQA before April 2021
├─mquake_2104 # old knowledge of MQuAKE before April 2021
│  
│  freshqa.csv # QA pairs selected with source from Wikipedia from FreshQA for use
│  freshqa_groundtruth.txt # ground truth of FreshQA
│  freshqa_IDs.txt # IDs of questions selected from original FreshQA dataset for use
│  freshqa_QR.json # related question mappings in FreshQA
│  mquake.csv # QA pairs sampled from original MQuAKE for use
│  mquake_groundtruth.json # related question mappings in FreshQA
```

## ⚙ Installation

```bash
# clone RAM
git clone https://github.com/bigai-nlco/RAM.git
cd RAM

#create conda env
conda create -n RAM
conda activate RAM

# install package
pip install -r requirements.txt

# export openai key
export OPENAI_API_KEY="[your_openai_api_key]"
```

## 🚀 Quick Start

### Step 1: Prerequisites

In this work, we use **all-MinLM-L6-v2** to construct database, and use **bert-base-nli-mean-tokens** to calculate the
semantic similarity of feedback and ground truth.

You can download these two models from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [bert-base-nli-mean-tokens](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens). You should put
all-MiniLM-L6-v2 and bert-base-nli-mean-tokens at the path inference/.

If you want to use other models, you can change them at line 81 and line 88 in inference/llm.py.

### Step 2: Data Preparation

Before you start the project, you should encode each article in datasets as an embedding in the vector database
ChromaDB.

```
python data_preparation.py --vectordb_name chroma_db
```

### Step 3: Inference

We test LLMs using Python codes under the path inference/ with four methods in our work. We select the method for
evaluation via --method and the specific dataset via --dataset. Let's take FreshQA as an example:

For Self-knowledge:

```
python inference/base.py --dataset freshqa --result_path ./result/freshqa_self_knowledge.json --method Self-knowledge --model_path ./model/Llama-2-7b-chat-hf
```

For RAG-only:

```
python inference/base.py --dataset freshqa --result_path ./result/freshqa_rag_only.json --method RAG-only --vectordb_name chroma_db --model_path ./model/Llama-2-7b-chat-hf
```

For RAM-R<sup>3</sup>:

```
python inference/RAM.py --dataset freshqa --result_path ./result/freshqa_ram.json --method RAM_R3 --vectordb_name chroma_db --model_path ./model/Llama-2-7b-chat-hf 
```

For RAM:

```
python inference/RAM.py --dataset freshqa --result_path ./result/freshqa_ram.json --method RAM --vectordb_name chroma_db --model_path ./model/Llama-2-7b-chat-hf 
```

For RAG-upd:

```
python inference/base.py --dataset freshqa --result_path ./result/freshqa_rag_upd.json --method RAG-upd --vectordb_name chroma_db --model_path ./model/Llama-2-7b-chat-hf
```

Open-source models can be downloaded and loaded from models/ by default, you can change the path via --model_path. You
can download
the [LlaMa-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [LlaMa-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
and [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) from Hugginig Face. For Vicuna-7B, you should modify prompt
according to instruct tuning. You can also determine the output result through --result_path.

The implementation parameters of models and metrics are in default settings. All the experiments can be run on 1 A100
each with 80G GPU

## 👩‍🏫 Different types of feedback in RAM

We test the performance using different types of feedback by using Python codes under feedback/ with three methods, and
we select method via --feedback_method.

For Hints:

Hints is a basic setting in our experimental method, as described in "For RAM" in Step 3 of Quick Start.

For Direct answer:

```
python feedback/RAM.py --dataset freshqa --result_path ./result/freshqa_DA.json --vectordb_name chroma_db --model_name Llama-2-7b --model_path ./model/Llama-2-7b-chat-hf --feedback_method Direct_answer
```

For No feedback:

```
python feedback/RAM.py --dataset freshqa --result_path ./result/freshqa_DA.json --vectordb_name chroma_db --model_name Llama-2-7b --model_path ./model/Llama-2-7b-chat-hf --feedback_method NA
```

## 🔍 Different retrieval strategies

We also evaluate RAM using four distinct retrieval strategies: Default(Stuff), Map-reduce, Refine, and Map-rerank. For
detailed information about the settings of the above strategies, please refer
to [this link](https://www.langchain.com.cn/modules/chains/index_examples/qa_with_sources).

You can replace lines 256 to 268 in inference/agents.py with the code implementations below, except for Default setting.

```python
docs = self.retriever.get_relevant_documents(argument)
tmp = []
for i in docs:
    doc_details = i.to_json()['kwargs']
    tmp.append(doc_details['metadata']['title'])
self.title = tmp
chain = load_qa_chain(self.llm, chain_type="map_reduce")  ## Replace "chain_type" with either "refine" or "map_rerank"
result = chain.run(input_documents=docs, question=self.hfeedback + ' ' + self.question)
```

## Human Study

### Interactivate Interface

The graphical interactive interfaces for RAM are presented below.

![Human_Study](./assets/demo.gif)

### Result

The figure below illustrates the effectiveness of communicative learning with real users across two datasets and four
distinct settings.

<div align="center">
  <img src="./assets/human_00.png" width="500px">
</div>

## 📝 **Citation**

If you would like to use our data or find our work interesting, please cite:

```bibtex
@article{li2024ram,
  title={RAM: Retrieval-augmented Generation As Continually Updated Memory via Communicative Learning},
  author={Li, Jiaqi and Wang, Xiaobo and Wang, Zihao and Zheng, Zilong },
  journal={arXiv preprint arXiv: },
  year={2024}
}
```

## 📣 **Contacts**

We sincerely appreciate users for their valuable contributions to human study in RAM.
We are very pleased to answer any questions about RAM: [nlp@bigai.ai](mailto:nlp@bigai.ai)

