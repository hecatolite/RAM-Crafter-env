import re, string, os
from typing import List
from enum import Enum
import tiktoken
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM, get_similarity_encoder, get_vectordb, get_model
from prompts import reflect_prompt, react_agent_prompt, feedback_agent_prompt, react_reflect_agent_prompt, \
    memupdate_agent_prompt, decompose_agent_prompt
from prompts import REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, FEEDBACKS, UPDATES, DECOMPOSES
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains.question_answering import load_qa_chain
from rank_bm25 import BM25Okapi
import numpy as np
import openai
import logging
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

openai.api_key = 'sk-bIL5v0zxFsMH9CP02a37C2E5F59e4061864a5eBb234e979c'
openai.api_base = 'https://pro.aiskt.com/v1'


class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context
    REFLEXION: Apply inference to the next reasoning trace
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply inference to the next reasoning trace
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial'
    REFLEXION = 'inference'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'


class Memory_update:
    def __init__(self,
                 logger: logging.Logger,
                 args: argparse.Namespace,
                 question: str,
                 key: str,
                 observation: list,
                 hfeedback: str,
                 llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     max_tokens=100,
                     model_name="gpt-3.5-turbo",
                     model_kwargs={"stop": "\n"},
                     # openai_api_key=os.environ['OPENAI_API_KEY']
                 ),
                 memupdate_prompt: PromptTemplate = memupdate_agent_prompt,
                 decompose_prompt: PromptTemplate = decompose_agent_prompt,
                 sim_encoder=get_similarity_encoder(),
                 ) -> None:
        self.logger = logger
        self.args = args
        self.question = question
        self.key = key
        self.llm = llm
        self.memupdate_prompt = memupdate_prompt
        self.memupdate_examples = UPDATES
        self.decompose_prompt = decompose_prompt
        self.decompose_examples = DECOMPOSES
        self.retriever = get_vectordb(df_collection_name='wiki_' + args.dataset + '_2104',
                                      df_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                           'database',
                                                           args.vectordb_name))[0]
        self.collection = get_vectordb(df_collection_name='wiki_' + args.dataset + '_2104',
                                       df_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                            'database',
                                                            args.vectordb_name))[1]
        self.vectordb = get_vectordb(df_collection_name='wiki_' + args.dataset + '_2104',
                                     df_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                          'database',
                                                          args.vectordb_name))[2]
        self.observe = observation
        self.hfeedback = hfeedback
        self.mem = ''
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
        self.sim_encoder = sim_encoder

    def generate_shortmem(self) -> None:

        for i in self.observe:
            self.mem += i
        self.mem += self.hfeedback

        ## deduplicate
        l = self.mem.split('.')
        deduplicated_mem = '\n'.join(list(sorted(set(l), key=l.index)))
        updated_mem = format_step(self.llm(self._build_memupdate_prompt(deduplicated_mem, self.key)))
        if updated_mem == 'None':
            updated_mem = self.key

        self.logger.info('####Exist mem: ' + deduplicated_mem)
        self.logger.info('####Facts: ' + self.key)
        self.logger.info('####Update mem: ' + updated_mem)

        current_mem = '##question_seg##' + updated_mem

        upd_title, upd_text = self.generate_longmem(current_mem, 6)

        return upd_title, upd_text

    def generate_longmem(self, mem, rate) -> None:

        for i in mem.split('##question_seg##'):
            if i != '':
                i = format_step(i)
                new_mem = i

                documents = self.vectordb.similarity_search_with_score(i)
                docs = [d for d in documents if d[1] < 1.3]

                if len(docs) < 1:
                    # tag = jieba.analyse.extract_tags(sentence=i, topK=1)[0]
                    self.logger.info('========Mem_Insert========')
                    self.collection.upsert(
                        documents=[i],
                        metadatas=[{"title": i, "source": "wiki"}],
                        ids=["id_" + i])
                    return i, ''
                else:
                    sim_doc = docs[0][0].to_json()['kwargs']
                    sim_text, sim_title = sim_doc['page_content'], sim_doc['metadata']['title']
                    sentences = sim_text.replace(';', '.').split('.')
                    sim_score = bm25sim_match(sentences, i)
                    print('sentences: ', sentences)
                    print('i: ', i)

                    existing_mem = self.collection.get(ids=["id_" + sim_title])['documents'][0]
                    self.logger.info('========Mem_Update========' + sim_title)
                    for j in sentences:
                        if j != '':
                            if sim_score[sentences.index(j)] > rate:
                                update_kn = format_step(self.llm(self._build_memupdate_prompt(j, i)))
                                self.logger.info(sim_score[sentences.index(j)])
                                self.logger.info('1.' + j)
                                self.logger.info('2.' + update_kn)
                                if update_kn != 'None':
                                    new_mem += ' ' + update_kn
                                else:
                                    new_mem += ' ' + j
                            else:
                                new_mem += ' ' + j

                    self.collection.upsert(
                        documents=[new_mem],
                        metadatas=[{"title": sim_title, "source": "wiki"}],
                        ids=["id_" + sim_title])

                    return sim_title, existing_mem

    def _build_memupdate_prompt(self, existing_memory, acquired_fact) -> str:
        return self.memupdate_prompt.format(
            examples=self.memupdate_examples,
            existing_memory=existing_memory,
            acquired_fact=acquired_fact)

    def _build_decompose_prompt(self, existing_mem) -> str:
        return self.decompose_prompt.format(
            examples=self.decompose_examples,
            existing_mem=existing_mem)


class ReactAgent:
    def __init__(self,
                 logger: logging.Logger,
                 args: argparse.Namespace,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 feedback_prompt: PromptTemplate = feedback_agent_prompt,
                 interact_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     max_tokens=100,
                     model_name="gpt-4",
                     model_kwargs={"stop": "\n"},
                     openai_api_key=os.environ['OPENAI_API_KEY']
                 ),
                 sim_encoder=get_similarity_encoder()
                 ) -> None:
        self.logger = logger
        self.args = args
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6
        self.feedback_examples = FEEDBACKS
        self.interact_llm = interact_llm
        self.feedback_prompt = feedback_prompt
        self.llm = get_model(df_new_token=100, model=args.model_path)
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
        self.retriever = get_vectordb(df_collection_name='wiki_' + args.dataset + '_2104',
                                      df_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                           'database',
                                                           args.vectordb_name))[0]
        self.collection = get_vectordb(df_collection_name='wiki_' + args.dataset + '_2104',
                                       df_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                            'database',
                                                            args.vectordb_name))[1]
        self.sim_encoder = sim_encoder
        self.title = ''
        self.__reset_agent()
        self.__reset_temporal_mem()

    def run(self, reset_exp, reset_agent=True) -> None:
        if reset_agent:
            self.__reset_agent()
        if reset_exp:
            self.__reset_temporal_mem()

        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        no = str(self.step_n)

        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        raw_thought = self.prompt_agent().split('Action ' + no + ':')[0]
        self.scratchpad += ' ' + raw_thought
        self.logger.info(self.scratchpad.split('\n')[-1])
        thought = raw_thought.strip('Thought ' + no + ':')

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        raw_action = self.prompt_agent().split(']')[0] + ']'
        self.scratchpad += ' ' + raw_action
        try:
            action = raw_action.split('[')
            action_type, argument = action[0], action[1].strip(']')
        except:
            action_type, argument = 'Act', raw_action
        self.logger.info(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action_type == 'Finish':
            self.answer = argument

            signal = self.gpt_correct()

            if signal:
                self.scratchpad += ' Answer is CORRECT.'
            else:
                self.scratchpad += ' Answer is INCORRECT.'

            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Act':
            try:
                docs = self.retriever.get_relevant_documents(argument)

                # self.title = doc_details['metadata']['title']
                chain = load_qa_chain(self.llm, chain_type="stuff")
                # extract = '【' + self.title + '】' + '^' + str(len(doc_details['page_content'].split())) + '^' + text[
                #                                                                                                :120]
                # self.logger.info(extract)
                concat = ''
                for i in range(3):
                    doc_details = docs[i].to_json()['kwargs']
                    concat += str(doc_details)

                if len(self.enc.encode(concat)) > 3000:
                    docs[0].page_content = concat[:3000]


                result = chain.run(input_documents=[docs[0]], question=self.hfeedback + ' ' + self.question)

                obs = format_step(result)
                if self.args.method == 'RAM':
                    if obs in self.observe:
                        feedback = self.get_feedback()
                        self.hfeedback += feedback
                        self.scratchpad += format_step(feedback)
                        self.answer = feedback
                    else:
                        self.observe.append(obs)
                        self.scratchpad += format_step(result)
                        self.answer = result
                elif self.args.method == 'RAM_R3':
                    if obs in self.observe:
                        self.scratchpad += format_step(result)
                        self.answer = result
                    else:
                        self.observe.append(obs)
                        self.scratchpad += format_step(result)
                        self.answer = result

                signal = self.gpt_correct()

                if signal:
                    self.scratchpad += 'Answer is CORRECT'
                    self.finished = True
                    self.step_n += 1

            except Exception as e:
                self.logger.info(e)
                self.scratchpad += f'Could not find that page, please try again.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Search[<topic>] and Finish[<answer>].'

        self.logger.info(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        return format_step(self.llm(self._build_agent_prompt()))

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            question=self.question,
            scratchpad=self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return embsim_match(self.answer, self.key, self.sim_encoder)

    def gpt_correct(self):
        Q, A, P = self.question, self.key, self.answer
        args = "Given one question, there is a ground truth and a predicted answer. Please decide whether they are the same or not in semantic.\
        Please only output True or False. Question: " + Q + ' ' + "ground truth = " + A + ' ' + "predicted answer = " + P
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
        rsp = response.choices[0].message.content
        if rsp == 'True':
            return True
        elif rsp == 'False':
            return False
        elif rsp == 'None':
            return None

        return rsp

    def is_halted(self) -> bool:  ## exceed max try/ exceed context wind + not finish
        return ((self.step_n > self.max_steps) or (
                len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def __reset_temporal_mem(self) -> None:
        self.observe: list = []
        self.hfeedback: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

    def get_feedback(self) -> str:
        fdo = format_step(self.interact_llm(self._build_feedback_prompt()))
        self.logger.info('***********original feedback:' + fdo)
        fd = exempt_label(fdo, self.key, self.sim_encoder)
        self.logger.info('***********feedback:' + fd)
        self.logger.info('***********ground truth:' + self.key + '\n')

        return fd

    def _build_feedback_prompt(self) -> str:
        return self.feedback_prompt.format(
            examples=self.feedback_examples,
            question=self.question,
            scratchpad=self.scratchpad,
            groundtruth=self.key)


class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 logger: logging.Logger,
                 args: argparse.Namespace,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 feedback_prompt: PromptTemplate = feedback_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 interact_llm: AnyOpenAILLM = AnyOpenAILLM(
                     temperature=0,
                     max_tokens=100,
                     model_name="gpt-4",
                     model_kwargs={"stop": "\n"},
                     openai_api_key=os.environ['OPENAI_API_KEY']),
                 sim_encoder=get_similarity_encoder()
                 ) -> None:

        super().__init__(logger, args, question, key, max_steps, agent_prompt, feedback_prompt, interact_llm,
                         sim_encoder)
        self.reflect_llm = get_model(df_new_token=100, model=args.model_path)
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ''

    def run(self, reset_exp, reset_agent=True,
            reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:

        if (self.is_finished() or self.is_halted()) and ((self.gpt_correct() is False) or (self.gpt_correct() is None)):
            self.reflect(reflect_strategy)

        ReactAgent.run(self, reset_exp, reset_agent)

        return self.observe, self.hfeedback, self.answer, self.title, self.step_n

    def reflect(self, strategy: ReflexionStrategy) -> None:
        self.logger.info('-----------Reflecting-----------')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        self.logger.info(self.reflections_str)

    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt()))

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.question,
            scratchpad=truncate_scratchpad(self.scratchpad, tokenizer=self.enc))

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad)


### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n\n', ' ').replace('\n', ' ').replace('\'', '')


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])


def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip(
        '\n').strip() + '\n(END PREVIOUS TRIAL)\n'


def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)


def zcore_normalization(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = [(value - mean) / (std_dev + 0.0000000000001) for value in data]
    return normalized_data


def embsim_match(answer, key, encoder, embsim_rate=0.9):
    embeddings = encoder.encode([answer, key])
    similarity = cosine_similarity(embeddings)[0][1]

    if similarity > embsim_rate:
        return True, similarity
    else:
        return False, similarity


def bm25sim_match(corpus, query):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    similarity = bm25.get_scores(tokenized_query).tolist()
    res = zcore_normalization(similarity)
    return res


def ldasim_match(corpus, query, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    res = []
    for sentence in corpus:
        X = vectorizer.fit_transform([sentence, query])
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        doc_topic_distributions = lda_model.fit_transform(X)
        similarity_score = cosine_similarity(doc_topic_distributions)[0, 1]
        res.append(similarity_score)

    return res


def bertsim_match(corpus, query):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    res = []
    max_len = min(max([len(tokenizer.encode(sent)) for sent in corpus + [query]]), 512)
    for sentence in corpus:
        encoded = tokenizer(sentence, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
        inputs = {key: torch.tensor(val) for key, val in encoded.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        probability = torch.softmax(logits, dim=1)[0][0].item()
        res.append(probability)

    return res


def exempt_label(answer, key, encoder):
    cand = answer.replace(',', '.').split('.')
    new_cad = [s for s in cand if not embsim_match(s, key, encoder, 0.85)[0]]
    return ','.join(new_cad)
