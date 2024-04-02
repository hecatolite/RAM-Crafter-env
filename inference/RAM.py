import sys

sys.path.append('...')
sys.path.append('..')
root = '../root/'
import warnings

warnings.filterwarnings("ignore")
import os
import csv
import json
from time import time
import openai
import argparse
from util import summarize_react_trial
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy, Memory_update
import logging

openai.api_key = 'sk-bIL5v0zxFsMH9CP02a37C2E5F59e4061864a5eBb234e979c'
openai.api_base = 'https://pro.aiskt.com/v1'

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None, help="the dataset you train")
    parser.add_argument('--result_path', type=str, default=None, help="result save path")
    parser.add_argument('--model_path', type=str, default=None, help="the model path")
    parser.add_argument('--vectordb_name', type=str, default=None, help="the name of the vectordb")
    parser.add_argument('--method', type=str, default=None, help="RAM or R3")

    return parser.parse_args(args)


def GPT_assis(Q, A, P):
    args = "Given one question, there is a ground truth and a predicted answer. Please decide whether they are the same or not in semantic.\
     Please only output True or False. Question: " + Q + ' ' + "ground truth = " + A + ' ' + "predicted answer = " + P
    args = args.replace('"', '^')
    prompt = [{"role": "user", "content": args}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=prompt,
        max_tokens=100
    )

    return response.choices[0].message.content


def RAM(q, a, args, n=3):
    agents = [agent_cls(logger, args, q, a)]
    trial, steps = 0, 0
    correct, incorrect, halted = 0, 0, 0
    upd_title, upd_text = '', ''
    for i in range(n):
        if trial == 0:
            exp = True
        else:
            exp = False
        for agent in [a for a in agents if not a.is_correct()[0]]:
            if strategy != ReflexionStrategy.NONE:
                obs, fd, pred, title, step = agent.run(reset_exp=exp, reflect_strategy=strategy)
            else:
                obs, fd, pred, title, step = agent.run()

        trial += 1
        steps += step
        c, ic, h = summarize_react_trial(agents)
        correct, incorrect, halted = correct + len(c), incorrect + len(ic), halted + len(h)
        logger.info(f'-----------Finished Trial {trial}, Correct: {correct}, Incorrect: {incorrect}, Halted: {halted}')
        if len(c) > 0 and args.method == 'RAM':
            upd_title, upd_text = Memory_update(logger, args, q, a, obs, fd).generate_shortmem()
            return pred, trial, title, steps, fd, upd_title, upd_text
        elif len(c) > 0 and args.method == 'RAM_R3':
            return pred, trial, title, steps, fd

    if args.method == 'RAM':
        upd_title, upd_text = Memory_update(logger, args, q, a, obs, fd).generate_shortmem()
        return pred, trial, title, steps, fd, upd_title, upd_text

    elif args.method == 'RAM_R3':
        return pred, trial, title, steps, fd


def QA(ID, q, a, args):
    logger.info('\n' + '~~~~~~~~~~' + str(ID) + ':' + q + a)
    title = ''
    answer[ID] = {}
    answer[ID]['Q'] = q
    answer[ID]['A'] = a
    answer[ID]['P'], answer[ID]['Time'], answer[ID]['R'], answer[ID]['Eval'] = [], [], [], []
    answer[ID]['Turn'], answer[ID]['Step'], answer[ID]['Feedback'] = '', '', ''

    time_1 = time()

    prediction, turns, title, step, feedback, upd_title, upd_text = RAM(q, a, args=args, n=3)

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
    answer[ID]['Turn'] = turns
    answer[ID]['Step'] = step
    answer[ID]['R'].append(title)
    answer[ID]['Feedback'] = feedback


if __name__ == '__main__':
    args = parse_args()

    strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
    agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter()
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    if not os.path.exists('./log'):
        os.makedirs('./log')
    fhlr = logging.FileHandler('./log/' + args.dataset + '_' + args.method + '.log')
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    dataset = args.dataset
    if dataset == 'freshqa':
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                               f'{dataset}_groundtruth.txt'), 'r') as f:
            ground_truth = f.readlines()
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                               'freshqa_IDs.txt'), 'r') as f:
            white = list(eval(f.read()))
    else:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                               f'{dataset}_groundtruth.json'), 'r') as f:
            ground_truth = json.loads(f.read())

    filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                            f'{dataset}.csv')
    path = args.result_path

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        with open(path, 'r') as f:
            answer = json.loads(f.read())
    except:
        answer = {}

    random = [7, 19, 27, 41, 45, 48, 68, 77, 80, 82, 85, 91, 116, 119, 121, 138, 160, 165, 174, 182, 185, 197, 200, 210, 212, 219, 227, 233, 235, 238, 248, 251, 257, 269, 278, 291, 295, 301, 308, 312, 321, 326, 333, 339, 348, 350, 356, 357, 362]

    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:

            if dataset == 'freshqa':
                ID, q = int(row[0]), row[2]
                a = ground_truth[ID].strip('\n')
                if ID >= 68 and ID in white and ID in random:
                    QA(ID, q, a, args=args)
                    with open(path, 'w') as g:
                        g.write(json.dumps(answer))
            elif dataset == 'mquake':
                ID, q = int(row[0]), row[1]
                a = ground_truth[str(ID)].strip('\n')
                if ID >= 0:
                    QA(ID, q, a, args=args)
                    with open(path, 'w') as g:
                        g.write(json.dumps(answer))
