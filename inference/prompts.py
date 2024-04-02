from langchain.prompts import PromptTemplate

##Performing steps based solely on the previous scratchpad (thought/action/observation)
REACT_INSTRUCTION = """Solve a question-answering task with interleaving Thought, Action, Observation steps. \
Thought can reason about the current situation, and Action can be two types: 
(1) Search[key words or phrases], which searches and returns the relevant articles or paragraphs from an external database as context.
(2) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

##Performing steps based on the previous scratchpad (thought/action/observation) and reflections
REACT_REFLECT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. You will be given a previous reasoning trial in which you were given access to an external database and a question to answer. \
(1)Thought can reason about the current situation, and Action can be two types: \
(2) Search[key words or phrases], which searches and returns the relevant articles or paragraphs from an external database as context.\
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}

{reflections}

Question: {question}{scratchpad}"""

##Reflecting based on the previous scratchpad (thought/action/observation)
REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self-reflection. \
You will be given a previous reasoning trial in which you had access to an external database and a question to answer. \
You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. \
In a few sentences, diagnose a possible reason for failure and devise a new, concise, high-level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""

react_agent_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=REACT_INSTRUCTION,
)

react_reflect_agent_prompt = PromptTemplate(
    input_variables=["examples", "reflections", "question", "scratchpad"],
    template=REACT_REFLECT_INSTRUCTION,
)

reflect_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad"],
    template=REFLECT_INSTRUCTION,
)

REFLECTION_HEADER = 'You have attempted to answer following question before and failed. \
The following reflection(s) provide a plan to avoid failing to answer the question in the same way you did previously. \
Use them to improve your strategy of correctly answering the given question.\n'

REFLECTION_AFTER_LAST_TRIAL_HEADER = '\
The following reflection(s) provide a plan to avoid failing to answer the question in the same way you did previously. \
Use them to improve your strategy of correctly answering the given question.\n'

LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. \
Below is the last trial you attempted to answer the question.\n'

FEEDBACK_INSTRUCTION = """There are two roles (Student and Teacher) in the question-answering task below. \
The Student is unsuccessful in answering the question because it has limited relevant context.\
You are the Teacher who is an expert in rich world knowledge and can provide additional facts in one or two sentence as feedback for the Student. \
You will be given reasoning steps of Student in previous trials and the Ground truth as direct answer. \
You will be punished if the feedback is semantically similar to Ground truth or contains the same knowledge as Ground truth in different expressions.\
Here are some examples:
{examples}

Question: {question}
Groundtruth: {groundtruth}

Student:
{scratchpad}

Teacher:
Feedback:"""

feedback_agent_prompt = PromptTemplate(
    input_variables=["examples", "question", "scratchpad", "groundtruth"],
    template=FEEDBACK_INSTRUCTION,
)

MEMORY_UPDATE = """Given the latest relevant fact, please update/edit the existing memory based on the fact.\
If the given the fact has nothing to do with the existing memory and there is no need to update/edit, then output 'None'.\
Here are some examples:
{examples}

Existing memory: 
{existing_memory}
Latest relevant fact: 
{acquired_fact}
Update and summarization: 
"""
# The texts are different knowledge on the same question/topic.\
# If there is a contradictory between different knowledge, the ones latter are most latest to memory.\

memupdate_agent_prompt = PromptTemplate(
    input_variables=["examples", "existing_memory", "acquired_fact"],
    template=MEMORY_UPDATE,
)

DECOMPOSE = """Please decompose the existing multi-hop memory into sub-facts if needed. \
If the existing memory cannot be decomposed further, then strictly output the original memory without any revisions.\
Here are some examples:
{examples}

Existing memory: {existing_mem}
Decomposed memory: 
"""
# The texts are different knowledge on the same question/topic.\
# If there is a contradictory between different knowledge, the ones latter are most latest to memory.\

decompose_agent_prompt = PromptTemplate(
    input_variables=["examples", "existing_mem"],
    template=DECOMPOSE,
)
