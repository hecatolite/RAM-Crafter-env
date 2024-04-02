from langchain.prompts import PromptTemplate

##Performing steps based solely on the previous scratchpad (thought/action/observation)
REACT_INSTRUCTION = """You’re a player trying to play the game of Crafter. Solve the decision-making task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be two types: 
(1) Act [action to take and its serial number], which given the player’s current observation, you need to choose the next executable action to finish the task. Output the answer in this format: 'The next action: xx.'
(2) Finish [completed task name], which you have finished the task.

Here are the list of all the executable actions to take and its prerequisite:
1. Move West: Flat ground west of the agent.
2. Move East: Flat ground east of the agent.
3. Move North: Flat ground north of the agent.
4. Move South: Flat ground south of the agent.
5. Do: Facing creature or material; have necessary tool.
6. Sleep: Energy level is below maximum.
7. Place Stone: Stone in inventory.
8. Place Table: Wood in inventory.
9. Place Furnace: Stone in inventory.
10. Place Plant: Sapling in inventory.
11. Make Wood Pickaxe: Nearby table; wood in inventory.
12. Make Stone Pickaxe: Nearby table; wood, stone in inventory.
13. Make Iron Pickaxe: Nearby table, furnace; wood, coal, iron an inventory.
14. Make Wood Sword: Nearby table; wood in inventory.
15. Make Stone Sword: Nearby table; wood, stone in inventory.
16. Make Iron Sword: Nearby table, furnace; wood, coal, iron in inventory.
17. Noop: Always applicable.

Here are some examples:
{examples}
(END OF EXAMPLES)

Task: {task}
The player’s in game observation and previous experience for reference: {get_observation}"""

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
    input_variables=["examples", "task", "get_observation"],
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

FEEDBACK_INSTRUCTION = """There are two roles (Student and Teacher) in the decision-making task below. \
The Student is unsuccessful in solving the task below because it has limited relevant context.\
You are the Teacher who is an expert in the game of Crafter and can provide additional instructions about how to complete the task in one or two sentence as feedback for the Student. \
You will be given reasoning steps of Student in previous trials. \
Here are some examples:
{examples}

Task: {task}

Student:
{scratchpad}

Teacher:
Feedback:"""

feedback_agent_prompt = PromptTemplate(
    input_variables=["examples", "task", "scratchpad"],
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
