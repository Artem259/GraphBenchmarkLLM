import networkx as nx
import random
import subprocess
import json
import re
import time
import uuid
import sys
from typing import Tuple

# --- CONFIG ---
NUM_GRAPHS = int(sys.argv[1])
SESSION_ID = str(uuid.uuid4())
MODELS = ["deepseek-r1:8b", "llama3.1:8b", "dolphin3:8b", "mistral:7b", "gemma:7b", "qwen2.5:7b"]
TASKS = ["cycle_detection", "subgraph_isomorphism", "planarity", "graph_coloring", "traversal_count"]
NUM_EXAMPLES = 4

assert NUM_GRAPHS % 2 == 0
assert NUM_EXAMPLES % 2 == 0

# --- UTILS ---
def run_ollama(model: str, prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600
        )
        return result.stdout.decode()
    except subprocess.TimeoutExpired:
        return "Timeout"

def extract_output(text: str, answer) -> str:
    if isinstance(answer, bool):
        match = re.findall(r'(True|False|true|false|Yes|yes|No|no)', text)
        if not match:
            return "Invalid"
        output = match[-1][0].upper()
        output = "T" if output == "Y" else output
        output = "F" if output == "N" else output
        return output

    match = re.findall(r'(\d+)', text)
    return match[-1] if match else "Invalid"

def generate_graph() -> nx.Graph:
    p = random.uniform(0.2, 0.8)
    G = nx.gnp_random_graph(random.randint(5, 10), p=p, directed=False)
    return G

# --- TASK FUNCTIONS ---
def task_shortest_path(G: nx.Graph) -> Tuple[str, int]:
    nodes = list(G.nodes)
    if len(nodes) < 2:
        return "", -1
    u, v = random.sample(nodes, 2)
    try:
        length = nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        length = -1
    prompt = (f"Given the graph edges: {list(G.edges)}. What is the length of the shortest path between {u} and {v}? "
              f"Provide your final answer clearly at the end of your response in the following format: 'ANSWER: [number]'")
    return prompt, length

def task_cycle_detection(G: nx.Graph) -> Tuple[str, bool]:
    try:
        nx.find_cycle(G)
        has_cycle = True
    except nx.NetworkXNoCycle:
        has_cycle = False
    prompt = (f"Does the graph with edges {list(G.edges)} contain a cycle? Answer True or False. "
              f"Provide your final answer clearly at the end of your response in the following format: 'ANSWER: True' or 'ANSWER: False'")
    return prompt, has_cycle

def task_connectivity(G: nx.Graph) -> Tuple[str, int]:
    count = nx.number_connected_components(G)
    prompt = (f"How many connected components are in the graph with edges {list(G.edges)}? "
              f"Provide your final answer clearly at the end of your response in the following format: 'ANSWER: [number]'")
    return prompt, count

def task_traversal_count(G: nx.Graph) -> Tuple[str, int]:
    start = random.choice(list(G.nodes))
    visited = list(nx.dfs_preorder_nodes(G, start))
    prompt = (f"Starting from node {start}, how many nodes will a DFS visit in the graph with edges {list(G.edges)}? "
              f"Provide your final answer clearly at the end of your response in the following format: 'ANSWER: [number]'")
    return prompt, len(visited)

def task_graph_coloring(G: nx.Graph) -> Tuple[str, int]:
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    num_colors = max(coloring.values()) + 1
    prompt = (f"What is the minimum number of colors needed to color the graph with edges {list(G.edges)} so no adjacent nodes share a color? "
              f"Provide your final answer clearly at the end of your response in the following format: 'ANSWER: [number]'")
    return prompt, num_colors

def task_subgraph_isomorphism(G: nx.Graph) -> Tuple[str, bool]:
    pattern = nx.cycle_graph(3)
    matcher = nx.algorithms.isomorphism.GraphMatcher(G, pattern)
    found = matcher.subgraph_is_isomorphic()
    prompt = (f"Does the graph with edges {list(G.edges)} contain a triangle (3-node cycle)? Answer True or False. "
              f"Provide your final answer clearly at the end of your response in the following format: 'ANSWER: True' or 'ANSWER: False'")
    return prompt, found

def task_planarity(G: nx.Graph) -> Tuple[str, bool]:
    planar, _ = nx.check_planarity(G)
    prompt = (f"Is the graph with edges {list(G.edges)} planar? Answer True or False. "
              f"Provide your final answer clearly at the end of your response in the following format: 'ANSWER: True' or 'ANSWER: False'")
    return prompt, planar

TASK_FUNCS = {
    "shortest_path": task_shortest_path,
    "cycle_detection": task_cycle_detection,
    "connectivity": task_connectivity,
    "traversal_count": task_traversal_count,
    "graph_coloring": task_graph_coloring,
    "subgraph_isomorphism": task_subgraph_isomorphism,
    "planarity": task_planarity
}

# --- FEW-SHOT PROMPTING ---
def generate_boolean_examples(task_func, true_needed, false_needed):
    examples = []
    true_count, false_count = 0, 0
    while true_count < true_needed or false_count < false_needed:
        G = generate_graph()
        prompt, ans = task_func(G)
        if ans is True and true_count < true_needed:
            examples.append((prompt, ans))
            true_count += 1
        elif ans is False and false_count < false_needed:
            examples.append((prompt, ans))
            false_count += 1
    return examples

def generate_numeric_examples(task_func, count):
    examples = []
    values = set()
    while len(examples) < count:
        G = generate_graph()
        prompt, ans = task_func(G)
        if ans not in values and ans != -1:
            examples.append((prompt, ans))
            values.add(ans)
    return examples

# --- MAIN LOOP ---
results = []
task_graphs = {}

for task in TASKS:
    true_count = 0
    false_count = 0
    i = 0
    while i < NUM_GRAPHS:
        G = generate_graph()
        prompt_func = TASK_FUNCS[task]
        zero_shot_prompt, answer = prompt_func(G)

        if isinstance(answer, bool):
            counts = [answer for (t, _), (_, _, answer) in task_graphs.items() if t == task]
            if answer is True and counts.count(True) >= NUM_GRAPHS // 2:
                continue
            if answer is False and counts.count(False) >= NUM_GRAPHS // 2:
                continue
        else:
            if answer == -1:
                continue

        if isinstance(answer, bool):
            examples = generate_boolean_examples(prompt_func, NUM_EXAMPLES // 2, NUM_EXAMPLES // 2)
        else:
            examples = generate_numeric_examples(prompt_func, NUM_EXAMPLES)

        examples_text = "\n\n".join(
            f"Q: {ex_prompt}\nA: {ex_answer}" for ex_prompt, ex_answer in examples
        )
        few_shot_prompt = f"{examples_text}\n\nNow, answer the following:\nQ: {zero_shot_prompt}"

        task_graphs[(task, i)] = (zero_shot_prompt, few_shot_prompt, answer)
        i += 1

# --- BENCHMARK LOOP ---
for model in MODELS:
    for (task, i), (zero_shot_prompt, few_shot_prompt, answer) in task_graphs.items():
        for prompt, num_examples in ((zero_shot_prompt, 0), (few_shot_prompt, NUM_EXAMPLES)):
            start_time = time.perf_counter()
            response = run_ollama(model, prompt)
            elapsed = time.perf_counter() - start_time

            guess = extract_output(response, answer)
            if isinstance(answer, bool):
                correct = str(answer)[0]
            else:
                correct = str(answer)
            is_correct = guess == correct
            results.append({
                "session_id": SESSION_ID,
                "task": task,
                "model": model,
                "prompt": prompt,
                "num_examples": num_examples,
                "ground_truth": correct,
                "model_guess": guess,
                "correct": is_correct,
                "time_sec": round(elapsed, 3)
            })
            print(f"[{model}, {task}, {i+1}/{NUM_GRAPHS}, {num_examples}]: "
                  f"{guess}, {correct}, {str(is_correct)[0]}, {elapsed:.2f}s")

# --- SAVE RESULTS ---
try:
    with open("result.json", "r") as f:
        previous_results = json.load(f)
except FileNotFoundError:
    previous_results = []

with open("result.json", "w") as f:
    json.dump(previous_results + results, f, indent=2)

print(f"Benchmarking complete. Session ID: {SESSION_ID}")
