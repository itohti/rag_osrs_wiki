import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ollama

model = SentenceTransformer('all-MiniLM-L6-v2')

# Keywords in a task description that tell us which spellbook the task needs.
SPELLBOOK_KEYWORDS = {
    "vengeance": "lunar",
    "barrage": "ancient",
    "thrall": "arceuus",
    "arceuus": "arceuus",
    "demonbane spells": "arceuus"
}

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def add_spellbook(data):
    """Add a 'spellbook' column based on keywords in the description (NaN if none match)."""
    description = data["description"].str.lower()
    for keyword, spellbook in SPELLBOOK_KEYWORDS.items():
        data.loc[description.str.contains(keyword), "spellbook"] = spellbook
    return data

def search(query, data, embeddings, top_k=5):
    query_embedding = model.encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    # print(f"Top {top_k} results for query '{query}':")
    # for idx in top_k_indices:
    #     idx_num = idx.item()
    #     # print(f"Index: {idx}, task_id: {data.iloc[idx_num]['task_id']}, Task Title: {data.iloc[idx_num]['title']}, boss_name: {data.iloc[idx_num]['boss_name']} Similarity: {similarities[idx].item()}")
    return built_context(data.iloc[top_k_indices][["task_id", "title", "boss_name", "description", "points", "completion_percent", "type", "spellbook"]])

def built_context(rows):
    context = ""
    for _, row in rows.iterrows():
        context += f"Task ID: {row['task_id']}\nTitle: {row['title']}\nBoss Name: {row['boss_name']}\nDescription: {row['description']}\nPoints: {row['points']}\nCompletion Percent: {row['completion_percent']}\nType: {row['type']}\nSpellbook: {row.get('spellbook', 'N/A')}\n\n"
    return context

def find_ranks(query, data, embeddings, task_ids):
    query_embedding = model.encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    ranks = similarities.argsort(descending=True).argsort()
    for idx in ranks:
        idx_num = idx.item()
        if data.iloc[idx_num]['task_id'] in task_ids:
            print(f"Task ID: {data.iloc[idx_num]['task_id']} is ranked {ranks[idx].item()} for query '{query}'.")

def evaluate_ranking(query, data, embeddings, eval_type, task_ids):
    context = search(query, data, embeddings, top_k=20)
    response = ollama.generate(model="llama3.1:8b", options={"temperature": 0.0}, prompt=f"<instructions>Answer from only the provided tasks. If the answer is not contained within the context, sat \'I don't know\'. Cite the task_ids for every claim. Ignore task that are not relevant.At the end of your response return the list of task ids in this format: TASK_IDS: []. If there are no relevant tasks, return an empty list.</instructions>\n\n <context>{context}</context>.\n\n This question is this <question>{query}</question>.")
    response_text = response["response"]
    task_ids_start = response_text.find("TASK_IDS: [")
    task_ids_end = response_text.find("]", task_ids_start)
    extracted_task_ids = response_text[task_ids_start:task_ids_end+1]

    extracted_task_ids_list = extracted_task_ids.replace("TASK_IDS: [", "").replace("]", "").split(",")
    extracted_task_ids_list = [int(task_id.strip()) for task_id in extracted_task_ids_list if task_id.strip().isdigit()]
    if eval_type == "exact":
        if set(extracted_task_ids_list).issubset(set(task_ids)) and set(task_ids).issubset(set(extracted_task_ids_list)):
            print(f"Query '{query}' passed the exact match evaluation.")
            return True
        else:
            print(f"Query '{query}' failed the exact match evaluation. Expected: {task_ids}, Got: {extracted_task_ids_list}")
            return False
    elif eval_type == "precision_only":
        correct_task_ids = set(extracted_task_ids_list).intersection(set(task_ids))
        precision = len(correct_task_ids) / len(extracted_task_ids_list) if extracted_task_ids_list else 0
        if (precision > 0.7):
            print(f"Query '{query}' passed the precision evaluation with precision: {precision:.2f}. Correct task IDs: {correct_task_ids}")
            return True
        else:
            print(f"Query '{query}' failed the precision evaluation with precision: {precision:.2f}. Correct task IDs: {correct_task_ids}")
            return False
    elif eval_type in ["refuse", "aggregate"]:
        if not extracted_task_ids_list:
            print(f"Query '{query}' correctly refused to answer.")
            return True
        else:
            print(f"Query '{query}' incorrectly provided task IDs: {extracted_task_ids_list}")
            return False

def main():
    file_path = "tasks_202610061311.csv"
    data = load_data(file_path)

    if data is None:
        print("Failed to load data.")
        return

    else:
        print("Data loaded")
        data.loc[data["boss_name"].notnull(), "embed_text"] = "boss: " + data["boss_name"] + "\n" + "task: " + data["title"] + "\n" + "description: "  + data["description"]
        data.loc[data["boss_name"].isnull(), "embed_text"] = "task: " + data["title"] + "\n" + "description: "  + data["description"]
        data = add_spellbook(data)
        has_spellbook = data["spellbook"].notnull() # metadata enrichment.
        data.loc[has_spellbook, "embed_text"] = data["embed_text"] + "\n" + "spellbook: " + data["spellbook"]

        embeddings = model.encode(data["embed_text"].tolist())

        print(embeddings.shape)

        file_path = "eval_questions.csv"
        queries = load_data(file_path)

        if queries is None:
            print("Failed to load queries.")
            return

        else:
            print("Queries loaded")
            queries["expected_task_ids"].fillna("", inplace=True)
            passed_tests = 0
            for _, row in queries.iterrows():
                query = row["question"]
                eval_type = row["eval_type"]
                task_ids = row["expected_task_ids"].split(";") 
                task_ids = [int(task_id.strip()) for task_id in task_ids if task_id.strip().isdigit()]
                print(f"\nEvaluating query: {query}")
                if evaluate_ranking(query, data, embeddings, eval_type, task_ids):
                    passed_tests += 1

            print(f"\nTotal passed tests: {passed_tests} out of {len(queries)}")



if __name__ == "__main__":
    main()