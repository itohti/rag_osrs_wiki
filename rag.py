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
    print(f"Top {top_k} results for query '{query}':")
    for idx in top_k_indices:
        idx_num = idx.item()
        print(f"Index: {idx}, task_id: {data.iloc[idx_num]['task_id']}, Task Title: {data.iloc[idx_num]['title']}, boss_name: {data.iloc[idx_num]['boss_name']} Similarity: {similarities[idx].item()}")

def find_ranks(query, data, embeddings, task_ids):
    query_embedding = model.encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    ranks = similarities.argsort(descending=True).argsort()
    for idx in ranks:
        idx_num = idx.item()
        if data.iloc[idx_num]['task_id'] in task_ids:
            print(f"Task ID: {data.iloc[idx_num]['task_id']} is ranked {ranks[idx].item()} for query '{query}'.")

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
        has_spellbook = data["spellbook"].notnull()
        data.loc[has_spellbook, "embed_text"] = data["embed_text"] + "\n" + "spellbook: " + data["spellbook"]

        embeddings = model.encode(data["embed_text"].tolist())

        print(embeddings.shape)

        find_ranks("Which tasks require the Lunar spellbook?", data, embeddings, [159, 515, 292])

        add_spellbook(data)
        print("----------------metadata enrichment--------------")
        find_ranks("Which tasks require the Lunar spellbook?", data, embeddings, [159, 515, 292])

        search("which tasks requires that I can't take any damage", data, embeddings)



if __name__ == "__main__":
    main()