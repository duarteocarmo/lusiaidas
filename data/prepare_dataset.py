from datasets import DatasetDict, Dataset

PRETEXT = "Escreve uma estrofe ao estilo de Os Lusíadas"
index = 45
data_file = "lusiadas.txt"
dataset_name = "duarteocarmo/lusiadas-completion"

with open(data_file, "r", encoding="utf-8") as f:
    file_data = f.read()

paragraphs = file_data.split("\n\n")
paragraphs = [p.strip() for p in paragraphs if len(p.split("\n")) >= 3]
paragraphs = ["\n".join(p.split("\n")[1:]) for p in paragraphs]
paragraphs = [{"instruction": PRETEXT, "output": p, "input": ""} for p in paragraphs]


num_examples = len(paragraphs)

train_text_list = paragraphs[: int(num_examples * 1)]
# val_text_list = paragraphs[int(num_examples * 0.9) :]

print("Total sentences: ", len(paragraphs))
print("Num train samples: ", len(train_text_list))
# print("Num val samples: ", len(val_text_list))

data = DatasetDict()
data["train"] = Dataset.from_list(train_text_list)
# data["test"] = Dataset.from_dict({"text": val_text_list})

print(f"Example:\n{data['train'][index]}")

data.push_to_hub(dataset_name, revision="main", private=False)
