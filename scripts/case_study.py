import os
from datasets import load_dataset

data_path = '../pls/qa.en.c.json'

# load the dataset
test_set = load_dataset('json', data_files=data_path, split='train[90%:]')

model_name_to_path = {
    "t5": "tmp/seq2seq_t5_c",
    "codet5": "tmp/seq2seq_codet5_c",
    "plbart": "tmp/seq2seq_plbart_c",
}

model_name_to_predictions = {}

for model_name, model_path in model_name_to_path.items():
    model_predictions = [line.strip() for line in open(os.path.join(model_path, "generated_predictions.txt"))]
    model_name_to_predictions[model_name] = model_predictions

print("len(test_set): ", len(test_set))
print("len(model_name_to_predictions['t5']): ", len(model_name_to_predictions["t5"]))
print("len(model_name_to_predictions['codet5']): ", len(model_name_to_predictions["codet5"]))
print("len(model_name_to_predictions['plbart']): ", len(model_name_to_predictions["plbart"]))

assert len(test_set) == len(model_name_to_predictions["t5"]) == len(model_name_to_predictions["plbart"])


n = len(test_set)
i = n-1
while i >= 0:
    input()
    print("title: ", test_set[i]["title"])
    print("question: ", test_set[i]["question"])
    print("answer: ", test_set[i]["answer"])
    print("t5: ", model_name_to_predictions["t5"][i])
    # print("codet5: ", model_name_to_predictions["codet5"][i])
    print()
    print("plbart: ", model_name_to_predictions["plbart"][i])
    print()
    i -= 1

