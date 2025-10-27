# La intencion es dejar aca el codigo "oficial"
# y usar el jupyter solo para pruebas, asi es mas facil el merge.

import csv
import json
from sklearn.metrics import f1_score, classification_report

with open("TA1C_dataset_detection_train.csv","r", encoding="UTF-8") as f:
    train = [x for x in csv.reader(f)]
with open("TA1C_dataset_detection_dev_gold.csv","r", encoding="UTF-8") as f:
    dev = [x for x in csv.reader(f)]

train_headlines = [x[4] for x in train[1:]]
train_clickbait = [x[5] for x in train[1:]]
dev_headlines = [x[4] for x in dev[1:]]
dev_clickbait = [x[6] for x in dev[1:]]

pruebas_path = r"./pruebas gemini"
pruebas = [pruebas_path + f"/prueba_{i}.json" for i in range(1,5)]

pruebas_parsed = []
for i in pruebas:
    with open(i, 'r', encoding='utf-8') as archivo:
        result = json.load(archivo)
        pruebas_parsed.append(result)
    print(f"Loaded {i}")

def convertir_a_clickbait_o_none(clck):
  return "CLICKBAIT" if clck == "Clickbait" else "NON-CLICKBAIT"

print()
print("Results evaluations:")

true_results = [convertir_a_clickbait_o_none(x) for x in dev_clickbait]
for i, prueba in enumerate(pruebas_parsed):
    results = [x[2] for x in prueba]
    print(f"Prueba {i}, ({pruebas[i]})")
    print(f"F1-Score macro: {str(round(f1_score(true_results, results, average='macro')*100, 2))}\n")
    print(classification_report(true_results, results))
    print()