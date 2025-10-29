# La intencion es dejar aca el codigo "oficial"
# y usar el jupyter solo para pruebas, asi es mas facil el merge.

import csv
import time
import json
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


with open("TA1C_dataset_detection_train.csv","r", encoding="UTF-8") as f:
    train = [x for x in csv.reader(f)]
with open("TA1C_dataset_detection_dev_gold.csv","r", encoding="UTF-8") as f:
    dev = [x for x in csv.reader(f)]

train_headlines = [x[4] for x in train[1:]]
train_clickbait = [x[5] for x in train[1:]]
dev_headlines = [x[4] for x in dev[1:]]
dev_clickbait = [x[6] for x in dev[1:]]

embeddings_model = SentenceTransformer("intfloat/multilingual-e5-large") # Se carga el modelo de embeddings e5

# Primero transformo todos los titulares de dev en embeddings
train_headlines_emb = embeddings_model.encode(train_headlines, convert_to_numpy=True)

def get_knn_examples(neigh, headline_emb, train_headlines, train_clickbait):
    distances, indices = neigh.kneighbors([headline_emb])
    examples = []
    for idx in indices[0]:
        examples.append((idx, train_headlines[idx], train_clickbait[idx]))
    return examples

def convertir_a_clickbait_o_none(clck):
  return "CLICKBAIT" if clck == "Clickbait" else "NON-CLICKBAIT"

def build_sample_prompt(examples):
    sample = ""
    cant_ejemplos = len(examples)
    
    if cant_ejemplos == 1:
      sample += "Aquí hay un ejemplo: \n"
    else:
      sample += f"Aquí hay {cant_ejemplos} ejemplos: \n"

    for idx, text, label in examples:
        sample += f"Titular: \"{text}\"\nClasificación: {convertir_a_clickbait_o_none(label)}\n\n"
    return sample

# Genero los NN de train
neigh = NearestNeighbors(n_neighbors=3, metric='cosine')
neigh.fit(train_headlines_emb)

# Cliente gemini
client = genai.Client(api_key="AIzaSyD3lBvT7vApEC2eIpCWm3WbgvrMz5r_qmU")

responses = []
for idx, headline in enumerate(dev_headlines):
    if idx > 3:
        break
    # Creo embedding del headline
    headline_emb = embeddings_model.encode(headline, convert_to_numpy=True)
    # Obtenego los k vecinos
    examples = get_knn_examples(neigh, headline_emb,  train_headlines, train_clickbait)
    # Construyo el sample a partir de los examples
    sample = build_sample_prompt(examples)
    
    print(sample)

    prompt = f"""Eres un asistente de clasificación de texto. Tu tarea es determinar si el titular de una noticia usa clickbait o no.

        Un titular "clickbait" es uno que:
        - Exagera intencionalmente, retiene o se burla de la información para provocar curiosidad o una respuesta emocional.
        - A menudo usa lenguaje sensacionalista, suspenso, o referencias vagas.

        Un titular "non-clickbait" es uno que:
        - Expresa claramentela información principal o el evento.
        - Evita exagerar o retener información.

        {sample}

        Clasifica el siguiente titular estrictamente como una de estas opciones:
        - "CLICKBAIT"
        - "NON-CLICKBAIT"

        Titular:
        "{headline}"

        Responde con solo una palabra: CLICKBAIT or NON-CLICKBAIT.
        """
    
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    response_text = response.candidates[0].content.parts[0].text
    sample_list = [int(ex[0]) for ex in examples]
    responses.append((sample_list, idx, response_text))

    print(f"sample-indexes: {sample_list}\nheadline-index: {idx}\nheadline: {headline}\nclassification: {dev_clickbait[idx]}\ngemini classification: {response_text}")
    print()
    time.sleep(4.0) # sleep 60 / 15 secs

file_name = "output4.json"
try:
    # 1. Open the file in write mode ('w')
    with open(file_name, 'w') as outfile:
        # 2. Use json.dump() to write the data to the file
        json.dump(responses, outfile, indent=4)
    print(f"Successfully saved data to {file_name}")
except IOError as e:
    print(f"Error writing to file: {e}")