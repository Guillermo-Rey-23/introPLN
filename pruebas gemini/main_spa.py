# La intencion es dejar aca el codigo "oficial"
# y usar el jupyter solo para pruebas, asi es mas facil el merge.

import csv
import time
import json
from google import genai


with open("TA1C_dataset_detection_train.csv","r", encoding="UTF-8") as f:
    train = [x for x in csv.reader(f)]
with open("TA1C_dataset_detection_dev_gold.csv","r", encoding="UTF-8") as f:
    dev = [x for x in csv.reader(f)]

train_headlines = [x[4] for x in train[1:]]
train_clickbait = [x[5] for x in train[1:]]
dev_headlines = [x[4] for x in dev[1:]]
dev_clickbait = [x[6] for x in dev[1:]]

def convertir_a_clickbait_o_none(clck):
  return "CLICKBAIT" if clck == "Clickbait" else "NON-CLICKBAIT"

def get_sample(dev_headline): # reemplazar
    return (7, train_headlines[7], train_clickbait[7])

def getLLMResponsesGemini(use_fixed = True, verbose = False):
    responses = []
    client = genai.Client(api_key="AIzaSyBGHG95z8hnWo-YrUtJdMxja69rW_npM6s")
    for index, dev_headline in enumerate(dev_headlines):
        sample_index = 3
        sample_headline = train_headlines[sample_index]
        sample_classification = train_clickbait[sample_index]
        if not use_fixed:
            sample_index, sample_headline, sample_classification = get_sample(dev_headline)
        sample = f"""Aquí hay un ejemplo: 
            "{sample_headline}"
            Clasificación: 
            "{convertir_a_clickbait_o_none(sample_classification)}"
            """
        
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
        "{dev_headline}"

        Responde con solo una palabra: CLICKBAIT or NON-CLICKBAIT.
        """

        response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
        )
        response_text = response.candidates[0].content.parts[0].text
        responses.append(([sample_index], index, response_text))
        if verbose:
            print(f"sample-indexes: {[sample_index]}\nheadline-index: {index}\nheadline: {dev_headline}\nclassification: {dev_clickbait[index]}\ngemini classification: {response_text}")
            print()
        time.sleep(4.0) # sleep 60 / 15 secs
    return responses

print(train_headlines[7])
print(train_headlines[3])

responses = getLLMResponsesGemini(use_fixed=True, verbose=True)
file_name = "output1.json"
try:
    # 1. Open the file in write mode ('w')
    with open(file_name, 'w') as outfile:
        # 2. Use json.dump() to write the data to the file
        json.dump(responses, outfile, indent=4)
    
    print(f"Successfully saved data to {file_name}")

except IOError as e:
    print(f"Error writing to file: {e}")