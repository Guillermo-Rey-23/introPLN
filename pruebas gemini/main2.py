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
    client = genai.Client(api_key="AIzaSyA9WI9qnU9-_q6VOyE7q4dL4QnI_Kxg3vE")
    for index, dev_headline in enumerate(dev_headlines):
        sample = f"""Here are two examples:
            Sample 1.
            Headline: 
            "{train_headlines[3]}"
            Classification: 
            "{convertir_a_clickbait_o_none(train_clickbait[3])}"

            Sample 2.
            Headline:
            "{train_headlines[7]}"
            Classification: 
            "{convertir_a_clickbait_o_none(train_clickbait[7])}"
            """
        prompt = f"""You are a text classification assistant. Your task is to determine whether a news headline uses clickbait or not.

        A "clickbait" headline is one that:
        - Intentionally exaggerates, withholds, or teases information to provoke curiosity or emotional response.
        - Often uses sensational language, cliffhangers, or vague references ("You won’t believe what happened next").

        A "non-clickbait" headline is one that:
        - Clearly states the main information or event.
        - Avoids exaggeration or withholding details.

        {sample}

        Classify the following headline strictly as one of these options:
        - "CLICKBAIT"
        - "NON-CLICKBAIT"

        Headline:
        "{dev_headline}"

        Answer with only one word: CLICKBAIT or NON-CLICKBAIT.
        """

        response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
        )
        response_text = response.candidates[0].content.parts[0].text
        responses.append(([3, 7], index, response_text))
        if verbose:
            print(f"sample-indexes: {[3, 7]}\nheadline-index: {index}\nheadline: {dev_headline}\nclassification: {dev_clickbait[index]}\ngemini classification: {response_text}")
            print()
        time.sleep(4.0) # sleep 60 / 15 secs
    return responses

print(train_headlines[7])
print(train_headlines[3])

responses = getLLMResponsesGemini(use_fixed=True, verbose=True)
file_name = "output.json"
try:
    # 1. Open the file in write mode ('w')
    with open(file_name, 'w') as outfile:
        # 2. Use json.dump() to write the data to the file
        json.dump(responses, outfile, indent=4)
    
    print(f"Successfully saved data to {file_name}")

except IOError as e:
    print(f"Error writing to file: {e}")