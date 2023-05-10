import re
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

try:
    model_path = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generator = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        temperature=0.7
)

except:
    print("Failed to access gpt2")
    exit()


#inicio de la conversacion usando el modelo GPT-2
print("Hello :)")
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ['bye', 'exit']:
            print("Bye")
            break
        else:
            response = generator(user_input, max_length=50, do_sample=True)[0]['generated_text']

            #impresión de la respuesta
            if response:
                print(" ")
                print("Chatbot: ", response)
                print(" ")
            else:
                print("Chatbot can't answer")
    except KeyboardInterrupt:
        print("Chatbot closed")
        break
