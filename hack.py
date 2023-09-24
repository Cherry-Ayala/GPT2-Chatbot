import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re

model_name = "gpt2"

try:
    model_path = model_name
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

except Exception as e:
    print("Failed to access gpt2")
    exit()

# Define investment portfolios
portfolios = [
    {'risk': 1, 'expectedReturn': 0.03, 'minInvestment': 1000},
    {'risk': 2, 'expectedReturn': 0.05, 'minInvestment': 2000},
    {'risk': 3, 'expectedReturn': 0.07, 'minInvestment': 3000},
    {'risk': 4, 'expectedReturn': 0.09, 'minInvestment': 4000},
    {'risk': 5, 'expectedReturn': 0.11, 'minInvestment': 5000}
]

# Function to extract information from user input
def extract_information(input_text):
    # Split the input by commas to get the values
    values = input_text.split(',')
    
    # Check if there are exactly three values (amount, risk, time)
    if len(values) == 3:
        try:
            investmentAmount = float(values[0].strip())
            riskLevel = int(values[1].strip())
            timePeriod = int(values[2].strip())
            return investmentAmount, riskLevel, timePeriod
        except ValueError:
            pass
    return None, None, None

# Create a text-generation pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Main conversation loop
print("Chupame los huevos y te ayudo a invertir")
while True:
    try:
        user_input = input("User (Enter investment amount, risk level (1-5), and years separated by commas): ")
        if user_input.lower() in ['bye', 'exit']:
            print("Bye")
            break
        else:
            investmentAmount, riskLevel, timePeriod = extract_information(user_input)

            if not investmentAmount or not (1 <= riskLevel <= 5) or not timePeriod:
                print("Invalid input. Please provide the investment amount, risk level (1-5), and years separated by commas.")
                continue

            # Generate a custom response from the GPT-2 model
            input_prompt = f"You want to invest {investmentAmount:.2f} dollars for {timePeriod} years with a risk level of {riskLevel}."
            custom_response = generator(input_prompt, max_length=50, do_sample=True)[0]['generated_text']

            portfolioRecommendations = [
                f"Risk Level {portfolio['risk']}: Expected Return after {timePeriod} years - ${portfolio['expectedReturn'] * investmentAmount:.2f}"
                for portfolio in portfolios
            ]

            print("chat:", custom_response)
            print("\nInvestment Recommendations:")
            print('\n'.join(portfolioRecommendations))

    except KeyboardInterrupt:
        print("\nChatbot closed")
        break



