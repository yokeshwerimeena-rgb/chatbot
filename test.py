import joblib
model = joblib.load('chatbot_model.pkl')
print("Chatbot Ready!")
print("Type 'exit' to stop.\n")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = model.predict([user_input])
    print("Chatbot:", response[0])