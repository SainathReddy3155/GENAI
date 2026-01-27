import boto3
import json
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

bedrock_client= boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)
MODEL_ID='amazon.nova-pro-v1:0'
st.title("Chatbot using AWS Bedrock")
def nova_payload_function(user_input):
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "You are helpful AI Assistant who answers the user asked questions professionaly"
                            f"User question: {user_input}"
                        )
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 200,
            "temperature": 1.0,
            "topP": 1.0
        }
    }
    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )  
    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]
user_input = st.text_input("Enter your query:")

if user_input:
    with st.spinner("Thinking..."):
        response = nova_payload_function(user_input)
        st.write("", response)

# if __name__=="__main__":
#     while True:
#         user_input=input("User : ")
#         if user_input.lower() in ["bye","quit","q"]:
#             break
#         else:
#             res=nova_payload_function(user_input)
#             print("AI : ",res)
            
