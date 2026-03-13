import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from time import sleep

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('myapp')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

@app.route('/message', methods=['POST'])
def receive_message():
    data = request.get_json()
    logger.info("Received message: %s", data)
    return main(
        serialize_response(data['messages']),
        data['model_name']
    )

def serialize_response(response : list[dict]):
    messages = []
    for msg in response:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']['text']))
        elif msg['role'] == 'assistant':
            messages.append(AIMessage(content=msg['content']['text']))
        elif msg['role'] == 'system':
            messages.append(SystemMessage(content=msg['content']['text']))
    return messages


def main(messages, model_name:str):
    model = ChatOpenAI(
        model=model_name,
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # model_with_tools = model.bind_tools([calculator, search, get_weather])

    # response = model_with_tools.invoke(query)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)