import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import logging
import sys

from code.main_agent import MultiAgent

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('myapp')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

@app.route('/message', methods=['POST'])
def receive_message():
    data = request.get_json()
    logger.info("Received message: %s", data)
    agent = MultiAgent(model_name=data['model'])
    return jsonify(agent.run(
        serialize_response(data['messages'])
    ))

def serialize_response(response : list[dict]) -> list[BaseMessage]:
    messages = []
    for msg in response:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content'][0]['text']))
        elif msg['role'] == 'assistant':
            messages.append(AIMessage(content=msg['content'][0]['text']))
        elif msg['role'] == 'system':
            messages.append(SystemMessage(content=msg['content'][0]['text']))
    return messages

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)