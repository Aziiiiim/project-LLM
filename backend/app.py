import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from time import sleep

from langchain_openai import ChatOpenAI

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
    sleep(2)  # Simulate processing time
    return jsonify("This is a response from the backend!")


def main(query, model_name:str):
    model = ChatOpenAI(
        model=model_name,
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # model_with_tools = model.bind_tools([calculator, search, get_weather])

    # response = model_with_tools.invoke(query)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)