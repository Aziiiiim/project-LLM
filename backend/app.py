from flask import Flask, jsonify, request
from flask_cors import CORS
from time import sleep

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

@app.route('/message', methods=['POST'])
def receive_message():
    data = request.get_json()
    print("Received message:", data)
    sleep(2)  # Simulate processing time
    return jsonify("This is a response from the backend!")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)