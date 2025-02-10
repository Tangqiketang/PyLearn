from flask import Flask, request, jsonify

app = Flask(__name__)

# 这里我们创建一个路由，用户可以通过这个路由与助手对话
@app.route("/chat", methods=["GET"])
def chat():
   # user_input = request.json.get("message")
   # return jsonify({"response": "hhhhhhhhhh"})
   return "aaaaa"

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api', methods=['POST'])
def api():
    data = request.json
    return jsonify({"message": "Received data", "data": data})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)