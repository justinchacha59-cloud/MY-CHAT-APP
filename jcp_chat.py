import os
from flask import Flask, request, jsonify, render_template
from groq import Groq
from flask_cors import CORS
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
app = Flask(__name__)
CORS(app)  # This prevents "Connection Errors" in the browser

# 2. Initialize Groq (Get key from https://groq.com)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Get data from frontend
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({"error": "Message is empty"}), 400

        # 3. Call Groq API (Llama 3.1 is very stable and free)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are JCP CHAT, a helpful AI created by Justin Chachap."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        bot_response = completion.choices[0].message.content
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "🔌❌  !Please try again later. 💭."}), 500

if __name__ == '__main__':
    # Use environment port for Render compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

