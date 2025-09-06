import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# --- IMPORTANT ---
# Set up your Gemini API Key. 
# For local development, create a .env file and add: GEMINI_API_KEY="your_key_here"
# The code below will automatically load it.
# For Cloud Run, you will set this as an environment variable in the deployment settings.
try:
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except ImportError:
    print("dotenv package not found. Make sure to set environment variables manually.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Handle cases where the API key is not set.
    # For now, we'll proceed and let the API call fail with a clear message.
    pass

# --- Page Routes (CORRECTED) ---

@app.route('/')
def landing():
    """Renders the landing page FIRST (your old index.html)."""
    return render_template('landing.html')

@app.route('/app')
def index():
    """Renders the main application page (your old app.html)."""
    return render_template('index.html')

# --- API Routes (No changes here) ---

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """Receives document text, calls Gemini for analysis, and returns the result."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    content = data.get('content')

    if not content:
        return jsonify({"error": "No content provided"}), 400

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Analyze the following document. Structure your response with the following headings EXACTLY as written, using "###" for each heading: ### Summary, ### Key Insights, ### Important Mentions, ### Vigilance Score (1-100) and Justification.

        For "Important Mentions" and "Key Insights", use bullet points starting with '*'.

        For "Important Mentions", extract and list all specific dates, deadlines, and financial amounts. If none are found, state "None found."

        For "Vigilance Score (1-100) and Justification", provide a numerical risk score from 1 (very low risk) to 100 (very high risk). After the score, provide a single sentence justifying your choice.

        \n\nDocument:\n---\n{content}"""
        
        response = model.generate_content(prompt)
        
        return jsonify({'analysis': response.text})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": f"An error occurred with the AI model: {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def chat_with_document():
    """Receives document context and a question, calls Gemini, and returns an answer."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    document = data.get('document')
    question = data.get('question')

    if not document or not question:
        return jsonify({"error": "Document context and question are required"}), 400

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a helpful AI paralegal. You have read the following legal document. a user, who is not a lawyer, will ask you questions about it. Answer their questions concisely and in simple, easy-to-understand language based ONLY on the provided document text.

        Document Text:
        ---
        {document}
        ---
        User's Question: "{question}" """
        
        response = model.generate_content(prompt)
        
        return jsonify({'answer': response.text})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": f"An error occurred with the AI model: {str(e)}"}), 500


# --- Main Entry Point ---
if __name__ == '__main__':
    # For local development, this will run the server on http://127.0.0.1:5000
    # For production (Cloud Run), a professional web server like Gunicorn will be used.
    app.run(debug=True, port=int(os.environ.get("PORT", 8080)))