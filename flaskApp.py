import logging
import time
from flask import Flask, request, jsonify, send_from_directory, Response
from scrape import TextIndexer
from apiModels import query_openai, query_llama, query_falcon
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r'*': {'origins': '*'}})

indexer = TextIndexer()

# Route to serve the HTML file
@app.route('/')
def assignment():
    return send_from_directory('.', 'modelComparsion.html')

@app.route('/initialize', methods=['GET'])
def initialize():
    try:
        base_url = "https://u.ae/en/information-and-services"
        indexer.scrape_and_index(base_url)
        return "Database initialized with scraped data."
    except Exception as e:
        logging.error(f"Initialization error: {str(e)}")
        return jsonify({'error': 'Initialization failed'}), 500

def stream_response(generator):
    def event_stream():
        for message in generator:
            yield f"data: {message}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

def construct_prompt(query, context):
    instructions = (
        "You are a knowledgeable assistant. Use the provided context to answer the user's question. "
        "Do not provide information that is not included in the context. If the context does not contain the answer, say 'I don't know based on the provided context.'."
    )
    prompt = f"{instructions}\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    return prompt

@app.route('/search_stream', methods=['GET'])
def search_stream():
    logging.debug("Entered /search_stream route")
    query = request.args.get('query')
    logging.debug(f"Query parameter: {query}")

    if not query:
        logging.debug("No query parameter provided")
        return jsonify({'error': 'Query parameter is required'}), 400

    def generate_responses(query):
        try:
            logging.debug(f"Received search request: {query}")
            results = indexer.search(query, k=5)
            if not results['documents']:
                yield 'data: {"error": "No relevant text found"}\n\n'
                return
            
            context_documents = [doc[0] for doc in results['documents']]
            context = " ".join(context_documents)
            
            models = {
                'GPT-3.5 Turbo': lambda prompt: query_openai(prompt, model="gpt-3.5-turbo"),
                'GPT-4': lambda prompt: query_openai(prompt, model="gpt-4"),
                'Llama': query_llama,
                'Falcon': query_falcon
            }
            
            for model_name, model_function in models.items():
                start_time = time.time()
                prompt = construct_prompt(query, context)
                model_response = model_function(prompt)
                end_time = time.time()

                response_time = end_time - start_time
                relevance_score = calculate_relevance_score(query, model_response)
                accuracy_score = calculate_accuracy_score(model_response)

                yield f'data: {{"model": "{model_name}", "response": "{model_response.strip()}", "response_time": {response_time}, "relevance_score": {relevance_score}, "accuracy_score": {accuracy_score}}}\n\n'
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            yield 'data: {"error": "Search failed"}\n\n'

    return stream_response(generate_responses(query))

def calculate_relevance_score(query, response):
    vectorizer = TfidfVectorizer().fit_transform([query, response])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def calculate_accuracy_score(response):
    expected_phrases = ["important term1", "key term2", "expected phrase3"]
    found_phrases = sum(phrase in response for phrase in expected_phrases)
    accuracy_score = found_phrases / len(expected_phrases)
    return accuracy_score

# Utility route to list all routes
@app.route('/routes', methods=['GET'])
def list_routes():
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        line = f"{rule.endpoint}: {methods} {rule}"
        output.append(line)
    return jsonify(output)

@app.before_request
def log_request_info():
    logging.debug(f"Request Method: {request.method}, Path: {request.path}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
