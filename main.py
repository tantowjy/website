from flask import Flask, request, jsonify, render_template
import os
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load tokenizer from pickle file
with open(r'tokenizer_A3.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='hoax_detection_A3.tflite')
interpreter.allocate_tensors()

# Get input and output tensor information
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    # Render the HTML file
    return render_template('deteksiHoaks.html')

@app.route('/deteksiBias')
def bias():
    # Render the HTML file
    return render_template('deteksiBias.html')

@app.route('/predictHoaks', methods=['POST'])
def predictHoaks():
    data = request.get_json()
    news_text1 = data['news_text1']
    news_text2 = data['news_text2']

    # Combine input texts
    news_text = [news_text1 + " " + news_text2]

    # Tokenization and padding of news
    new_sequences = tokenizer.texts_to_sequences(news_text)
    max_len = 100  # Make sure the maximum length matches the one used when training the model
    new_padded = pad_sequences(new_sequences, maxlen=max_len)

    # Convert input data to float32 type
    new_padded = new_padded.astype('float32')

    # Set the input tensor with compacted data
    interpreter.set_tensor(input_details[0]['index'], new_padded)

    # Run the interpreter to make predictions
    interpreter.invoke()

    # Get the prediction result from the output tensor
    predictions_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Interpreting prediction results
    predicted_labels_tflite = "Hoax" if predictions_tflite[0][0] > 0.5 else "Not Hoax"
    # confidence = float(predictions_tflite[0][0]) if predicted_labels_tflite == "Hoax" else float(1 - predictions_tflite[0][0])
    confidence = float(predictions_tflite[0][0])

    # Prepare response
    response = {
        'prediction': predicted_labels_tflite,
        'confidence': confidence
    }

    return jsonify(response)

@app.route('/predictBias', methods=['POST'])
def predictBias():
    data = request.get_json()
    news_text1 = data['news_text1']
    news_text2 = data['news_text2']

    # Combine input texts
    news_text = [news_text1 + " " + news_text2]

    # Tokenization and padding of news
    new_sequences = tokenizer.texts_to_sequences(news_text)
    max_len = 100  # Make sure the maximum length matches the one used when training the model
    new_padded = pad_sequences(new_sequences, maxlen=max_len)

    # Convert input data to float32 type
    new_padded = new_padded.astype('float32')

    # Set the input tensor with compacted data
    interpreter.set_tensor(input_details[0]['index'], new_padded)

    # Run the interpreter to make predictions
    interpreter.invoke()

    # Get the prediction result from the output tensor
    predictions_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Interpreting prediction results
    predicted_labels_tflite = "Hoax" if predictions_tflite[0][0] > 0.5 else "Not Hoax"
    # confidence = float(predictions_tflite[0][0]) if predicted_labels_tflite == "Hoax" else float(1 - predictions_tflite[0][0])
    confidence = float(predictions_tflite[0][0])

    # Prepare response
    response = {
        'prediction': predicted_labels_tflite,
        'confidence': confidence
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
