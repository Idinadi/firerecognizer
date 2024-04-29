from flask import Flask, request, jsonify
import json, cv2
import pickle
import numpy as np
import urllib.request

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))
sift = cv2.SIFT_create()
min_length = 768

def preprocess(text):
    resp = urllib.request.urlopen(text)
    #image = cv2.imread(text)
    image = np.asarray(bytearray(resp.read()), dtype="unit8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    final_features = descriptors.flatten()
    X_uniform = [final_features[:min_length]]
    return X_uniform



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    processed = preprocess(data['text'])
    preds = model.predict_proba(processed)[0]
    print(preds)
    if preds[0] > preds[1]:
        return jsonify({"message" : "Negative", "pred": preds[0]})
    return jsonify({"message" : "Fire Detected", "pred": preds[1]})

#if __name__=="__main__":
   # app.run(host="127.0.0.1", port=8000, debug=True)
