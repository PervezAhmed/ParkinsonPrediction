import numpy as np
from flask import  Flask, request,render_template
import pickle
import numpy
import pandas

app  = Flask(__name__, template_folder='templates', static_folder='static')

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_text_sp=[]
    for key, value in request.form.items():
        input_text_sp.append(value)
        # print("key: {0}, value: {1}".format(key, value))

    np_data = np.asarray(input_text_sp, dtype=np.float32)
    prediction = model.predict(np_data.reshape(1,-1))

    if prediction == 1:
        output = "This person has a parkinson disease"
    else:
        output = "This person has no parkinson disease"

    return render_template("index.html", message= output)

if __name__ == "__main__":
    app.run(debug=True)