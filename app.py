from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))  

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/index",methods=['POST','GET'])
def index():
    return render_template("index.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        return redirect(url_for('index'))
    else:
        return render_template("login.html")


@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        return redirect(url_for('index'))
    else:
        return render_template("signup.html")

@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == "POST":
        try:
            # Extract and convert date and time inputs
            date_dep = request.form["Dep_Time"]
            Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
            Journey_month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)
            Dep_hour = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").hour)
            Dep_min = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").minute)
            
            date_arr = request.form["Arrival_Time"]
            Arrival_hour = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
            Arrival_min = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").minute)
            
            # Calculate duration in minutes
            Duration_minutes = abs((Arrival_hour * 60 + Arrival_min) - (Dep_hour * 60 + Dep_min))
            
            # Total stops
            Total_stops = int(request.form["stops"])
            
            # Airline one-hot encoding
            airline = 'Airline_' + request.form['airline'].replace(" ", "_").upper()
            
            # Source one-hot encoding
            source = 'Source_' + request.form['Source'].replace(" ", "_").upper()
            
            # Destination one-hot encoding
            destination = 'Destination_' + request.form['Destination'].replace(" ", "_").upper()
            
            # Construct feature dictionary
            features = {
                'Total_Stops': Total_stops,
                'Journey_Day': Journey_day,
                'Journey_Month': Journey_month,
                'Depature_Hours': Dep_hour,
                'Depature_Minutes': Dep_min,
                'Arrival_Hours': Arrival_hour,
                'Arrival_Minutes': Arrival_min,
                'Duration_Minutes': Duration_minutes,
                airline: 1,
                source: 1,
                destination: 1
            }
            
            # Ensure all features needed by the model are included, defaulting to 0
            feature_order = [
                'Total_Stops', 'Depature_Minutes', 'Arrival_Hours', 'Arrival_Minutes', 'Journey_Day', 'Journey_Month',
                'Duration_Minutes', 'Airline_AIR ASIA', 'Airline_AIR INDIA', 'Airline_GOAIR', 'Airline_INDIGO',
                'Airline_JET AIRWAYS', 'Airline_MULTIPLE CARRIERS', 'Airline_MULTIPLE CARRIERS PREMIUM ECONOMY',
                'Airline_SPICEJET', 'Airline_TRUJET', 'Airline_VISTARA', 'Airline_VISTARA PREMIUM ECONOMY',
                'Source_BANGLORE', 'Source_CHENNAI', 'Source_DELHI', 'Source_KOLKATA', 'Source_MUMBAI',
                'Destination_BANGLORE', 'Destination_COCHIN', 'Destination_DELHI', 'Destination_HYDERABAD',
                'Destination_KOLKATA', 'Destination_NEW DELHI'
            ]
            model_input = [features.get(f, 0) for f in feature_order]
            
            # Predict using the model
            prediction = model.predict([model_input])
            output = "{:.2f}".format(prediction[0])
            return render_template('index.html', prediction_text=f'Predicted Flight Price: ₹{output}')
        
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)