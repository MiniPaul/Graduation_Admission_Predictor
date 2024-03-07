from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__,static_folder='static')

data = pd.read_csv("admission.csv")
data = data.drop(columns=["Serial No."])
data.rename(columns={'GRE Score':'GRE','TOEFL Score':'TOEFL','University Rating':'UnivRating','Chance of Admission ':'Chance'},inplace=True)
x = data.drop('Chance', axis=1)
y = data['Chance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
regressors = [
    LinearRegression(),
    RandomForestRegressor(random_state=1),
    GradientBoostingRegressor(random_state=1)
]

stacked_reg = StackingCVRegressor(
    regressors=regressors,
    meta_regressor=LinearRegression(),
    cv=4
)
stacked_reg.fit(x_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gre = float(request.form['gre'])
    toefl = float(request.form['toefl'])
    univ_rating = int(request.form['univ_rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research_exp = int(request.form['research_exp'])

    pred = stacked_reg.predict([[gre, toefl, univ_rating, sop, lor, cgpa, research_exp]])
    prediction = pred*100
    
    return render_template('prediction_result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
