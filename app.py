#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
from feature import generate_data_set
# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X,y)

app = Flask(__name__)
blacklist = ['example.com', 'google.com', 'yahoo.com']
data_file_path = 'data.data'



@app.route("/")
def index():
    return render_template("index.html", xx= -1)

@app.route('/dashboard')
def dashboard():
   return render_template('dashboard.html', xx= -1)

@app.route('/index')
def index1():
   return render_template('index.html')


@app.route('/home')
def home():
   return render_template('index.html')


@app.route('/aboutus')
def aboutus():
   return render_template('aboutus.html')

@app.route('/register')
def registration():
   return render_template('register.html')


@app.route('/blacklist')
def login():
   return render_template('blacklist.html')


@app.route('/explore')
def explore():
   return render_template('explore.html')
@app.route('/index2')
def index2():
    return render_template('index2.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('dashboard.html',xx =round(y_pro_non_phishing,2),url=url )
        # else:
        #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
        #     return render_template('index.html',x =y_pro_non_phishing,url=url )
    return render_template("dashboard.html", xx =-1)
@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form['url']
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Perform web scraping here using BeautifulSoup
    
    # Example: Extracting all the links from the page
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))
    
    return render_template('index2.html', links=links)

@app.route('/check_url', methods=['POST'])
def check_url():
    url = request.form['url']
    
    # Check if the URL is blacklisted
    # if url in blacklist:
    #     result = 'The URL is not blacklisted.'
    # else:
    #     result = 'The URL is  blacklisted.'

    # Check if the URL is present in the .data file
    if check_in_data_file(url):
        result = ' The URL is blacklisted.'
    else:
        result = ' The URL is  not blacklisted.'

    return render_template('blacklist.html', result=result)

def check_in_data_file(url):
    with open('two-level-tlds.data', 'r') as file:
        for line in file:
            if url in line:
                return True
    return False


if __name__ == "__main__":
    app.run(debug=True)