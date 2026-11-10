Bank Marketing: Will the Customer Say Yes?

This is my midterm project for the ML Zoomcamp course.

1. The problem

A bank calls its customers on the phone to sell a term deposit. A term deposit is a savings product. The customer puts money in the bank for a fixed time and cannot take it out early.

Calling customers costs time and money. Most calls end with "no". Only about 12 out of 100 customers say yes. So the bank wastes many calls on people who will never say yes.

My goal: build a model that gives each customer a score between 0 and 1. The score is the chance that the customer will say yes.

How the bank can use it:

Before a campaign, the bank scores all its customers with the model.
The bank sorts the customers from the highest score to the lowest.
The bank calls the customers at the top of the list first.

With the same number of calls, the bank finds more customers who say yes.

I use only facts that the bank knows before the call. For example: age, job, loans, and the result of the last campaign.

I removed the column duration. This column says how long the phone call was. The bank only knows it after the call ends. If I used it, the model would look much better than it really is, because it would "cheat". So I dropped it on purpose.

How I measure the model: I use ROC AUC. Only 12% of customers say yes, so plain accuracy is not a good measure. A model that always says "no" would get 88% accuracy and still be useless. AUC checks if the model puts the "yes" customers above the "no" customers in the list. A score of 0.5 means guessing. A score of 1.0 is perfect.

2. The data
Name: Bank Marketing from the UCI Machine Learning Repository
File: data/bank-full.csv
Size: 45,211 customers, 16 columns plus the answer column y (yes or no)
The file uses ; between the values, not a comma.
The data comes from a Portuguese bank.
Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

The file is already in the data folder of this project. If you need to download it again, see data/README.md.

Paper that belongs to the data: S. Moro, P. Cortez and P. Rita, "A Data-Driven Approach to Predict the Success of Bank Telemarketing", Decision Support Systems, 2014.

3. What I found in the data (short summary)
There are no empty cells. Missing information is written as the word unknown. I keep unknown as its own group.
The column poutcome (result of the last campaign) is unknown for about 82% of customers. Most customers were not called in an earlier campaign.
Age has a U shape. Very young customers (18 to 25) say yes about 24% of the time. Customers older than 65 say yes about 41% of the time. Customers in the middle (36 to 55) say yes only about 9% of the time. A simple correlation number does not show this. It shows up only when I group the ages.
The older group (65+) is small (600 customers), so its number is less sure.
Customers who were called many times in this campaign are a bit less likely to say yes.
4. The models

I trained four kinds of models and tuned the settings of each one:

Model	Best settings	Validation AUC
Logistic regression	C = 0.1	0.768
Decision tree	max_depth = 15, min_samples_leaf = 100	0.777
Random forest	100 trees, max_depth = 15, min_samples_leaf = 10	0.799
XGBoost (winner)	eta = 0.1, max_depth = 3, min_child_weight = 1, 200 rounds	0.804

Final score of the winner (XGBoost):

5-fold cross-validation: AUC = 0.799 (plus or minus 0.003)
Test set (data the model never saw): AUC = 0.808

The three numbers are close to each other, so the model is stable.

Most important features: the result of the earlier campaign (poutcome = success) is by far the most important. After that come the contact type, the month of the contact, and loan information.

5. Files in this project
.
├── README.md                 this file
├── data/                     the data (bank-full.csv)
├── notebook.ipynb            data cleaning, EDA, feature importance, models, tuning
├── train.py                  trains the final model and saves it as model.bin
├── predict.py                web service (Flask) that gives the score
├── test_predict.py           sends one example customer to the service
├── model.bin                 the saved model
├── requirements.txt          packages needed to run everything on my computer
├── requirements-docker.txt   packages needed inside the Docker container
└── Dockerfile                recipe to build the container
6. How to run the project

You need Python 3.11 or newer. Open a terminal inside the project folder.

Step 1: Make a virtual environment

Windows (PowerShell):

powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

If PowerShell says scripts are disabled, run this once and try again:

powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Mac or Linux:

bash
python -m venv .venv
source .venv/bin/activate

When it works, you see (.venv) at the start of the line.

Step 2: Install the packages
bash
pip install -r requirements.txt
Step 3: Check the data

The file data/bank-full.csv must exist. It is already in this repository. If it is missing, follow data/README.md.


Step 4: Look at the notebook (optional)

The notebook `notebook.ipynb` shows all my work: data cleaning, EDA (looking at the data), feature importance, the four models and the tuning.

1. Open `notebook.ipynb` in VS Code.
2. In the top right corner, click the kernel name. Choose **Select Another Kernel**, then **Python Environments**, then the one that says `.venv`.
3. Click **Run All**, or run the cells one by one with `Shift + Enter`.

The notebook needs about 10 to 20 minutes to run. The random forest part is the slowest. You do **not** need to run the notebook to use the project. `train.py` does the final training by itself.

If you see `ModuleNotFoundError`, the wrong kernel is selected, or the packages are not installed. Check Step 1 and Step 2.

Step 5: Train the model
bash
python train.py

This prints the cross-validation score and the test score. It saves the model as model.bin. The test AUC should be about 0.808.

Step 6: Start the web service
bash
python predict.py

The service now runs at http://localhost:9696. Leave this terminal open.

Step 7: Test the service

Open a second terminal, activate .venv again, and run:

bash
python test_predict.py

You get an answer like this (the number can be different):

200 {'subscribe': False, 'subscribe_probability': 0.15}
subscribe_probability is the chance that the customer says yes.
subscribe is True when the chance is 0.5 or more. Only about 12% of customers say yes, so most answers are False. The bank should use the probability to sort its customers.

You can change the example customer inside test_predict.py. For example, set "poutcome": "success" and the probability goes up a lot.

Example input (one customer):

json
{
  "age": 42, "job": "management", "marital": "married", "education": "tertiary",
  "default": "no", "balance": 1500, "housing": "no", "loan": "no",
  "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
  "pdays": -1, "previous": 0, "poutcome": "unknown"
}

The service has two addresses:

GET /ping answers PONG (to check that the service is alive)
POST /predict takes one customer as JSON and gives back the score
7. Run with Docker

Docker puts the service and all its packages in one box (a container). It works the same on every computer.

You need Docker Desktop installed and running. The file model.bin must exist (see Step 5).

Build the container:

bash
docker build -t bank-marketing .

Run the container:

bash
docker run -it --rm -p 9696:9696 bank-marketing

Then open a second terminal and run python test_predict.py like before. Press Ctrl+C to stop the container.

8. Cloud deployment
https://machine-learning-zoomcamp-homework-kgxl.onrender.com/ping

PONG

Service URL :(https://machine-learning-zoomcamp-homework-kgxl.onrender.com)
Check that it is alive: open (https://machine-learning-zoomcamp-homework-kgxl.onrender.com/ping)in a browser. It shows PONG.
Test it: python test_predict.py https://machine-learning-zoomcamp-homework-kgxl.onrender.com
Screenshot of the test: screenshots/cloud-test.png

How I deployed it:

I pushed this project to GitHub (with model.bin and the data).
On https://render.com I made a new Web Service and connected my GitHub repository.
I chose Docker as the language. Render builds my Dockerfile by itself.
When the deploy finished, I tested the URL.
Render URL : python test_predict.py https://machine-learning-zoomcamp-homework-kgxl.onrender.com
And in the browser :
https://machine-learning-zoomcamp-homework-kgxl.onrender.com/ping

9. Limits of this project
The data is from one bank in one time period. The model may not work well for other banks or other years.
I dropped duration on purpose. The score is lower than in many online examples, but it is honest.
The model finds patterns, not causes. For example, if more customers say yes in March, it does not mean that calling in March makes them say yes.