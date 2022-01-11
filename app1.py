import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

app = Flask(__name__)
CATEGORIES = ["LOAN ISSUED","LOAN DENIED"]
df_fico_grade = pd.read_csv('grade_to_fico.csv')
df_fico_apr = pd.read_csv('grade_to_apr.csv')
# load the model from disk
with open('rf_model.sav', 'rb') as f:
    rf_test = pickle.load(f)

with open('knn_regression_grade.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('knn_funded_amnt.pkl', 'rb') as f:
    knn_fa = pickle.load(f)

home_to_int = {'MORTGAGE': 4,
               'RENT': 3,
               'OWN': 5,
               'ANY': 2,
               'OTHER': 1,
               'NONE':0 }

sub_grade_to_char={1:'A1',2:'A2',3:'A3',4:'A4',5:'A5',6:'B1',7:'B2',8:'B3',9:'B4',10:'B5'
                   ,11:'C1',12:'C2',13:'C3',14:'C4',15:'C5',16:'D1',17:'D2',18:'D3',19:'D4',20:'D5'
                   ,21:'E1',22:'E2',23:'E3',24:'E4',25:'E5',26:'F1',27:'F2',28:'F3',29:'F4',30:'F5'
                   ,31:'G1',32:'G2',33:'G3',34:'G4',35:'G5'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = []
    int_features = [x for x in request.form.values()]
    features = [np.array(int_features)]
    #features = np.array([['3600', '677', '9/2/2009', '36', '10', 'MORTGAGE', '55000', '0', '5.91', '12', '29.7', '1']])

    term = int(features[0][3])
    loan_amnt = int(features[0][0])

    # calculate credit history
    er_credit_open_date = pd.to_datetime(features[0][2])
    issue_d = pd.to_datetime("today")
    credit_hist = issue_d - er_credit_open_date
    cht = credit_hist.days

    # calculate grade from FICO
    sub_grade = knn.predict(np.reshape([int(features[0][1])], (1, -1)))[0]
    # calculate grade
    grade = round(sub_grade / 5) + 1
    # get interest rate
    apr_row = df_fico_apr[df_fico_apr['grade_num'] == sub_grade]

    if term == '36':
        term_int = 1
        int_rate = float(apr_row['36_mo'].values[0] / 1000)
        installment = float(loan_amnt) * (int_rate * (1 + int_rate) ** 36) / (((1 + int_rate) ** 36) - 1.0)

    else:
        term_int = 2
        int_rate = apr_row['60_mo'].values[0] / 100
        installment = float(loan_amnt) * (int_rate * (1 + int_rate) ** 36) / (((1 + int_rate) ** 36) - 1)

    #final_features.insert(index, number)
    # make integer
    emp_length_int = int(features[0][4])
    annual_inc_int = int(features[0][6])
    verification_status_int = int(features[0][7])
    dti_float = float(features[0][8])
    open_acc_int = int(features[0][9])
    revol_util_float = float(features[0][10])
    mort_acc_int = int(features[0][11])
    fico_avg_score_int = int(features[0][1])
    loan_amnt_int = int(loan_amnt)
    inst_amnt_ratio = float(installment / loan_amnt)

    funded_amnt = knn_fa.predict(np.reshape([loan_amnt_int], (1, -1)))[0]
    funded_amnt_int = int(funded_amnt)

    UI_df = pd.DataFrame(index=[1])
    UI_df['loan_amnt'] = loan_amnt_int
    UI_df['funded_amnt'] = funded_amnt_int
    UI_df['term'] = term_int
    UI_df['int_rate'] = int_rate
    UI_df['grade'] = grade
    UI_df['sub_grade'] = sub_grade
    UI_df['emp_length'] = emp_length_int
    UI_df['home_ownership'] = home_to_int[features[0][5].upper()]
    UI_df['annual_inc'] = annual_inc_int
    UI_df['verification_status'] = verification_status_int
    UI_df['dti'] = dti_float
    UI_df['open_acc'] = open_acc_int
    UI_df['revol_util'] = revol_util_float
    UI_df['mort_acc'] = mort_acc_int
    UI_df['credit_hist'] = cht
    UI_df['fico_avg_score'] = fico_avg_score_int
    UI_df['inst_amnt_ratio'] = inst_amnt_ratio


    Result = rf_test.predict(UI_df)
    PREDICTION = (CATEGORIES[int(Result)])

    if float(UI_df['dti'])>43:
        res = 'Debt to income too high!'
        PREDICTION = (CATEGORIES[1])
    elif float(UI_df['revol_util']) >= 90:
        res = 'Amount of credit used up too high!'
        PREDICTION = (CATEGORIES[1])
    elif int(UI_df['fico_avg_score']) < 350:
        res = 'Not Elligible for Loan due to Invalid FICO Score!'
        PREDICTION = (CATEGORIES[1])
    elif Result == 1:
        res = 'Loan Denied'
    else:
        res = 'Congratulations!'
        PREDICTION = (CATEGORIES[int(Result)])


    output = PREDICTION

    if term_int == '1':
        term_dict = 36
    else:
        term_dict = 60

    # create original output dict
    output_info = dict()
    output_info['Provided Annual Income(USD)'] = int(UI_df['annual_inc'])
    output_info['Provided FICO Score'] = int(UI_df['fico_avg_score'])
    output_info['Interest Rate(%)'] = float(UI_df['int_rate']) * 100  # revert back to percentage
    output_info['Estimated Installment Amount(USD)'] = int(installment)
    output_info['Loan term (months)'] = int(term)
    output_info['Loan Grade'] = sub_grade_to_char[35 - int(sub_grade)]
    output_info['Loan Amount'] = int(UI_df['loan_amnt'])


    return render_template('index.html', prediction_text='The Algorithm Predicted : {}'.format(output),input_info=output_info,
                                 result=res,)



if __name__ == "__main__":
    app.run(debug=False)