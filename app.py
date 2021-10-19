import streamlit as st

from termcolor import colored
from math import pi

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dropout,Dense,TimeDistributed,Input,Reshape

from diagrams import Diagram, Cluster
from diagrams.custom import Custom

#--------------------------------------------------Human Digital Twin-------------------------------------------------------

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Digital Twin", page_icon=None)


class digital_twin:
  def __init__(self,path_list):
    ''' 
    /// columns: 'Id','Skills','Years Of Experience','Interests','Workstyles','Trainings','Trainings Passed Percentage',
    'Salary','Age Group','Gender','Height','Weight','BMI','Physical Activity Score','Lifestyles','Health Conditions' /// 
    '''
    sns.set(style='darkgrid')

    # 1. import the data
    if path_list != None: self.data_list = [pd.read_csv(path) for path in path_list]

    self.dates = [data.columns[0] for data in self.data_list]
    for i in range(0,len(self.dates)):
      header = self.data_list[i].iloc[0]
      self.data_list[i] = self.data_list[i][1:]
      self.data_list[i].columns = header
    
    numeric_columns = ['Monthly Avg Income','Age Group','BMI','Percent Present','Overall Rating']
    for i in range(len(self.data_list)):
      for col in numeric_columns:
        self.data_list[i][col] = self.data_list[i][col].apply(np.float32)

    self.career_base = pd.read_csv("data/career_base.csv")
    
    # 2. loading the ML models
    # --- career LSTM model paths
    self.career_model_path = "data/trained_models/career_model/lstm_model.h5" # model path
    self.career_inpdct_path = "data/trained_models/career_model/input_dictionary.pkl" # input dictionary path
    self.career_outdct_path = "data/trained_models/career_model/output_dictionary.pkl" # output dictionary path

    # --- mental health LSTM model
    self.mental_model_path = "data/trained_models/mental_health_model/lstm_model.h5" # model path
    self.mental_inpdct_path = "data/trained_models/mental_health_model/input_dictionary.pkl" # input dictionary path
    self.mental_outdct_path = "data/trained_models/mental_health_model/output_dictionary.pkl" # output dictionary path

    # --- physical health LSTM model
    self.physical_model_path = "data/trained_models/physical_health_model/lstm_model.h5" # model path
    self.physical_inpdct_path = "data/trained_models/physical_health_model/input_dictionary.pkl" # input dictionary path
    self.physical_outdct_path = "data/trained_models/physical_health_model/output_dictionary.pkl" # output dictionary path
    
    # --- loading the CatBoost model for predicting employee attrition
    self.attrition_model = CatBoostRegressor()
    self.attrition_model.load_model("data/trained_models/attrition_prediction_model.cbm")

    # combined health effort of all the domains/projects --> [mental effort, physical effort]
    self.combined_score = {
        'software_developer': [0.8,0.2],'senior_software_developer': [0.75,0.25],'database_admin': [0.9,0.1],'business_analyst': [0.2,0.8],'site_reliability_engineer': [0.6,0.4],
        'data_scientist': [0.85,0.15],'research_&_development': [0.95,0.05],'technical_writer': [0.7,0.3],'technical_support': [0.6,0.4],'trainer_or_teacher': [0.7,0.3],
        'technical_recruiter': [0.3,0.7],'sales_engineer': [0.1,0.9],'manager': [0.75,0.25],'scrum_master': [0.6,0.4],'machine_learning_engineer': [0.7,0.3]
    }


  # --------------------------------------------------- radar chart -------------------------------------------------------------
  def __radar_chart__(self,target_list,pred_list,kpi=None):
    # the score dictionary
    score_dict = dict()

    # initialize the score dictionary
    for fit in target_list:
      score_dict[fit] = 0

    N = len(target_list)

    for final in pred_list:
      for fit in final:
        for key in score_dict.keys():
          if key == fit:
            score_dict[key] += 1    

    # sort the dictionary
    score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=False))

    angles = [n / float(N) * 2 * pi for n in range(N)]

    values = list(score_dict.values())
    categories = list(score_dict.keys())

    ax = plt.subplot(111, polar=True)
    plt.xticks(angles, categories, color='black', size=8)

    max_idx = max(score_dict, key=score_dict.get)
    max_value = score_dict[max_idx]

    ax.set_rlabel_position(0)
    plt.yticks(list( range(0,(max_value + 1)) ), [str(n) for n in list( range(0,(max_value + 1)) )], color="red", size=8)
    plt.ylim(0,max_value)

    if values[-2] == 0:
      ax.plot(angles, values, linewidth=1, linestyle='solid')
    else:
      ax.plot(angles, values, linewidth=0, linestyle='solid')

    ax.fill(angles, values, 'b', alpha=0.4)
    ax.figure.set_size_inches(5,5)

    if kpi != None:
      kpi.pyplot()
    else:
      st.pyplot()

    return score_dict


  # --------------------------------------------------- pie chart -------------------------------------------------------------
  def __get_pieplot__(self,lst,labels=None,ax=None):
    # create data
    size_of_groups=lst

    # Create a pieplot
    ax.pie(size_of_groups)
    if labels:
      ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=labels)
    else:
      ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=lst)

    # add a circle at the center to transform it in a donut chart
    my_circle=Circle( (0,0), 0.5, color='white')
    ax.add_patch(my_circle)


  # --------------------------------------------------- bubble chart -------------------------------------------------------------
  def __bubble_chart__(self,data,axs):
    xlabels, ylabels, score = [], [], []

    for k in data.keys():
      for v in data[k]:
        xlabels.append(k)
        ylabels.append(v)
        score.append(data[k][v])

    sns.scatterplot(x=xlabels,y=ylabels,size=score,alpha=0.5,sizes=(0,4000),legend=False, palette="summer",ax=axs)
    axs.set_xlim(-1,len(set(xlabels)) + 1)
    axs.set_ylim(-1,len(set(ylabels)) + 1)
    axs.set_xlabel("Years")
    axs.set_ylabel("Roles predicted")


  # --------------------------------------------------- clean text data -------------------------------------------------------------
  def __clean_data__(self,columns,id,inp_dict,max_len):
    inp, col_list = list(), list()
    #data = self.data_list[-1].iloc[id]

    for i in range(len(self.data_list)):
      data = self.data_list[i].iloc[id].copy()
      temp = None

      for col in columns:
        data[col] = data[col].replace(", ",",")
        data[col] = data[col].strip(" ")
        data[col] = data[col].replace(" ","_")

        if temp == None: 
          temp = [w for w in data[col].split(",")]
        else:
          temp += [w for w in data[col].split(",")]
      
      inp.append(temp)
        

    #print(inp)
    # initializing the tokenizer with pre-built dictionary
    test_token = Tokenizer()
    test_token.word_index = inp_dict
    xseq = test_token.texts_to_sequences(inp)
    xpadseq = pad_sequences(xseq,maxlen=max_len,padding='post')

    return xpadseq
  

  # --------------------------------------------------- basic info -------------------------------------------------------------
  def __basic_info__(self,id):
    data = self.data_list[-1].iloc[id]

    st.header("\n1. Basic Information")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(label = "Id", value = id + 1)
    kpi3.metric(label = "Name", value = str(data['Name']))
    kpi3.metric(label = "Age", value = str(data['Age Group']))
    kpi4.metric(label = "Gender", value = str(data['Gender']))

    kpi1, kpi2 = st.columns(2)

    kpi1.metric(label = "Current Role", value = str(data['Current Role']))
    kpi2.metric(label = "Overall Rating", value = str(np.round(data['Overall Rating'],2)))

    # education & experience 
    st.subheader("\nEducation & Experience")

    total_exp = float(data["Years Of Experience Before"]) + float(data["Num Months In Company"])/12
    exp = float(data["Num Months In Company"])/12

    edu_dict = {"diploma":1,"bachelor":2,"master":3,"doctorate":4,"post doctorate":5}

    st.markdown("##")

    fig, axs = plt.subplots(ncols = 2, figsize=(10,2))
    
    sns.barplot(x = [edu_dict[data['Education'].split(",")[0]]], y = [data['Education'].split(",")[0]], ax = axs[0], palette='summer')
    axs[0].set_xlim(0,5)
    axs[0].invert_xaxis()
    axs[0].set_xlabel("Degree Level")

    sns.barplot(x = [total_exp, exp], y = ["Total Experience","Experience In The Company"], ax = axs[1], palette='twilight')
    axs[1].yaxis.tick_right()
    axs[1].set_xlabel("Years")

    #plt.show()
    st.pyplot(fig)
  

  # --------------------------------------------------- career model -------------------------------------------------------------
  def __career_lstm__(self,id):
    # load the target list
    target_list = ['software_developer','senior_software_developer','database_admin','business_analyst','site_reliability_engineer',
                   'data_scientist','research_&_development','technical_writer','technical_support','trainer_or_teacher',
                   'technical_recruiter','sales_engineer','manager','scrum_master','machine_learning_engineer']
    
    MAX_LEN = 40
    output = list()

    # load LSTM model
    model = tf.keras.models.load_model(self.career_model_path)

    # load input dictionary
    dfile = open(self.career_inpdct_path, "rb")
    inp_dict = pickle.load(dfile)

    # load output dictionary
    dfile = open(self.career_outdct_path, "rb")
    out_dict = pickle.load(dfile)

    columns = ['Skills', 'Interests', 'Workstyles']
    data = self.__clean_data__(columns,id,inp_dict,MAX_LEN)

    # predicting the possible career paths
    predicted = model.predict(data)
    final_list = list()

    for i in range(len(predicted)):
      res = list()
      pred = np.round( np.argmax(predicted[i],axis=1) )

      for w in pred:
        for word, index in out_dict.items():
          if index == w:
            res.append(word)
            break
        
      final_list.append(res)

    output.append(final_list)

    return final_list, target_list
    

  def __attrition_score__(self,id):
    # columns for attrition
    columns = ['Age Group','Monthly Avg Income','Over Time','Overall Rating','Num Months In Company','Gender']
    data = self.data_list[-1]

    data['Over Time'] = 0

    data['Gender'] = data['Gender'].apply(lambda x: x.replace(' ',''))
    data = data.replace({'female':0,'male':1})

    for i, string in zip(range(len(data)), data['Workstyles']):
      lst = [w for w in string.split(",")]

      if 'overtime' in lst:
        data.iloc[i]['Over Time'] = 1
    
    input_list = [data.iloc[id][col] for col in columns]

    score = np.round(self.attrition_model.predict(input_list) * 100, 2)

    return score


  def __career_image__(self,id):
    # career info and analysis
    st.markdown("""---""")
    st.header("\n\n2. Career Analysis")
    # --- income and rating 
    st.subheader(f"\n2.1. Income and Overall Rating")

    rate_list = list()
    income_list = list()

    for data in self.data_list:
      rate_list.append( data.iloc[id]['Overall Rating'] )
      income_list.append( data.iloc[id]['Monthly Avg Income']/1000 )

    fig, axs = plt.subplots(ncols=2,figsize=(8,4))

    sns.lineplot(x=self.dates,y=rate_list,ax=axs[0])
    axs[0].set_xlabel("Years")
    axs[0].set_ylabel("Overall Rating")

    sns.lineplot(x=self.dates,y=income_list,ax=axs[1])
    axs[1].set_xlabel("Years")
    axs[1].set_ylabel("Average Monthly Income (K)")

    fig.tight_layout(pad=2.0)
    st.pyplot()

    NUM_SKILLS, NUM_WSTYLES = 5, 3

    skill_dict, wstyle_dict = dict(), dict()

    for data in self.data_list:
      skill_string = data.iloc[id]['Skills']
      skill_list = [w for w in skill_string.split(",")]

      wstyle_string = data.iloc[id]['Workstyles']
      wstyle_list = [w for w in wstyle_string.split(",")]


      # --- skills
      for skill in skill_list:
        if skill in skill_dict.keys():
          skill_dict[skill] += 1
        else:
          skill_dict[skill] = 1
      
      # --- workstyles
      for wstyle in wstyle_list:
        if wstyle in wstyle_dict.keys():
          wstyle_dict[wstyle] += 1
        else:
          wstyle_dict[wstyle] = 1
    

    s_labels, s_values = list(), list()

    for key in list(skill_dict.keys()):
      #skill_string += ( str(key) + " (" + str(skill_dict[key] - 1) + " years), " )
      if skill_dict[key] - 1 != 0:
        s_labels.append(key)
        s_values.append(skill_dict[key] - 1)

    w_labels, w_values = list(), list()

    for key in list(wstyle_dict.keys()):
      #wstyle_string += ( str(key) + " (" + str(wstyle_dict[key] - 1) + " years), " )
      if wstyle_dict[key] - 1 != 0:
        w_labels.append(key)
        w_values.append(wstyle_dict[key] - 1)

    st.markdown("##")
    st.subheader(f"\n2.2. Skills and Workstyles based on Experience")

    fig, axs = plt.subplots(ncols=2, figsize=((len(w_labels)+len(s_labels)),5))

    sns.barplot(x=s_labels, y=s_values, ax=axs[0], palette="summer")
    axs[0].set_xlabel("Skills")
    axs[0].set_title("Skills based on years of experience")
    axs[0].set_xticklabels(s_labels,rotation=90)

    sns.barplot(x=w_labels, y=w_values, ax=axs[1], palette="summer")
    axs[1].set_xlabel("Workstyles")
    axs[1].set_title("Workstyles based on years of experience")
    axs[1].set_xticklabels(w_labels,rotation=90)

    st.pyplot(fig)

    # --- trainings till now
    st.markdown("##")

    kpi1, kpi2 = st.columns((3,1))

    kpi2.subheader(f"2.3. Performance in training")

    scores = []

    for data in self.data_list:
      completed = [w for w in data.iloc[id]['Trainings Completed'].split(",")]
      all_courses = [w for w in data.iloc[id]['Trainings Enrolled'].split(",")]

      scores.append( np.round((len(completed)*100)/len(all_courses),2) )

    fig, axs = plt.subplots(ncols=1,figsize=(6,4))

    sns.lineplot(x=self.dates,y=scores,ax=axs)
    axs.set_ylabel("Trainings completed (%)")
    
    kpi1.pyplot()

    string = ", ".join(set(all_courses) - set(completed))


    st.markdown("##")
    kpi1, kpi2 = st.columns(2)

    kpi1.caption("Trainings Completed")
    text = f'<p style="color:Black; font-size: 18px; font-weight: 500; font-family: Georgia, serif;">{", ".join(set(completed)).title()}</p>'
    kpi1.markdown(text, unsafe_allow_html=True)

    kpi2.caption("Trainings Left")
    if string != "":
      text = f'<p style="color:Red; font-size: 18px; font-weight: 500; font-family: Georgia, serif;">{string.title()}</p>'
      kpi2.markdown(text, unsafe_allow_html=True)


    # Attendance details
    st.markdown("##")
    st.subheader("\n\n2.4. Attendance percentage")
    
    length = len(self.dates)
    fig, axs = plt.subplots(ncols=length,figsize=(5*length,4))

    for i, data in zip(range(length), self.data_list):
      present = np.float32(data['Percent Present'].iloc[id])
      absent = 100 - np.float32(data['Percent Present'].iloc[id])

      value_lists = [present,absent]
      labels = [f'present: { np.round(present,1) }%',f'absent { np.round(absent,1) }%']

      self.__get_pieplot__(value_lists,labels,axs[i])  

    st.pyplot(fig)

    # career path change and future preference with time
    st.markdown("##")

    kpi1, kpi2 = st.columns((1,3))

    kpi1.subheader("\n\n2.5. Current career and best-fits")

    curr_role = self.data_list[-1].iloc[id]['Current Role']
    text = f'<p style="color:Black; font-size: 18px; font-weight: 400; font-family: Georgia, serif;">Current career: { curr_role }</p>'
    kpi1.markdown(text, unsafe_allow_html=True)
    
    career_predicted, target_list = self.__career_lstm__(id)
    score_dict = self.__radar_chart__(target_list,career_predicted,kpi2)

    kpi1, kpi2 = st.columns((1,2))
    best_fits = set()

    for fit in score_dict.keys():
      if score_dict[fit] > 0:
        best_fits.add(fit)
    
    best_fits = list(best_fits)
    best_fits.insert(0,"Select Best-Fit")
    result = kpi1.selectbox("Choose your Best-Fit", best_fits)

    kpi2.caption("Skills To Learn")

    if result == "Select Best-Fit":
      text = f'<p style="color:Black; font-size: 18px; font-weight: 400; font-family: Georgia, serif;">Choose one of your Best-Fits to know skills yet to be learnt</p>'
      kpi2.markdown(text, unsafe_allow_html=True)
    else:
      present_skills = [w for w in self.data_list[-1].iloc[id]['Skills'].split(", ")]

      indx = self.career_base.index[self.career_base['Best Fit'] == result.replace("_"," ")][0]
      all_skills = [w for w in self.career_base.iloc[indx]['Skills'].split(", ")]

      target_skills = list(set(all_skills) - set(present_skills))

      text = f'<p style="color:Black; font-size: 18px; font-weight: 400; font-family: Georgia, serif;">{", ".join(target_skills).title()}</p>'
      kpi2.markdown(text, unsafe_allow_html=True)

    # get the attrition risk score
    attrition_risk_score = self.__attrition_score__(id)
    
    st.markdown("##")
    
    st.caption("Attrition Risk Score")
    if attrition_risk_score > 30:
      text = f'<p style="color:Red; font-size: 36px; font-weight: 600; margin-top: -12px;">{attrition_risk_score}</p>'
    elif attrition_risk_score > 10 and attrition_risk_score <= 30:
      text = f'<p style="color:#EACE09; font-size: 36px; font-weight: 600; margin-top: -12px;">{attrition_risk_score}</p>'
    else:
      text = f'<p style="color:Green; font-size: 36px; font-weight: 600; margin-top: -12px;">{attrition_risk_score}</p>'

    st.markdown(text, unsafe_allow_html=True)
    '''
    risk_kpi = st.columns(1)
    risk_kpi.metric(label = "Attrition Risk Score", value = attrition_risk_score)
    '''

    return career_predicted, target_list


  # -------------------------------------------------Health Model-----------------------------------------------------------
  def __mental_lstm__(self,id):
    # load the target list
    target_list = ['mentally_healthy','slightly_disturbed','emotionally_weak','decreased_life_enjoyment',
                   'relationship_difficulties','self_harm','anxiety','depression','loss_of_productivity','insomnia','violence',
                   'burnout','dissociation']

    MAX_LEN = 50
    output = list()

    # load LSTM model
    model = tf.keras.models.load_model(self.mental_model_path)

    # load input dictionary
    dfile = open(self.mental_inpdct_path, "rb")
    inp_dict = pickle.load(dfile)

    # load output dictionary
    dfile = open(self.mental_outdct_path, "rb")
    out_dict = pickle.load(dfile)

    columns = ['Health Conditions', 'Lifestyles', 'Workstyles']
    data = self.__clean_data__(columns,id,inp_dict,MAX_LEN)

    # predicting the possible mental risks
    predicted = model.predict(data)
    final_list = list()

    for i in range(len(predicted)):
      res = list()
      pred = np.round( np.argmax(predicted[i],axis=1) )

      for w in pred:
        for word, index in out_dict.items():
          if index == w:
            res.append(word)
            break
        
      final_list.append(res)

    output.append(final_list)

    return final_list, target_list


  def __mental_image__(self,id):
    # mental risks possible
    st.markdown("""---""")

    kpi1, kpi2 = st.columns((3,1))
    kpi2.header("\n\n3. Predicted mental risks")
    
    risk_list, target_list = self.__mental_lstm__(id)

    # radar chart for the output
    self.__radar_chart__(target_list,risk_list,kpi1)

    return risk_list, target_list


  def __physical_lstm__(self,id):
    # load LSTM model
    model = tf.keras.models.load_model(self.physical_model_path)

    # load input dictionary
    dfile = open(self.physical_inpdct_path, "rb")
    inp_dict = pickle.load(dfile)

    # load output dictionary
    dfile = open(self.physical_outdct_path, "rb")
    out_dict = pickle.load(dfile)

    # load the target list
    target_list = ['diabetes_type_2','physically_healthy','heart_diseases',
                   'stroke','lung_disorder','body_pain','indigation','cancer','prediabetes']

    MAX_LEN = 30
    output = list()

    columns = ['Health Conditions', 'Lifestyles', 'Workstyles']
    data = self.__clean_data__(columns,id,inp_dict,MAX_LEN)

    # predicting the possible mental risks
    predicted = model.predict(data)
    final_list = list()

    for i in range(len(predicted)):
      res = list()
      pred = np.round( np.argmax(predicted[i],axis=1) )

      for w in pred:
        for word, index in out_dict.items():
          if index == w:
            res.append(word)
            break
        
      final_list.append(res)

    output.append(final_list)

    return final_list, target_list

  def __physical_image__(self,id):
    # mental risks possible
    st.markdown("""---""")

    kpi1, kpi2 = st.columns((1,3))
    kpi1.header("\n\n4. Predicted physical risks")
    
    risk_list, target_list = self.__physical_lstm__(id)

    # radar chart for the output
    self.__radar_chart__(target_list,risk_list,kpi2)

    # fitbit data
    st.subheader("\n\nFitbit Data Analysis")

    with st.form(key = "fitbit"):
      year = st.text_input("Enter Year")
      submit = st.form_submit_button(label = "Submit")

      st.markdown("##")

    try:
      index = self.dates.index(year)
    
      data = self.data_list[index].iloc[id]

      if int(data['Fitbit Data Path']):
        fitbit_path = f"data/testing_data/Fitbit_data - {year}/{year} - {data['Id']}.csv"
        fitbit_data = pd.read_csv(fitbit_path)

        kpi1, kpi2, kpi3 = st.columns((2,1,1))

        kpi1.metric(label = "Total Days Recorded", value = len(fitbit_data))
        if int(fitbit_data['TotalSteps'].mean()) < 10000:
          text = f'<p style="color:Red; font-size: 18px; font-weight: 500; font-family: Georgia, serif;">Average steps is less than 10000!!!</p>'
          kpi1.markdown(text, unsafe_allow_html=True)

        kpi2.metric(label = "Aprox. Daily Steps", value = int(fitbit_data['TotalSteps'].mean()))
        kpi3.metric(label = "Aprox. Daily Calories Burnt", value = np.round(fitbit_data['Calories'].mean()))

        # steps histogram
        kpi1, kpi2 = st.columns(2)

        fig, axs = plt.subplots(ncols=1)

        sns.histplot(x = fitbit_data['TotalSteps'], kde=True, bins=15, element='step', ax=axs)
        axs.set_title("Steps Distribution")

        kpi1.pyplot(fig)

        # calorie histogram
        fig, axs = plt.subplots(ncols=1)

        sns.histplot(x = fitbit_data['Calories'], kde=True, bins=15, element='step', ax=axs)
        axs.set_title("Calories Burnt Distribution")

        kpi2.pyplot(fig)

        st.markdown("##")

        kpi4, kpi5 = st.columns((3,1))

        sleep = np.round(fitbit_data['TotalMinutesSleep'].mean()/24,2)

        kpi5.metric(label = "Avgerage Daily Sleep", value = sleep)
        kpi5.markdown("---")
        kpi5.metric(label = "Avgerage Daily In Bed", value = np.round(fitbit_data['TotalMinutesInBed'].mean()/24,2))

        if sleep < 8:
          text = f'<p style="color:Red; font-size: 18px; font-weight: 500; font-family: Georgia, serif;">Average sleep time is less than 8 hrs!!!</p>'
          kpi5.markdown(text, unsafe_allow_html=True)
        
        fig, axs = plt.subplots(ncols=1)

        sns.histplot(x = fitbit_data['TotalMinutesInBed'], kde=False, bins=15, element='step', ax=axs)
        sns.histplot(x = fitbit_data['TotalMinutesSleep'], kde=False, bins=15, element='step', color='#686868', ax=axs)

        axs.set_title("Plot showing In-Bed Time & Sleep distribution")
        plt.legend(['In Bed','Sleep'])

        kpi4.pyplot(fig)

      else:
        st.write("No Fitbit data available")
    
    except ValueError:
      st.write(f"No records for year {year}")

    return risk_list, target_list


  # ---------------------------------------------------- combined model -------------------------------------------------------------
  def __balance_score__(self,id):
    # ML model can also be used
    bmi_score = {"under-weight":2,"normal":3,"over-weight":2,"obese":1}
    data = self.data_list[-1].iloc[id]

    # BMI Inference
    if data['BMI'] <= 18.5:
      string = "under-weight"
    elif data['BMI'] > 18.5 and data['BMI'] <= 25:
      string = "normal"
    elif data['BMI'] > 25 and data['BMI'] <= 31:
      string = "over-weight"
    else:
      string = "obese"

    kpi2, kpi1 = st.columns((2,1))
    
    kpi1.caption("Body Mass Index Inference")
    text = f'<p style="color:#686868; font-size: 36px; font-weight: 600; margin-top: -12px;">{string}</p>'
    kpi1.markdown(text, unsafe_allow_html=True)
    

    score = bmi_score[string]

    wb_health = ['regular health checkup', 'energetic']
    wb_lifestyles = ['meditation', 'exercise', 'disciplined', 'balanced diet', 'focused', 'humour', 'deep sleep', 'balanced life', 'empathy']
    wb_workstyles = ['problem solving', 'creative', 'effective communicative', 'patient', 'perfectionist', 'confident', 'goal oriented', 
                     'persuasive', 'collaborative', 'flexible', 'emotional intelligence']
    
    for data in self.data_list:
      health = [w for w in data.iloc[id]['Health Conditions'].split(",")]
      for style in health:
        if style in wb_health:
          score += 1

      lifestyles = [w for w in data.iloc[id]['Lifestyles'].split(",")]
      for style in lifestyles:
        if style in wb_lifestyles:
          score += 0.8
      
      workstyles = [w for w in data.iloc[id]['Workstyles'].split(",")]
      for style in workstyles:
        if style in wb_workstyles:
          score += 0.8
      
    total = bmi_score['normal'] + (0.8*len(wb_workstyles)) + (0.8*len(wb_lifestyles)) + len(wb_health) 

    score = np.round(score/total * 100)

    #st.markdown("##")
    kpi2.subheader("\n\n5.1. Wellbeing Score")
    
    fig, axs = plt.subplots(ncols=1, figsize=(6,3))
    self.__get_pieplot__([score, 100 - score],['Positive Wellbeing','Negative Wellbeing'],axs)

    kpi2.pyplot()

    if score < 50:
      text = f'<p style="color:Red; font-size: 18px; font-weight: 500; font-family: Georgia, serif;">Wellbeing score needs to be improved!!!</p>'
      kpi2.markdown(text, unsafe_allow_html=True)

    return score

  def __get_role_score__(self,role,skill_list):
    role = role.replace("_"," ")

    index = self.career_base.index[self.career_base['Best Fit'] == role][0]
    target_list = [w for w in self.career_base.iloc[index]['Skills'].split(",")]
    
    score = 0

    for skill in skill_list:
      if skill in target_list:
        score += 1

    return score/len(target_list) * 100


  def __combined_image__(self,id,career_predicted,mental_predicted,physical_predicted):
    physical_score = {
        'diabetes_type_2':2,'physically_healthy':0,'heart_diseases':3,
        'stroke':3,'lung_disorder':2,'body_pain':1,'indigation':1,'cancer':3,'prediabetes':1
    }

    mental_score = {
        'mentally_healthy':0,'slightly_disturbed':1,'emotionally_weak':1,'decreased_life_enjoyment':1,
        'relationship_difficulties':2,'self_harm':3,'anxiety':3,'depression':3,'loss_of_productivity':2,'insomnia':2,'violence':3,
        'burnout':2,'dissociation':2
    }

    st.markdown("""---""")
    st.header("\n\n5. Final Insight")

    #get the balance score
    self.__balance_score__(id)
    
    # create a relation between current role and predicted roles
    roles, ylabels, current_roles = dict(), set(), list()

    for i, data in zip(range(len(self.data_list)),self.data_list):
      skill_list = [w for w in data.iloc[i]['Skills'].split(",")]
      temp = dict()

      for fit in career_predicted[i]:
        if fit not in temp.keys():
          temp[fit] = self.__get_role_score__(fit,skill_list)

      roles[self.dates[i]] = temp

    # create stacked bar to show mental vs physical health risk score
    st.markdown("##")
    st.subheader("\n5.2. Mental Health Vs Physical Health Risk")

    mental, physical, scale_mental, scale_physical = 0, 0, {}, {}

    for risk_list in mental_predicted:
      for risk in risk_list:
        mental += mental_score[risk]
        if risk in scale_mental.keys():
          scale_mental[risk] += 1
        else:
          scale_mental[risk] = 1

    
    for risk_list in physical_predicted:
      for risk in risk_list:
        physical += physical_score[risk]
        if risk in scale_physical.keys():
          scale_physical[risk] += 1
        else:
          scale_physical[risk] = 1
    
    kpi1, kpi2 = st.columns((3,1))

    fig, axs = plt.subplots(ncols=1, figsize=(7,5))

    mental_total = sum(list(mental_score.values())) * scale_mental[max(scale_mental, key=scale_mental.get)]
    physical_total = sum(list(physical_score.values())) * scale_physical[max(scale_physical, key=scale_physical.get)]

    mental = mental/mental_total
    physical = physical/physical_total

    sns.barplot(x = [mental, physical], y=['Mental Risk','Physical Risk'], ax=axs)
    axs.set_xlim(0,1)
    axs.set_xlabel("Mental Risk VS Physical Risk")
    
    kpi1.pyplot()
    

    if mental > physical:
      text = f'<p style="color:Black; font-size: 18px; font-weight: 400; font-family: Georgia, serif;">Mental Health is needed to be taken care of!!!</p>'
      kpi2.markdown(text, unsafe_allow_html=True)
    else:
      text = f'<p style="color:Black; font-size: 18px; font-weight: 400; font-family: Georgia, serif;">Physical Health is needed to be taken care of!!!</p>'
      kpi2.markdown(text, unsafe_allow_html=True)
    
    # combined model
    best_fits = roles[list(roles.keys())[-1]].keys()
    similarity = pd.DataFrame( index=['Mental Effort','Physical Effort'], columns=best_fits )

    for col in best_fits:
      similarity.loc['Mental Effort'][col] = (self.combined_score[col][0]) * (1 - mental)
      similarity.loc['Physical Effort'][col] = (self.combined_score[col][1]) * (1 - physical)
    
    similarity = similarity/similarity.to_numpy().sum()
    similarity = similarity.apply(np.float32)
    
    st.markdown("##")
    st.subheader("\n5.3. Combined Recommendations")

    text = f'<p style="color:Black; font-size: 18px; font-weight: 400; font-family: Georgia, serif;">The final recommendations are based on both the career parameters (skills, interests, workstyles) and health parameters (current health conditions and lifestyle)</p>'
    st.markdown(text, unsafe_allow_html=True)

    kpi1, _ = st.columns((2,1))

    fig, axs = plt.subplots(nrows=1)

    sns.heatmap(similarity, annot=True, cmap='Blues', ax=axs)
    axs.set_xlabel('List of Best-Fits')
    axs.set_ylabel('Mental & Physical Health')
    axs.set_title("Health Compatibility with Best-Fit")

    kpi1.pyplot()

    # create a bubble chart visualization
    _, kpi1 = st.columns((1,5))

    fig, axs = plt.subplots(nrows=1)

    self.__bubble_chart__(roles,axs)
    axs.set_title("Timeline View of Best-Fits")

    for col in roles[self.dates[-1]].keys():
      similarity[col] = similarity[col] * roles[self.dates[-1]][col]
    
    max_value, final_insight = 0, ""
    for col in similarity.columns:
      if similarity[col].max() > max_value:
        final_insight = col
        max_value = similarity[col].max()
    
    fig.tight_layout(pad=2.0)
    kpi1.pyplot()

    return final_insight


  # ---------------------------------------------------- xxxxxxxxxxxx ---------------------------------------------------------------
  def digital_image(self,id):
    id = id - 1

    total_experience = f"{int(int(self.data_list[-1].iloc[id]['Years Of Experience Before']) + (int(self.data_list[-1].iloc[id]['Num Months In Company']) / 12))} years {int(self.data_list[-1].iloc[id]['Num Months In Company']) % 12} months"
    
    # basic info
    self.__basic_info__(id)

    # career image
    career_predicted, career_target = self.__career_image__(id)

    # health image
    mental_risk, mental_target = self.__mental_image__(id)
    physical_risk, physical_target = self.__physical_image__(id)

    # combined image
    final_insight = self.__combined_image__(id, career_predicted, mental_risk, physical_risk)

    # flow diagram of this year
    with Diagram(f"Digital Sketch of {self.data_list[-1].iloc[id]['Email Id'].split('@')[0].title()}", show=True, filename="data/images/digital_twin", direction='TB'):
      human_image = Custom(f"Employee Id: {id + 1}, Age: {self.data_list[-1].iloc[id]['Age Group']}\nEducation: {self.data_list[-1].iloc[id]['Education'].title()}\nYears of Experience: {total_experience}", f"../images/{self.data_list[-1].iloc[id]['Email Id'].split('@')[0]}.png")
      digital_image = Custom("Digital Twin", "../images/cyborg.png")

      human_image >> digital_image

      with Cluster("Health Image"): 
        physical_image = Custom("Physical Health\nImage", "../images/physical_health.png")
        mental_image = Custom("Mental Health\nImage", "../images/mental_health.png")  

      career_image = Custom("Career Image", "../images/career.png")

      digital_image >> career_image

      with Cluster("Possible Best-Fits"):
        for career in set(career_predicted[-1]):
          if career == final_insight:
            best_match = Custom("", f"../images/{career}.png")
            career_image - best_match
          else:
            career_image - Custom("", f"../images/{career}.png")
        
      best_match - Custom("Best Match", f"../images/{final_insight}.png")

      digital_image >> mental_image

      for risk in set(mental_risk[-1]):
        mental_image - Custom("", f"../images/{risk}.png")

      digital_image >> physical_image

      for risk in set(physical_risk[-1]):
        physical_image - Custom("", f"../images/{risk}.png")

    st.subheader("Digital Twin Summary")
    st.image("data/images/digital_twin.png")
  

  def overview(self,):
    st.header("Admin View")

    kpi1, kpi2, kpi3, kpi4 = st.columns((1,1,1,1))

    change = len(self.data_list[-1]) - len(self.data_list[-2])
    kpi1.metric(label = "Team Count", value = len(self.data_list[-1]), delta = change)  

    change = np.round((self.data_list[-1]['Percent Present'].mean() - self.data_list[-2]['Percent Present'].mean())/self.data_list[-2]['Percent Present'].mean()*100)
    kpi2.metric(label = "Average Absenteeism", value = np.round(self.data_list[-1]['Percent Present'].mean(),2), delta = f'{change}%')

    change = np.round((self.data_list[-1]['Monthly Avg Income'].sum() - self.data_list[-2]['Monthly Avg Income'].sum())/self.data_list[-2]['Monthly Avg Income'].sum()*100)
    kpi3.metric(label = "Average Monthly Expense", value = np.round(self.data_list[-1]['Monthly Avg Income'].sum(),2), delta = f'{change}%')

    change = np.round((self.data_list[-1]['Overall Rating'].mean() - self.data_list[-2]['Overall Rating'].mean())/self.data_list[-2]['Overall Rating'].mean()*100) 
    kpi4.metric(label = "Average Overall Rating", value = np.round(self.data_list[-1]['Overall Rating'].mean(),2), delta = f'{change}%')

    kpi5, kpi6 = st.columns((1,1))

    kpi5.metric(label = "Average Team Age", value = np.round(self.data_list[-1]['Age Group'].mean(),2))

    val = np.round(self.data_list[-1]['Years Of Experience Before'].apply(np.float32).mean() + (self.data_list[-1]['Num Months In Company'].apply(np.float32).mean())/12,2)
    kpi6.metric(label = "Average Team Experience", value = f'{val} Years')

    st.subheader("Team Diversity")

    domain_dict = dict()
    for i in range(len(self.data_list[-1])):
      key = self.data_list[-1].iloc[i]['Current Role']

      if key not in domain_dict.keys():
        domain_dict[key] = 1
      else:
        domain_dict[key] += 1
    
    gender_dict = dict()
    for i in range(len(self.data_list[-1])):
      key = self.data_list[-1].iloc[i]['Gender']

      if key not in gender_dict.keys():
        gender_dict[key] = 1
      else:
        gender_dict[key] += 1

    fig, axs = plt.subplots(ncols=2)

    self.__get_pieplot__(domain_dict.values(),labels=domain_dict.keys(),ax=axs[1])
    axs[1].set_title("Domain distribution")

    self.__get_pieplot__(gender_dict.values(),labels=gender_dict.keys(),ax=axs[0])
    axs[0].set_title("Gender distribution")

    st.pyplot(fig)

    st.markdown("""---""")
    st.header("Get Employee Info")

    with st.form(key = "admin_info"):
      id = st.number_input("Enter Employee Id")
      submit = st.form_submit_button(label = "Submit")

    if id in self.data_list[-1]['Id']:
      dt.digital_image(int(id))
    else:
      st.text("Please enter a valid Id")

# ------------------------------------------------------ End of Class ----------------------------------------------------


check_dict = {'swagata@dt.com': (1,"abcd"), 'sounak@dt.com': (2,"efgh"), 'akshey@dt.com': (3,"ijkl")}
admin_dict = {'admin@dt.com': "admin"}

path_list = ["data/testing_data/Test_dataset - 2010.csv",
             "data/testing_data/Test_dataset - 2011.csv",
             "data/testing_data/Test_dataset - 2012.csv"]


st.title("Human Digital-Twin Prototype")
st.subheader("A digital twin with insights into overall living conditions of an individual.")
st.text("Developed by Sounak & Swagata")


def login_form():
  with st.form(key = "authentication"):
    email_id = st.text_input("Enter Email-Id")
    password = st.text_input("Enter password", type="password")

    submit = st.form_submit_button(label = "Submit")

  return email_id, password


dt = digital_twin(path_list)
email_id, password = login_form()

if email_id not in admin_dict.keys():

  if email_id in check_dict.keys():
    if password == check_dict[email_id][1]:
      st.text("Authentication Successfull")
      dt.digital_image(check_dict[email_id][0])
    else:
      st.text("Wrong Id or Password")
  else:
    st.text("Please provide your credentials")

else:

  if password == admin_dict[email_id]:
    dt.overview()
  else:
    st.text("Wrong Id or Password")

