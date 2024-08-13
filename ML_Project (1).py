#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import janitor


# In[5]:


data=pd.read_excel('Mental disorder symptoms.xlsx')


# In[6]:


mental_data=data.clean_names()
mental_data


# In[7]:


mental_data=mental_data.rename(columns={'ag+1_629e': 'age'})


# In[8]:


mental_data.isna().sum()


# In[9]:


mental_data.duplicated().sum()


# In[10]:


mental_data.head(20)


# In[11]:


mental_data.drop_duplicates(inplace=True)
mental_data.head(20)


# In[12]:


plt.figure(figsize=(5, 5))
plt.scatter(mental_data['age'],mental_data['disorder'])
plt.title('Age Distribution')
plt.show()


# In[13]:


plt.figure(figsize=(15, 10))
sns.countplot(x='disorder', data=mental_data)
plt.title('Disorder Count')
plt.xticks(rotation=45)
plt.show()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
mental_data['disorder'] = label_encoder.fit_transform(mental_data['disorder'])

X = mental_data.drop('disorder', axis=1)
X = X.drop('age', axis=1)

y = mental_data['disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score


# In[16]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


# In[17]:


svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)


# In[18]:


models = {'RandomForest': rf_predictions, 'SVM': svm_predictions}

for model_name, predictions in models.items():
    print(f'---{model_name}---')
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(f'F1 Score: {f1_score(y_test, predictions, average="weighted")}')
    print(f'Recall: {recall_score(y_test, predictions, average="weighted")}')
    print(classification_report(y_test, predictions))


# In[19]:


performance_data = {
    'Model': ['RandomForest', 'SVM'],
    'Accuracy': [accuracy_score(y_test, rf_predictions), accuracy_score(y_test, svm_predictions)],
    'F1 Score': [f1_score(y_test, rf_predictions, average="weighted"), f1_score(y_test, svm_predictions, average="weighted")],
    'Recall': [recall_score(y_test, rf_predictions, average="weighted"), recall_score(y_test, svm_predictions, average="weighted")]
}

performance_df = pd.DataFrame(performance_data)


# In[20]:


performance_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance')
plt.ylabel('Score')
plt.show()


# In[22]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1_weighted')
grid_search_rf.fit(X_train, y_train)


best_params_rf = grid_search_rf.best_params_
best_rf_model = grid_search_rf.best_estimator_


# In[23]:


best_rf_predictions = best_rf_model.predict(X_test)
print(f'---Tuned RandomForest---')
print(f'Accuracy: {accuracy_score(y_test, best_rf_predictions)}')
print(f'F1 Score: {f1_score(y_test, best_rf_predictions, average="weighted")}')
print(f'Recall: {recall_score(y_test, best_rf_predictions, average="weighted")}')
print(classification_report(y_test, best_rf_predictions))


# In[24]:


tuned_performance_data = {
    'Model': ['Tuned RandomForest'],
    'Accuracy': [accuracy_score(y_test, best_rf_predictions)],
    'F1 Score': [f1_score(y_test, best_rf_predictions, average="weighted")],
    'Recall': [recall_score(y_test, best_rf_predictions, average="weighted")]
}

tuned_performance_df = pd.DataFrame(tuned_performance_data)


# In[25]:


tuned_performance_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Tuned Model Performance')
plt.ylabel('Score')
plt.show()


# In[26]:


import pandas as pd
import pickle


def Find_disorder(answers):
    
    def preprocess_new_data(answers):
         yes_no_columns = [
        'feeling_nervous', 'panic', 'breathing_rapidly', 'sweating', 'trouble_in_concentration',
        'having_trouble_in_sleeping', 'having_trouble_with_work', 'hopelessness', 'anger', 'over_react',
        'change_in_eating', 'suicidal_thought', 'feeling_tired', 'close_friend', 'social_media_addiction',
        'weight_gain', 'introvert', 'popping_up_stressful_memory', 'having_nightmares', 
        'avoids_people_or_activities', 'feeling_negative', 'trouble_concentrating', 
        'blamming_yourself', 'hallucinations', 'repetitive_behaviour', 'seasonally', 'increased_energy']

    
    
         for column in yes_no_columns:
                  new_data[column] = new_data[column].replace({'YES ': 'YES', ' NO ': 'NO', 'NO ': 'NO', 'YES': 'YES', 'NO': 'NO'})
                  new_data[column] = new_data[column].map({'YES': 1, 'NO': 0})

    return new_data
  
   
    
    
    def predict_new_data(answers):
        columns = ['feeling_nervous', 'panic', 'breathing_rapidly', 'sweating',
               'trouble_in_concentration', 'having_trouble_in_sleeping',
               'having_trouble_with_work', 'hopelessness', 'anger', 'over_react',
               'change_in_eating', 'suicidal_thought', 'feeling_tired', 'close_friend',
               'social_media_addiction', 'weight_gain', 'introvert',
               'popping_up_stressful_memory', 'having_nightmares',
               'avoids_people_or_activities', 'feeling_negative',
               'trouble_concentrating', 'blamming_yourself', 'hallucinations',
               'repetitive_behaviour', 'seasonally', 'increased_energy']
        new_data = pd.DataFrame([answers], columns=columns)

        new_data_processed = preprocess_new_data(new_data)
    
  
        new_data_processed = new_data_processed.reindex(columns=columns, fill_value=0)
    
    
        predictions = loaded_model.predict(new_data_processed)
    
        return predictions
    

    predictions = predict_new_data(answers)
    
    def decode_result(predictions):
        if predictions== 0:
            return 'ADHD'
        elif predictions== 1:
            return 'ASD'
        elif predictions== 2:
            return 'LONELINESS'
        elif predictions== 3:
            return 'MDD'
        elif predictions== 4:
            return 'OCD'
        elif predictions== 5:
            return 'PDD'
        elif predictions== 6:
            return 'PTSD'
        elif predictions== 7:
            return 'ANEXITY'
        elif predictions== 8:
            return 'BiPolar'
        elif predictions== 9:
            return 'Eating Disorder'
        elif predictions== 10:
            return 'Psychotic depression'
        elif predictions== 11:
            return 'sleeping disorder'

    






# In[42]:


with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)

with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[ ]:




