import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics


def get_performance(predictions, y_test, classes=[1, 0]):
    # Put your code
    accuracy = accuracy_score(y_test, predictions)  
    precision = precision_score(y_test, predictions)  
    recall = recall_score(y_test, predictions) 
    f1_scor = f1_score(y_test, predictions)  
    
    report = classification_report(y_test, predictions, labels = classes)
    
    cm = confusion_matrix(y_test, predictions, labels = classes)
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_scor)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1_scor


def plot_roc(model, y_test, features):
    # Put your code
    y_pred_probs = model.predict_proba(features)
    y_pred = y_pred_probs[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred)
    roc_auc = roc_auc_score(y_test,  y_pred)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc