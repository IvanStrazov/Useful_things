__name__ = 'my_functions'
__version__ = '1.0.3'
__info__ = 'My own functions for public using. Author: Ivan Strazov.'


import pandas as pd, numpy as np

def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type) == 'bool':
                continue
            elif str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file, sep=',', header='infer', index_col=None, report=False):
    """
    create a dataframe&report and optimize its memory usage
    """
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, sep=sep, header=header, index_col=index_col)
    df = reduce_mem_usage(df)
    
    if report == True:
        display(df.head())
        print('Shape', end='\n\n')
        display(df.shape)
        print('\nTypes', end='\n\n')
        display(df.dtypes)

        if df.isnull().sum().sum() > 0:
            print('\nNaN', end='\n\n')
            display(df.isnull().sum())
        else:
            print('\nNaN\n-')
    
    return df




from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def cm_plot(y_test, y_pred):
    '''
    plot simple & normal confusion matrix
    '''
    # Simple confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.matshow(cm, cmap=plt.cm.gray);
    plt.show();
    
    # Normal confusion matrix
    cm = cm / cm.sum(axis=1, keepdims=True)
    plt.matshow(cm, cmap=plt.cm.gray);
    plt.show();
    

    
    
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def quality_metrics(y, y_pred):
    mv = []
    mv.append(['Accuracy', round(accuracy_score(y, y_pred), 2)])
    mv.append(['Precision', round(precision_score(y, y_pred), 2)])
    mv.append(['Recall', round(recall_score(y, y_pred), 2)])
    mv.append(['F1', round(f1_score(y, y_pred), 2)])
    
    # Metrics
    print(SimpleTable(mv, ['Metric', 'Value']))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    cmp = cm*100/cm.sum()
    cmp = np.round_(cmp, 2)
    print(SimpleTable(np.append([['Negative_Model','Positive_Model']], cm, axis=0).T, 
                      ['Amount','Negative_Real','Positive_Real']))
    print(SimpleTable(np.append([['Negative_Model','Positive_Model']], cmp, axis=0).T, 
                      ['Percent','Negative_Real','Positive_Real']))
    
    
    
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_roc_curve(model, X, y):
    '''
    plot ROC
    '''
    y_pred = model.predict_proba(X)[:,1]
    
    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")

    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1)
    
    plt.plot(fpr, tpr, lw=2, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC = {}'.format(round(auc(fpr, tpr), 2)))
    plt.show()
    
def plot_precision_recall_curve(model, X, y):
    '''
    plot PRC
    '''
    y_pred = model.predict_proba(X)[:,1]
    
    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")

    plt.figure(figsize=(10, 8))
    precisions, recalls, thresholds = precision_recall_curve(y, y_pred, pos_label=1)
    
    plt.plot(recalls, precisions, lw=2, label='PRC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PRC-AUC = {}'.format(round(auc(recalls, precisions), 2)))
    plt.show()