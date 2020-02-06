__name__ = 'my_functions'
__version__ = '1.0.4
__info__ = 'My own functions and modifications of useful fucntions from public sources.'
__author__ = 'Ivan Strazov'



def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    import pandas as pd, numpy as np
    
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


def import_data(file, 
                sep=',', decimal=',', header='infer', index_col=None, 
                report=False):
    """
    create a dataframe & report and optimize its memory usage
    """
    import pandas as pd
    
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, sep=sep, decimal=decimal, header=header, index_col=index_col)
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




def quality_metrics(y, y_pred):
    '''
    Quality metrics for binary classification in pretty table
    '''
    from statsmodels.iolib.table import SimpleTable
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    
    
    


def plot_roc_curve(y_test, y_pred_proba):
    '''
    plot ROC
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc
    
    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")

    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    
    plt.plot(fpr, tpr, lw=2, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC = {}'.format(round(auc(fpr, tpr), 2)))
    plt.show()
    
def plot_precision_recall_curve(y_test, y_pred_proba):
    '''
    plot PRC
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve, auc
    
    sns.set(font_scale=1.5)
    sns.set_color_codes("muted")

    plt.figure(figsize=(10, 8))
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
    
    plt.plot(recalls, precisions, lw=2, label='PRC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PRC-AUC = {}'.format(round(auc(recalls, precisions), 2)))
    plt.show()
    
    
    
def p_weibull(l):
    '''
    Input: list of player activity on all lifecycle
    
    Output:
    1) Aggregated indicators;
    2) Probability that the event must occur during period of last non-activity;
    3) Weibull distribution compliance level.
    '''
    
    import numpy as np
    
    def list_creation(l):
        '''
        Input: binary activity list
        Output: non-activity list, delta
        '''
        for i in range(len(l)):
            if l[i] == 0:
                continue
            else:
                break

        l = l[i:]
        l_len = len(l)
        lst = [1]

        for i in l:
            if i == 0:
                lst[-1] += 1
            else:
                lst.append(1)
                
        delta = lst[-1]
        lst = np.array(lst)
        lst = lst[lst > 0].tolist()
        
        lst_fin = list()
        for i in lst:
            lst_fin.extend([j for j in range(1, i+1)])
        
        return sorted(lst_fin), delta, l_len
    
    def F_creation(l):
        '''
        list to linear features
        '''
        n = len(l)
        y = list()

        for i in set(l):
            amount = l.count(i)
            y.append(amount)

        l = np.array(list(set(l)))
        y = np.array(y)
        y = y / y.sum()
        y = y.cumsum().astype('float64')

        X_lin = np.log(l + 1e-6).reshape(l.shape[0], 1)
        y_lin = np.log((np.log(1 / (1 - y + 1e-6))))
        
        return X_lin, y_lin
    
    def find_ab(X, y):
        '''
        calculate parameters of weibull distribution and reliability lvl
        '''
        
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(X, y)
        score = lr.score(X, y)
        beta = lr.coef_[0]
        alpha = np.exp(-lr.intercept_/beta)
        
        return alpha, beta, score
    
    def probability(delta, alpha, beta):
        '''
        the probability that the event will occur during the specified period
        '''
        
        return 1 - np.exp(- ((delta + 1e-6) / (alpha + 1e-6)) ** (beta + 1e-6))
    
    l, delta, l_len = list_creation(l)
    
    if len(l) > 5:
        l_np = np.array(l)
        l_ikr = delta / (np.quantile(l_np, 0.75) + 1.5 * (np.quantile(l_np, 0.75) - np.quantile(l_np, 0.25)))
        l_med = delta / np.median(l_np)
        l_75 = delta / np.quantile(l_np, 0.75)
        l_max = l_np.max()
        
        X, y = F_creation(l)
        
        if X.shape[0] > 2:
            alpha, beta, score = find_ab(X, y)
            
            return np.array([np.round(score,3), probability(delta, alpha, beta), l_ikr, l_med, l_75, l_len, delta])

        else:
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    else:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    
    
def feat_imp(cols, fi):
    '''
    Function for amazing showing of feature importances
    
    Input:
    1) cols - list of feature names
    2) fi - np.array of feature importances
    
    Output:
    1) Table with features and their importances;
    2) Vizualization over barplot.
    '''
    
    import numpy as np
    from statsmodels.iolib.table import SimpleTable
    
    fi = np.round(fi, 3)
    indices = np.argsort(fi)[::-1]
    cols = [cols[i] for i in indices]
    
    
    print(SimpleTable(np.append([cols], [fi], axis=0).T,
                      ['Feature','Importance']))
    
    all_colors = list(plt.cm.colors.cnames.keys())
    c = np.random.choice(all_colors, fi.shape[0], replace=False)
    
    plt.figure()
    plt.title('Feature importances')
    plt.bar(range(fi.shape[0]), fi[indices], color=c, width=.5)
    plt.xticks(range(fi.shape[0]), cols, rotation=45)
    plt.show();