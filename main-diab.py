import pandas as pd
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
# py.init_notebook_mode(connected=True)
import squarify

# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
import lightgbm as lgbm
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from yellowbrick.classifier import DiscriminationThreshold

# Stats
import scipy.stats as ss
from scipy import interp
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Time
# from contextlib import contextmanager
# @contextmanager
# def timer(title):
#     t0 = time.time()
#     yield
#     print("{} - done in {:.0f}s".format(title, time.time() - t0))

#ignore warning messages
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("data/pima_indian_diabetes.csv")

print(data.info(), data.head())

# 2 datasets
D = data[(data['Outcome'] != 0)]
H = data[(data['Outcome'] == 0)]

# #------------COUNT-----------------------
# def target_count():
#     trace = go.Bar( x = data['Outcome'].value_counts().values.tolist(),
#                     y = ['healthy','diabetic' ],
#                     orientation = 'h',
#                     text=data['Outcome'].value_counts().values.tolist(),
#                     textfont=dict(size=15),
#                     textposition = 'auto',
#                     opacity = 0.8,marker=dict(
#                     color=['lightskyblue', 'gold'],
#                     line=dict(color='#000000',width=1.5)))
#
#     layout = dict(title =  'Count of Outcome variable')
#
#     fig = dict(data = [trace], layout=layout)
#     py.iplot(fig)
#
# #------------PERCENTAGE-------------------
# def target_percent():
#     trace = go.Pie(labels = ['healthy','diabetic'], values = data['Outcome'].value_counts(),
#                    textfont=dict(size=15), opacity = 0.8,
#                    marker=dict(colors=['lightskyblue', 'gold'],
#                                line=dict(color='#000000', width=1.5)))
#
#
#     layout = dict(title =  'Distribution of Outcome variable')
#
#     fig = dict(data = [trace], layout=layout)
#     py.iplot(fig)
#
# target_count()
# target_percent()


# data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#
# # Define missing plot to detect all missing values in dataset
# def missing_plot(dataset, key) :
#     null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])
#     percentage_null = pd.DataFrame((len(dataset[key]) - (len(dataset[key]) - dataset.isnull().sum()))/len(dataset[key])*100, columns = ['Count'])
#     percentage_null = percentage_null.round(2)
#
#     trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, text = percentage_null['Count'],  textposition = 'auto',marker=dict(color = '#7EC0EE',
#             line=dict(color='#000000',width=1.5)))
#
#     layout = dict(title =  "Missing Values (count & %)")
#
#     fig = dict(data = [trace], layout=layout)
#     py.iplot(fig)
#
# # Plotting
# missing_plot(data, 'Outcome')


# plt.style.use('ggplot') # Using ggplot2 style visuals
#
# f, ax = plt.subplots(figsize=(11, 15))
#
# ax.set_facecolor('#fafafa')
# ax.set(xlim=(-.05, 200))
# plt.ylabel('Variables')
# plt.title("Overview Data Set")
# ax = sns.boxplot(data = data,
#   orient = 'h',
#   palette = 'Set2')


# def correlation_plot():
#     #correlation
#     correlation = data.corr()
#     #tick labels
#     matrix_cols = correlation.columns.tolist()
#     #convert to array
#     corr_array  = np.array(correlation)
#     trace = go.Heatmap(z = corr_array,
#                        x = matrix_cols,
#                        y = matrix_cols,
#                        colorscale='Viridis',
#                        colorbar   = dict() ,
#                       )
#     layout = go.Layout(dict(title = 'Correlation Matrix for variables',
#                             #autosize = False,
#                             #height  = 1400,
#                             #width   = 1600,
#                             margin  = dict(r = 0 ,l = 100,
#                                            t = 0,b = 100,
#                                          ),
#                             yaxis   = dict(tickfont = dict(size = 9)),
#                             xaxis   = dict(tickfont = dict(size = 9)),
#                            )
#                       )
#     fig = go.Figure(data = [trace],layout = layout)
#     py.iplot(fig)
#
# correlation_plot()


# def median_target(var):
#     temp = data[data[var].notnull()]
#     temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
#     return temp
#
# def plot_distribution(data_select, size_bin) :
#     # 2 datasets
#     tmp1 = D[data_select]
#     tmp2 = H[data_select]
#     hist_data = [tmp1, tmp2]
#
#     group_labels = ['diabetic', 'healthy']
#     colors = ['#FFD700', '#7EC0EE']
#
#     fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
#
#     fig['layout'].update(title = data_select)
#
#     py.iplot(fig, filename = 'Density plot')
#
# plot_distribution('Insulin', 0)
#
# median_target('Insulin')
# data.loc[(data['Outcome'] == 0 ) & (data['Insulin'].isnull()), 'Insulin'] = 102.5
# data.loc[(data['Outcome'] == 1 ) & (data['Insulin'].isnull()), 'Insulin'] = 169.5


# plot_distribution('Glucose', 0)
# median_target('Glucose')
# data.loc[(data['Outcome'] == 0 ) & (data['Glucose'].isnull()), 'Glucose'] = 107
# data.loc[(data['Outcome'] == 1 ) & (data['Glucose'].isnull()), 'Glucose'] = 140


# plot_distribution('SkinThickness', 10)
# median_target('SkinThickness')
# data.loc[(data['Outcome'] == 0 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27
# data.loc[(data['Outcome'] == 1 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32


# plot_distribution('BloodPressure', 5)
# median_target('BloodPressure')
# data.loc[(data['Outcome'] == 0 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 70
# data.loc[(data['Outcome'] == 1 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5


# plot_distribution('BMI', 0)
# median_target('BMI')
# data.loc[(data['Outcome'] == 0 ) & (data['BMI'].isnull()), 'BMI'] = 30.1
# data.loc[(data['Outcome'] == 1 ) & (data['BMI'].isnull()), 'BMI'] = 34.3
#
# #plot distribution
# plot_distribution('Age', 0)
# plot_distribution('Pregnancies', 0)
# plot_distribution('DiabetesPedigreeFunction', 0)
#
# missing_plot(data, 'Outcome')
