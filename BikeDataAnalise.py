#数据处理包导入

import numpy as np
import pandas as pd

#画图包导入
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
sns.set()

#日期处理包导入
import calendar
from datetime import datetime 

# #jupyter notebook绘图设置
# %matplotlib inline
# %config InlineBackend.figure_format="retina"

#读取数据
BikeData = pd.read_csv("train.csv")
# date_list = BikeData.datetime.values.tolist()
# date = []
# for temp_date in date_list:
#     if temp_date == 'None' or temp_date == 'Null':
#         temp_date == '-1'
#     else:
#         temp_date = temp_date.split(' ')[0]
#     date.append(temp_date)
# BikeData["date"] = date
# print(BikeData.date)
BikeData["date"] = BikeData.datetime.apply(lambda x: x.split()[0])
BikeData["hour"] = BikeData.datetime.apply(lambda x: x.split()[1].split(":")[0])
# print(BikeData.datetime[1])
datestring = BikeData.datetime[1].split()[0]
# print(datestring)
BikeData["weekday"] = BikeData.date.apply(lambda datestring: calendar.day_name[datetime.strptime(str(datestring),r"%Y-%m-%d").weekday()])
BikeData["month"] = BikeData.date.apply(lambda datestring: calendar.month_name[datetime.strptime(str(datestring),r"%Y-%m-%d").month])

BikeData["season_label"] = BikeData.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

#天气映射处理
BikeData["weather_label"] = BikeData.weather.map({1:"sunny",2:"cloudy",3:"rainly",4:"bad-day"})

#是否是节假日映射处理
BikeData["holiday_map"] = BikeData["holiday"].map({0:"non-holiday",1:"hoiday"})
msno.matrix(BikeData,figsize=(12,5))
plt.rcParams['font.sans-serif']=['SimHei']   #这两行用来显示汉字
plt.rcParams['axes.unicode_minus'] = False
plt.show()
correlation = BikeData[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(correlation)
mask[np.tril_indices_from(mask)] = False
fig , ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(correlation,mask=mask,vmax=.8,square=True,annot=True)
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}
# 添加第一个子图， 租车人数分布的箱线图
sns.boxplot(data=BikeData,y="count",orient="v",ax=axes[0][0])

# 添加第二个子图，租车人数季节分布的箱线图
sns.boxplot(data=BikeData,y="count",x="season",orient="v",ax=axes[0][1])

# 添加第三个子图，租车人数时间分布的箱线图
sns.boxplot(data=BikeData,y="count",x="hour",orient="v",ax=axes[1][0])

#添加第四个子图，租车人数工作日分布的箱线图
sns.boxplot(data=BikeData,y="count",x="workingday",orient="v",ax=axes[1][1])

# 设置第一个子图坐标轴和标题
axes[0][0].set(ylabel='Count',title="Box Plot On Count")

# 设置第二个子图坐标轴和标题
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")

# 设置第三个子图坐标轴和标题
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")

# 设置第四个子图坐标轴和标题
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")

plt.show()
BikeData["humidity_band"] = pd.cut(BikeData['humidity'],5)
BikeData["temp_band"] = pd.cut(BikeData["temp"],5)

#假期字段映射处理
BikeData["holiday_map"] = BikeData["holiday"].map({0:"non-holiday",1:"hoiday"})

sns.FacetGrid(data=BikeData,row="humidity_band",size=3,aspect=2).\
map(sns.barplot,'temp_band','count','holiday_map',palette='deep',ci=None).\
add_legend()

plt.show()
sns.FacetGrid(data=BikeData,size=8,aspect=1.5).\
map(sns.pointplot,'hour','count','season_label',palette="deep",ci=None).\
add_legend()

plt.show()
sns.FacetGrid(data=BikeData,size=8,aspect=1.5).\
map(sns.pointplot,'month','count','weather_label',palette="deep",ci=None).\
add_legend()

plt.show()

sns.FacetGrid(data=BikeData,size=8,aspect=1.5).\
map(sns.pointplot,'hour','count','weekday',palette="deep",ci=None).\
add_legend()

plt.show()
