# The code should be run cell by cell (because there is a lot of graphs)
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

hotel=pd.read_csv("C:/Users/Loupicha/Documents/M2/Python_Crash_Course/bookings.csv")
print(hotel.shape)
print(hotel.describe())

#%% Cleaning data

hotel.info() # checking for missing values (None)
hotel=hotel.convert_dtypes() # changing for clean types
print(hotel.dtypes)
#%%
duplicates=hotel[hotel.duplicated()]
duplicates[duplicates["adults"]==2]
hotel.drop_duplicates(inplace=True)
# With the two first lines, we saw that there were sometimes twice the same row
# for two adults (presumed to be couples) but we decided to remove them anyway
# in order to make an analysis at a household level
print(hotel.shape)
#%%
hotel["meal"]=hotel["meal"].replace(["Undefined"],"SC")
# SC and Undefined become the same categories of meal
#%%
# Ordering months :
ordered_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
hotel["arrival_date_month"] = pd.Categorical(hotel["arrival_date_month"], categories=ordered_months, ordered=True)



#%% ANALYSIS

# Proportions of cancelation/non-cancelation :
prop_cancel=(hotel["canceled"].value_counts())/(hotel["canceled"].value_counts().sum())
print(prop_cancel)


#%% Distribution of variables :

print(hotel["hotel"].describe())
sns.histplot(data=hotel,x="hotel")
#%%
print(hotel["customer_type"].describe())
sns.histplot(data=hotel,x="customer_type")
#%%
print(hotel["arrival_date_month"].describe())
sns.histplot(data=hotel,x="arrival_date_month")
plt.xticks(rotation=70)
#%%
print(hotel["stays_in_weekend_nights"].describe())
sns.displot(x=hotel["stays_in_weekend_nights"])
#%%
print(hotel["stays_in_week_nights"].describe())
#%%
# New variable for total nights spent :
hotel["total_nights"]=hotel["stays_in_weekend_nights"]+hotel["stays_in_week_nights"]
print(hotel["total_nights"].describe())
sns.histplot(data=hotel,x="total_nights")
#%%
print(hotel["adults"].value_counts(normalize=True))
#%%
print(hotel["meal"].describe())
sns.histplot(data=hotel,x="meal")
#%%
hotel2=hotel.set_index(hotel["country"]).loc[hotel["country"].value_counts()>100].drop("country",axis=1).reset_index()
# Restrictions on the countries with more than 100 reservations for readability
sns.histplot(data=hotel2,x="country")
#%%
print(hotel["market_segment"].describe())
sns.histplot(data=hotel,x="market_segment")
plt.xticks(rotation=70)
#%%
print(hotel["distribution_channel"].value_counts(normalize=True))
sns.histplot(data=hotel,x="distribution_channel")
#%%
print(hotel["reserved_room_type"].value_counts(normalize=True))
sns.histplot(data=hotel,x="reserved_room_type")
#%%
print(hotel["assigned_room_type"].value_counts(normalize=True))
sns.histplot(data=hotel,x="assigned_room_type")
#%%
print(hotel["booking_changes"].value_counts(normalize=True))
#%%
print(hotel["deposit_type"].describe())
sns.histplot(data=hotel,x="deposit_type")
#%%
print(hotel["required_car_parking_spaces"].value_counts(normalize=True))
#%%
print(hotel["total_of_special_requests"].describe())
sns.histplot(data=hotel,x="total_of_special_requests")
#%%
print(hotel["adr"].describe())
sns.kdeplot(x="adr",data=hotel)



#%% Univariate impact on cancelations :

correlation_matrix=hotel.corr()
print(correlation_matrix['canceled'].sort_values(ascending=False))
# The correlation matrix allows to get a summary of (numeric) variables of interest

#%%
sns.countplot(x="hotel",hue="canceled",data=hotel)
#%%
sns.countplot(x="arrival_date_month",hue="canceled",data=hotel)
plt.xticks(rotation=70)
#%%
sns.countplot(x="total_nights",hue="canceled",data=hotel)
print(pd.crosstab(hotel["total_nights"],hotel["canceled"]).apply(lambda r:r/r.sum(),axis=1))
# The crosstab allows to get the probabilities
#%%
sns.countplot(x="meal",hue="canceled",data=hotel)
print(pd.crosstab(hotel["meal"],hotel["canceled"]).apply(lambda r:r/r.sum(),axis=1))
#%%
sns.countplot(x="country",hue="canceled",data=hotel2) # uses restricted version (>100 reservations)
#%%
sns.countplot(x="market_segment",hue="canceled",data=hotel)
plt.xticks(rotation=70)
print(pd.crosstab(hotel["market_segment"],hotel["canceled"]).apply(lambda r:r/r.sum(),axis=1))
#%%
sns.countplot(x="distribution_channel",hue="canceled",data=hotel)
plt.xticks(rotation=70)
#%%
sns.countplot(x="assigned_room_type",hue="canceled",data=hotel)
#%%
sns.countplot(x="booking_changes",hue="canceled",data=hotel)
print(pd.crosstab(hotel["booking_changes"],hotel["canceled"]).apply(lambda r:r/r.sum(),axis=1))
#%%
sns.countplot(x="deposit_type",hue="canceled",data=hotel)
print(pd.crosstab(hotel["deposit_type"],hotel["canceled"]).apply(lambda r:r/r.sum(),axis=1))
#%%
sns.countplot(x="customer_type",hue="canceled",data=hotel)
print(pd.crosstab(hotel["customer_type"],hotel["canceled"]).apply(lambda r:r/r.sum(),axis=1))
#%%
sns.kdeplot(x="adr",hue="canceled",data=hotel)
#%%
sns.countplot(x="required_car_parking_spaces",hue="canceled",data=hotel)
#%%
sns.countplot(x="total_of_special_requests",hue="canceled",data=hotel)
print(pd.crosstab(hotel["total_of_special_requests"],hotel["canceled"]).apply(lambda r:r/r.sum(),axis=1))



#%% Multivariate analysis (interactions between variables)

# Average daily rate :
sns.lineplot(x="assigned_room_type",y="adr",data=hotel,err_style=None)
# Allows to see the price of each type of room (however other expenses are included !)
#%%
sns.lineplot(x="arrival_date_month",y="adr",data=hotel)
plt.xticks(rotation=70)
#%%
sns.lineplot(x="arrival_date_month",y="adr",hue="hotel",data=hotel)
plt.xticks(rotation=70)


#%% Number of nights :
sns.catplot(x="arrival_date_month",y="total_nights",data=hotel,kind="bar")
plt.xticks(rotation=70)
hotel.groupby(["arrival_date_month","hotel"])["total_nights"].mean().unstack().plot(kind='bar')
# using hotel to discriminate further
#%%
print(hotel.groupby(["meal","hotel"]).aggregate(nights_meal=("total_nights","mean")).unstack())
sns.catplot(x="meal",y="total_nights",data=hotel,kind="bar")
#%%
sns.boxplot(x="country",y="total_nights",data=hotel2)
print(hotel2.groupby(["country","hotel"]).aggregate(nights_country=("total_nights","mean")).unstack())
# Gives countries with number of reservations >100
#%%
print(hotel.groupby(["market_segment"]).aggregate(nights_market=("total_nights","mean")).unstack())
sns.boxplot(y=hotel['market_segment'],x=hotel['total_nights'])
#%%
print(hotel.groupby(["distribution_channel"]).aggregate(nights_distrib=("total_nights","mean")).unstack())
sns.boxplot(x="distribution_channel",y="total_nights",data=hotel)
#%%
print(hotel.groupby(["assigned_room_type"]).aggregate(nights_room=("total_nights","mean")).unstack())
sns.catplot(x="assigned_room_type",y="total_nights",data=hotel,kind="bar")
#%%
print(hotel.groupby("customer_type").aggregate(nights_custtype=("total_nights","mean")).unstack())
sns.boxplot(x="customer_type",y="total_nights",data=hotel)


#%% Customer type :
sns.countplot(x="arrival_date_month",hue="customer_type",data=hotel)
plt.xticks(rotation=70)
#%%
pd.crosstab(hotel["customer_type"],hotel["hotel"]).plot(kind='bar')
#%%
hotel.groupby(["customer_type","meal"]).size().unstack().plot.pie(figsize=(20,20),subplots=True,autopct='%1.1f%%')
#%%
sns.countplot(hue="customer_type",x="reserved_room_type",data=hotel)
#%%
print(pd.crosstab(hotel["deposit_type"],hotel["customer_type"]).apply(lambda r:r/r.sum(),axis=0))


#%% Country :
hotel3=hotel.set_index(hotel["country"]).loc[hotel["country"].value_counts()>200].drop("country",axis=1).reset_index()
# Restricting even more to countries with number of reservations > 200
#%%
sns.lineplot(x="arrival_date_month",y="adr",hue="country",data=hotel3,err_style=None)
plt.xticks(rotation=70)
#%%
sns.countplot(x="country",hue="arrival_date_month",data=hotel3)
plt.legend(bbox_to_anchor=(1,1))
#%%
sns.countplot(x="country",hue="market_segment",data=hotel3)
#%%
sns.boxplot(x="country",y="adr",hue="hotel",data=hotel3)
plt.legend(bbox_to_anchor=(1,1))
#%%
sns.countplot(x="country",hue="customer_type",data=hotel3)
plt.legend(bbox_to_anchor=(1,1))
#%%
sns.countplot(x="country",hue="deposit_type",data=hotel3)
#%%
sns.countplot(x="reserved_room_type",hue="country",data=hotel3)

#%% Trivariate analysis to enlighten further for each country (especially Portugal)
sns.catplot(x="country",hue="distribution_channel",col="canceled",data=hotel3,kind='count')
#%%
sns.catplot(x="country",hue="customer_type",col="canceled",data=hotel3,kind='count')
#%%
sns.catplot(x="country",hue="market_segment",col="canceled",data=hotel3,kind='count')


#%% Market segment :
print(hotel.groupby("market_segment").aggregate(mktseg_price=("adr","mean")).unstack())
#%%
sns.lineplot(x="arrival_date_month",y="adr",hue="market_segment",data=hotel,err_style=None) # using price this time
plt.xticks(rotation=70)
plt.legend(bbox_to_anchor=(1,1))
#%%
sns.countplot(x="market_segment",hue="deposit_type",data=hotel)
plt.xticks(rotation=70)
#%%
print(pd.crosstab(hotel["market_segment"],hotel["hotel"]).apply(lambda r:r/r.sum(),axis=0))
sns.countplot(x="market_segment",hue="hotel",data=hotel)
plt.xticks(rotation=70)
#%%
sns.countplot(x="reserved_room_type",hue="market_segment",data=hotel)
plt.legend(bbox_to_anchor=(1,1))


#%% Distribution channel :
sns.lineplot(x="arrival_date_month",y="adr",hue="distribution_channel",data=hotel,err_style=None)
plt.xticks(rotation=70)
#%%
pd.crosstab(hotel3["market_segment"],hotel["distribution_channel"]).apply(lambda r:r/r.sum(),axis=0).plot.pie(figsize=(25,25),subplots=True,autopct='%1.1f%%')
#%%
print(pd.crosstab(hotel["reserved_room_type"],hotel["distribution_channel"]).apply(lambda r:r/r.sum(),axis=0))

#%%
diff_assigned_reserved=hotel[hotel["reserved_room_type"]!=hotel["assigned_room_type"]]
print(diff_assigned_reserved.shape[0]/hotel.shape[0]) # Proportion of people whose assigned room type is different from the reserved one

prop_cancel_in_diff=(diff_assigned_reserved["canceled"].value_counts())/(diff_assigned_reserved["canceled"].value_counts().sum())
print(prop_cancel_in_diff) # Proportion of cancelations among these people 




#%% MACHINE LEARNING

hotel4=hotel2 # Copy of the data set restricted to > 100 reservations

from sklearn import tree, metrics, neighbors, model_selection
import math

hotel4 = pd.get_dummies(hotel4, columns=["hotel","arrival_date_month","meal","market_segment","distribution_channel","reserved_room_type","assigned_room_type","deposit_type","customer_type","country"])
hotel4 = hotel4.astype('int') # necessary to turn all values in integers for the rest of the code to work

hotel4.info()

#%% 
#Creating validation and test data and labels :
data = hotel4.drop(["canceled"], axis=1).to_numpy()
label = hotel4["canceled"].to_numpy()
validation_data, test_data, validation_label, test_label = model_selection.train_test_split(data, label, test_size=0.25)

#%% 
#Choosing the number of k with the highest cross validation score
knn_cv=[]
ks=np.array(range(1,201,2))
for k in ks:
    knn=neighbors.KNeighborsClassifier(k)
    knn_cv.append(model_selection.cross_val_score(knn,validation_data,validation_label,cv=5))

knn_acc=np.array([x.mean() for x in knn_cv])
knn_acc_std=np.array([x.std() for x in knn_cv])/math.sqrt(5)
best_acc_pos=knn_acc.argmax()
best_k=ks[best_acc_pos]
print("Best k :",best_k,", with accuracy :",knn_acc[best_acc_pos])

#%%

import matplotlib.pyplot as plt

# Building the graph of accuracy with respect to complexity (for knn)
fig,ax=plt.subplots()
ax.plot(ks,knn_acc)
ax.plot(ks,knn_acc+2*knn_acc*knn_acc_std,'C2')
ax.plot(ks,knn_acc-2*knn_acc*knn_acc_std,'C2')
ax.vlines(best_k,min(knn_acc),max(knn_acc+knn_acc_std),linestyles='dotted')
ax.hlines(knn_acc[best_acc_pos]-knn_acc_std[best_acc_pos],min(ks),max(ks),colors='C3',linestyles='dashed')
ax.set_xlabel("Knn")
ax.set_ylabel("Accuracy")
fig.tight_layout()
plt.show()

#%%
# Building the K-nn model (using the k previously defined) :
knn_best=neighbors.KNeighborsClassifier(best_k)
knn_best.fit(validation_data,validation_label)
knn_best_pred=knn_best.predict(test_data)
print(metrics.confusion_matrix(test_label,knn_best_pred).T)
print(metrics.accuracy_score(test_label,knn_best_pred))

#%%
# Building the decision tree model :
decision_tree=tree.DecisionTreeClassifier(min_samples_split=100)
decision_tree.fit(validation_data,validation_label)
scores=model_selection.cross_val_score(decision_tree,validation_data,validation_label, cv=5)
print(scores.mean(),scores.std()/math.sqrt(5))

tree_predict=decision_tree.predict(test_data)
print(decision_tree.get_depth(),decision_tree.get_n_leaves())
print(metrics.confusion_matrix(test_label,tree_predict).T)
print(metrics.accuracy_score(test_label,tree_predict))

#%%
# Plotting the tree :
fig, ax = plt.subplots(figsize=(32,32))
tree.plot_tree(decision_tree,filled=True,fontsize=8)

#%%
# Thanking the teacher :
print("Thank you very much for your nice Python-refresher courses !")

