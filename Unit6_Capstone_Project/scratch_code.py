






'''

# Regression 1 : Multiple Linear Regression
features = df[['education_code', 'age']]
labels = df['income']



x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=1)
lm = LinearRegression()
model = lm.fit(x_train, y_train)
print(lm.score(x_test, y_test))


# Regression 2 : K-Nearest Neighbors Regression
features = df[['education_code', 'age', 'job']]
labels = df['income']


# Classification 1 : K-Nearest Neighbors Classification
features = df[['diet_code', 'job']]
labels = df['sex']

# Classification 2 : K-Nearest Neighbors Classification
features = df[['diet_code', 'job']]
labels = df['sex']





# Blues for plots
blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "0E1E2B"]


plt.hist(df.age, bins=50, facecolor='#51ACC5', alpha=1.0)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution of Users")
plt.xlim(16, 80)
plt.show()


plt.hist(df.income, bins=100, facecolor='#3E849E', alpha=1.0)
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Income Distribution of Users")
plt.xlim(-100, 1000000)
plt.show()


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
fig1, ax1 = plt.subplots()
labels = ['Men', 'Women']
ax1.pie(df.groupby('sex_code').sex.count(), labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, colors=blues)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Gender Distribution of Users')
plt.show()





#print(df['age'].value_counts()/len(df))
#print(df['body_type'].value_counts()/len(df))
df['body_type'] = df['body_type'].map({'jacked': 0, 'athletic': 1, 'fit': 2, 'thin': 3, 'skinny': 4, 'average': 5, 'a little extra': 6, 'curvy': 7, 'full figured': 8, 'overweight': 9, 'used up': 10, 'rather not say': 11})
print(df['diet'].value_counts()/len(df))  # Many

#print(df['drinks'].value_counts()/len(df))
df['drinks'] = df['drinks'].map({'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4, 'desperately': 5})
#print(df['drugs'].value_counts()/len(df))
df['drugs'] = df['drugs'].map({'never': 0, 'sometimes': 1, 'often': 2})
print(df['education'].value_counts()/len(df))  # Many
print(df['ethnicity'].value_counts()/len(df))  # Many

print(df['job'].value_counts()/len(df))

#print(df['location'].value_counts()/len(df))  # Many
#print(df['offspring'].value_counts()/len(df))  # Many
#print(df['orientation'].value_counts()/len(df))
df['orientation'] = df['orientation'].map({'straight': 0, 'bisexual': 1, 'gay': 2})
#print(df['pets'].value_counts()/len(df))  # Many
print(df['religion'].value_counts()/len(df))  # Many
#print(df['sex'].value_counts()/len(df))
df['sex'] = df['sex'].map({'m': 0, 'f': 1})
#print(df['sign'].value_counts()/len(df))  # Many
#print(df['smokes'].value_counts()/len(df))
df['smokes'] = df['smokes'].map({'no': 0, 'trying to quit': 1, 'when drinking': 2, 'sometimes': 3, 'yes': 4})
#print(df['speaks'].value_counts()/len(df))  # Many
#print(df['status'].value_counts()/len(df))
df['status'] = df['status'].map({'single': 0, 'available': 1, 'unknown': 2, 'seeing someone': 3, 'married': 4})


plt.hist(df.age, bins=50)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
#plt.show()

#plt.pie(df['smokes'])
#plt.show()


'''
# Blues for plots
# blues = ["#66D7EB", "#51ACC5", "#3E849E", "#2C5F78", "#1C3D52", "0E1E2B"]


csv_filepath = os.path.join('Source_Data', 'profiles.csv')
df = load_data(csv_filepath)
inspect_data(df)


'''
                age        height          income
count  59946.000000  59943.000000    59946.000000
mean      32.340290     68.295281    20033.222534
std        9.452779      3.994803    97346.192104
min       18.000000      1.000000       -1.000000
25%       26.000000     66.000000       -1.000000
50%       30.000000     68.000000       -1.000000
75%       37.000000     71.000000       -1.000000
max      110.000000     95.000000  1000000.000000
'''
# A lot of entries with no reported income to cull

# Columns
'''
Index(['age', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'essay0',
       'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7',
       'essay8', 'essay9', 'ethnicity', 'height', 'income', 'job',
       'last_online', 'location', 'offspring', 'orientation', 'pets',
       'religion', 'sex', 'sign', 'smokes', 'speaks', 'status'],
      dtype='object')'''



# Fill NANs
#print(df.isna().any())
#df.fillna({'diet':0, 'weekend_checkins':0, 'average_tip_length':0, 'number_tips':0, 'average_caption_length':0, 'number_pics':0}, inplace=True)

#print(df.isna().any())


'''
fig1, ax1 = plt.subplots()
ax1.pie(pie_values, labels=pie_labels, autopct='%1.0f%%', shadow=False, startangle=90, colors=blues)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Income Distribution of Users')
'''

patches, texts = plt.pie(pie_values, labels=pie_labels, colors=blues, startangle=90)
labels = ['{0} - {1:1.1f}%'.format(i, j) for i, j in zip(pie_labels, porcent)]

sort_legend = False
if sort_legend:
    patches, labels, dummy = zip(
        *sorted(zip(patches, labels, pie_values), key=lambda pie_labels: pie_labels[2], reverse=True))

plt.legend(patches, labels, loc='left', bbox_to_anchor=(-0.1, 1.), fontsize=12)
plt.title('Income Distribution of Users')
plt.show()
