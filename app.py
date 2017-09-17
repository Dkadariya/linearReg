from sklearn import preprocessing
import xlrd
import os 
import matplotlib.pyplot as plt
from collections import Counter
from xlrd.sheet import ctype_text  

complete_data=[]
train_data=[]
test_data=[]
prdt_risk=[]
actual_risk=[]
error_list=[]
epoc=[]
independent_variable=[]
fname = os.path.dirname(os.path.realpath(__file__))+"/fluML.xlsx"
# print (fname)

# Open the workbook
xl_workbook = xlrd.open_workbook(fname)

xl_sheet = xl_workbook.sheet_by_index(0)
print ('Sheet name: %s' % xl_sheet.name)

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

#fetching the whole dataset
for row_idx in range(1, xl_sheet.nrows):
	knowT_obj = xl_sheet.cell(row_idx, 13)
	risk_obj = xl_sheet.cell(row_idx, 9)  # Get cell object by row, col
	if(risk_obj.value !=u'' and knowT_obj.value !=u''):
		complete_data.append([knowT_obj.value,risk_obj.value])

minmax = dataset_minmax(complete_data)
print (minmax)
normalize_dataset(complete_data, minmax)

#dataset usage ratio
percentage=80;
train_data_count= (xl_sheet.nrows*percentage)/100
test_data_count=xl_sheet.nrows-train_data_count

print test_data_count
print (len(complete_data))


for row_idx in range(1, train_data_count):
	train_data.append(complete_data[row_idx])

for row_idx in range(train_data_count, len(complete_data)-1):
	test_data.append(complete_data[row_idx])
	independent_variable.append(complete_data[row_idx][0])
	actual_risk.append(complete_data[row_idx][1])

a = dict(Counter(independent_variable))
print (a)

b = dict(Counter(actual_risk))
print (b)
# print (test_data)
	# print (cell_obj.value)
# def predict(dset, cff):
# 	prdt = cff[0]
# 	for i in range(len(dset)-1):
# 		prdt += cff[i + 1] * dset[i]
# 	return prdt
 
# dataset = [[-0.77,-0.554],[-0.345,-0.554],[-0.406,-0.182],[-0.575,0.554],[0,0.554],[0.169,0.951],[0.345,0.951],[-0.169,1.393],[-0.169,-1.393],[0,0.554],[-0.345,-0.554],[1.074,-0.951],[-0.864,0.951],[0.376,-0.554],[-0.864,-0.951],[-0.77,0.554],[-0.523,1.393],[-0.169,1.393],[-0.575,-0.951],[0,1.393],[0,1.393],[0,0.951],[0,1.393],[-0.523,0.951],[-0.575,1.393],[0.169,0.951],[-0.695,1.393],[0,0.951],[-0.77,0.951],[0,-0.182],[0.345,0.554],[0,-0.182],[0.169,0.554],[0.345,0.554],[0.169,-0.182],[0.864,-0.554],[0.345,-0.951],[-0.197,-0.951],[-0.169,-1.393],[-0.169,-0.182],[0.197,-0.951],[0.181,-1.393],[0,-0.182],[-0.345,0.182],[0,-0.951],[0.197,-1.393],[0.169,-0.951],[0,-0.951],[0,-1.393],[-0.523	,-0.182],[-0.345,0.554],[0,-0.182],[0.523,-0.182],[1.074,0.182],[0.169,-1.393],[-0.231,-0.182],[0,0.951],[-0.169,1.393],[0,-1.393],[0.345,-0.951],[-1.074,-0.951],[0.169,0.951],[0.345,-0.182],[-0.575,-0.554],[0,-0.182],[-0.523,0.182],[0.575,-0.182],[0,0.182],[0.77,-0.182],[0.197	,	-0.182],[0.169	,	-0.554],[-0.345	,	-0.951],[-0.345	,	0.951],[-0.523	,	0.182],[0.77	,	0.182],[0		,-0.182],[-0.376	,	-0.182],[0.523	,	0.554],[0		,0.951],[-0.575	,	1.393],[-0.169	,	0.554],[0		,0.554],[0.345	,	-0.182],[-0.181	,	-1.393],[0.523	,	1.393],[-0.169	,	-0.182],[0.169	,	-0.182],[0.169	,	-0.554],[-0.169	,	0.554],[0.575	,	0.554],[1.453	,	-0.554],[0		,-0.951],[0.197	,	-1.393],[0.77	,	0.554],[0.169	,	0.951],[0		,-0.951],[0.406	,	-0.554],[-0.231	,	-0.554],[-0.575	,	0.182],[0		,0.554],[0.77	,	-0.951],[-0.864	,	-0.951],[0		,-0.182],[0.169	,	0.182],[-0.376	,	0.554],[-0.864	,	-0.951],[-0.345	,	0.951],[-0.169	,	-0.182],[0.169	,	-0.182],[0		,1.393],[-0.376	,	0.951],[-0.197	,	-0.554],[-0.197	,	0.554],[-1.074	,	0.951],[0.532	,	-0.182],[0.532	,	-0.182],[-0.864	,	0.182],[-0.169	,	1.393],[0.575	,	0.554],[0		,-0.182],[0.575	,	1.393],[-0.169	,	0.182],[0		,-0.951],[-1.074	,	0.951],[0.169	,	-0.182],[0		,-0.554],[-0.77	,	1.393],[0.181	,	0.182],[-0.231	,	-1.393],[-0.169	,	-0.951],[0.864	,	-0.182],[0		,-0.182],[0.77	,	-0.182],[0		,-1.393],[0.575	,	-0.951],[0.575	,	-0.951],[0.406	,	-1.393],[0		,-0.951],[-0.376	,	0.182],[0		,-0.951],[0.169	,	0.182],[0		,-0.554],[0		,-0.182],[-0.345	,	-0.182],[0.406	,	-0.951],[0.575	,	-0.951],[-0.864	,	-0.554],[-0.345	,	0.182],[0.169	,	0.951],[1.074	,	-0.554],[1.453	,	-1.393],[-1.074	,	-0.182],[-0.169	,	0.554],[-0.575	,	-1.393],[0.169	,	-0.951],[0		,1.393],[-0.169	,	-0.554],[0.169	,	-0.554],[1.074	,	0.951],[-0.575	,	0.554],[0		,0.554],[0.695	,	-0.951],[0		,0.182],[0.169	,	0.554]
# ]
# cff = [0.22998234937311363, 0.8017220304137576]
# for dset in dataset:
# 	prdt = predict(dset, cff)
# 	print("Expected=%.3f, Predicted=%.3f Error=%.3f" % (dset[-1], prdt,dset[-1]-prdt))


	# ==========================================================================================================

# Open a file
fo = open("pridcted.txt", "w")




def predict(dset, cff):
	prdt = cff[0]
	for i in range(len(dset)-1):
		prdt += cff[i + 1] * dset[i]
	return prdt
 
# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	print (coef)
	for epoch in range(n_epoch):
		sum_error = 0
		epoc.append(epoch)
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			sum_error += error**2
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
				# print (coef)
		error_list.append(sum_error/(2*train_data_count))
		# print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef
 
# Calculate coefficients
dataset = [[-0.77,-0.554],[-0.345,-0.554],[-0.406,-0.182],[-0.575,0.554],[0,0.554],[0.169,0.951],[0.345,0.951],[-0.169,1.393],[-0.169,-1.393],[0,0.554],[-0.345,-0.554],[1.074,-0.951],[-0.864,0.951],[0.376,-0.554],[-0.864,-0.951],[-0.77,0.554],[-0.523,1.393],[-0.169,1.393],[-0.575,-0.951],[0,1.393],[0,1.393],[0,0.951],[0,1.393],[-0.523,0.951],[-0.575,1.393],[0.169,0.951],[-0.695,1.393],[0,0.951],[-0.77,0.951],[0,-0.182],[0.345,0.554],[0,-0.182],[0.169,0.554],[0.345,0.554],[0.169,-0.182],[0.864,-0.554],[0.345,-0.951],[-0.197,-0.951],[-0.169,-1.393],[-0.169,-0.182],[0.197,-0.951],[0.181,-1.393],[0,-0.182],[-0.345,0.182],[0,-0.951],[0.197,-1.393],[0.169,-0.951],[0,-0.951],[0,-1.393],[-0.523	,-0.182],[-0.345,0.554],[0,-0.182],[0.523,-0.182],[1.074,0.182],[0.169,-1.393],[-0.231,-0.182],[0,0.951],[-0.169,1.393],[0,-1.393],[0.345,-0.951],[-1.074,-0.951],[0.169,0.951],[0.345,-0.182],[-0.575,-0.554],[0,-0.182],[-0.523,0.182],[0.575,-0.182],[0,0.182],[0.77,-0.182],[0.197	,	-0.182],[0.169	,	-0.554],[-0.345	,	-0.951],[-0.345	,	0.951],[-0.523	,	0.182],[0.77	,	0.182],[0		,-0.182],[-0.376	,	-0.182],[0.523	,	0.554],[0		,0.951],[-0.575	,	1.393],[-0.169	,	0.554],[0		,0.554],[0.345	,	-0.182],[-0.181	,	-1.393],[0.523	,	1.393],[-0.169	,	-0.182],[0.169	,	-0.182],[0.169	,	-0.554],[-0.169	,	0.554],[0.575	,	0.554],[1.453	,	-0.554],[0		,-0.951],[0.197	,	-1.393],[0.77	,	0.554],[0.169	,	0.951],[0		,-0.951],[0.406	,	-0.554],[-0.231	,	-0.554],[-0.575	,	0.182],[0		,0.554],[0.77	,	-0.951],[-0.864	,	-0.951],[0		,-0.182],[0.169	,	0.182],[-0.376	,	0.554],[-0.864	,	-0.951],[-0.345	,	0.951],[-0.169	,	-0.182],[0.169	,	-0.182],[0		,1.393],[-0.376	,	0.951],[-0.197	,	-0.554],[-0.197	,	0.554],[-1.074	,	0.951],[0.532	,	-0.182],[0.532	,	-0.182],[-0.864	,	0.182],[-0.169	,	1.393],[0.575	,	0.554],[0		,-0.182],[0.575	,	1.393],[-0.169	,	0.182],[0		,-0.951],[-1.074	,	0.951],[0.169	,	-0.182],[0		,-0.554],[-0.77	,	1.393],[0.181	,	0.182],[-0.231	,	-1.393],[-0.169	,	-0.951],[0.864	,	-0.182],[0		,-0.182],[0.77	,	-0.182],[0		,-1.393],[0.575	,	-0.951],[0.575	,	-0.951],[0.406	,	-1.393],[0		,-0.951],[-0.376	,	0.182],[0		,-0.951],[0.169	,	0.182],[0		,-0.554],[0		,-0.182],[-0.345	,	-0.182],[0.406	,	-0.951],[0.575	,	-0.951],[-0.864	,	-0.554],[-0.345	,	0.182],[0.169	,	0.951],[1.074	,	-0.554],[1.453	,	-1.393],[-1.074	,	-0.182],[-0.169	,	0.554],[-0.575	,	-1.393],[0.169	,	-0.951],[0		,1.393],[-0.169	,	-0.554],[0.169	,	-0.554],[1.074	,	0.951],[-0.575	,	0.554],[0		,0.554],[0.695	,	-0.951],[0		,0.182],[0.169	,	0.554]
]
# normalize the data attributes
# norm_D = preprocessing.normalize(dataset)
norm_D = train_data
# print (norm_D)
l_rate = 0.0001
n_epoch = 70
cff = coefficients_sgd(norm_D, l_rate, n_epoch)
#print(cff)



for data in test_data:
	prdt = predict(data, cff)
	prdt_risk.append(prdt)
	fo.write(str(prdt));
	fo.write("\n");
	# print("Expected=%.3f, Predicted=%.3f Error=%.3f" % (data[-1], prdt,data[-1]-prdt))
p = dict(Counter(prdt_risk))
print (p)
# print (error_list)
# print (epoc)
# print (prdt_risk)
# plt.plot(epoc,error_list, 'g^')
# plt.axis([0, 120, 0, 0.25])
plt.plot(independent_variable,actual_risk, 'g^',independent_variable,prdt_risk, 'r--')
plt.axis([-2, 2, -2, 2])
plt.show()

# Close opend file
fo.close()