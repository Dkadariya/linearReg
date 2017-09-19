from sklearn import preprocessing
import xlrd
import os 
import matplotlib.pyplot as plt
from collections import Counter
from xlrd.sheet import ctype_text  
from math import sqrt

complete_data=[]
c_data=[]
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


for row_idx in range(1, len(complete_data)-1):
	c_data.append(complete_data[row_idx])
	independent_variable.append(complete_data[row_idx][0])
	actual_risk.append(complete_data[row_idx][1])




# print (error_list)
# print (independent_variable)
# print
# print
# print (prdt_risk)
# print (error_list)
# plt.plot(epoc,error_list, 'g^')
# plt.axis([0, 950, 0, 0.25])
plt.plot(independent_variable,actual_risk, 'g^')
plt.show()

# Close opend file
fo.close()