from sklearn import preprocessing
import xlrd
import os 
from xlrd.sheet import ctype_text  

risk=[]
fname = os.path.dirname(os.path.realpath(__file__))+"/fluML.xlsx"
# print (fname)

# Open the workbook
xl_workbook = xlrd.open_workbook(fname)

xl_sheet = xl_workbook.sheet_by_index(0)
print ('Sheet name: %s' % xl_sheet.name)

#dataset usage ratio
percentage=20;
test_data_count= (xl_sheet.nrows*percentage)/100
print test_data_count


for row_idx in range(1, test_data_count):
	knowT_obj = xl_sheet.cell(row_idx, 13)
	risk_obj = xl_sheet.cell(row_idx, 9)  # Get cell object by row, col
	if(risk_obj.value !=u'' and knowT_obj.value !=u''):
		risk.append([risk_obj.value,knowT_obj.value])
	# print (cell_obj.value)

print (risk)

dataset=[[-100,34],[200,65],[-150,65],[567,45],[234,-23],[-987,42],[98,-53],[-67,-34]]
normalized = preprocessing.normalize(dataset)