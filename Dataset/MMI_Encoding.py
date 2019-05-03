import pandas as pd
import numpy as np
import csv

fileNew = 'MMI_OHE.csv'
with open('mmi-combined2.csv', 'r') as csv_read_file:

    csv_reader = csv.reader(csv_read_file, delimiter=',')
    newrow = []
    #print(row, "\n\n\n")
    #newrow.append(int(row[0]))
    for i in range(0, 47):
        newrow.append(0)
    for row in csv_reader:
        #print(row)
        for i in range(1, len(row)):
            try:
                newrow[int(row[i])] = int(row[i])
            except:
                pass
    #print(newrow)
    colTitles = []
    colTitles.append('Emotion')
    for i in range(0,len(newrow)):
        if int(newrow[i])!=0:
            colTitles.append(str(newrow[i]))
    print (colTitles)


with open('mmi-combined2.csv', 'r') as csv_read_file:

    with open(fileNew, 'w') as csvfile:

        writer = csv.writer(csvfile, lineterminator='\n', delimiter=',', quotechar='"')
        csv_reader = csv.reader(csv_read_file, delimiter=',')
        writer.writerow(colTitles)
        print(colTitles)
        for row in csv_reader:
            newrow = []
            newrow.append(int(row[0]))
            for i in range(0, len(colTitles) - 1):
                newrow.append(0)
            for i in range(1, len(row)):
                try:
                    for j in range(1, len(colTitles)):
                        if int(colTitles[j]) == int(row[i]):
                            newrow[j] = 1
                except:
                    pass
            print(newrow)
            writer.writerow(newrow)



