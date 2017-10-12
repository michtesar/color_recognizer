import numpy as np

def integerize(data):
    if np.size(data) == 1:
        if data == 'Black':
            Y = 0
        elif data == 'White':
            Y = 1
        elif data == 'Red':
            Y = 2
        elif data == 'Green':
            Y = 3
        elif data == 'Blue':
            Y = 4
        elif data == 'Orange':
            Y = 5
        elif data == 'Yellow':
            Y = 6
        elif data == 'Purple':
            Y = 7
    else:
        Y = list()
        for i in range(len(data)):
            a = data[i]
            if a == 'Black':
                Y.append(0)
            elif a == 'White':
                Y.append(1)
            elif a == 'Red':
                Y.append(2)
            elif a == 'Green':
                Y.append(3)
            elif a == 'Blue':
                Y.append(4)
            elif a == 'Orange':
                Y.append(5)
            elif a == 'Yellow':
                Y.append(6)
            elif a == 'Purple':
                Y.append(7)    
    
    return Y