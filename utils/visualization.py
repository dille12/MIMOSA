#
import math
import matplotlib.pyplot as plt
yVals = []
xVals = []
for i in range(700):
    x = i/100 - 4
    xVals.append(x)
    yVals.append((abs(math.sin(x))*3+1)*((abs(x)-3)*0.1+1)+4)

# Create plot
plt.plot(xVals, yVals, linestyle='-', color='b', label='y = x^2')


plt.text(-0.25, 4.712, f'Global optimum', fontsize=10, ha='right', va='bottom')
plt.text(-2.5, 4.712, f'Local optimum', fontsize=10, ha='right', va='bottom')
plt.text(2.8, 5.3, f'Local optimum', fontsize=10, ha='right', va='bottom')


plt.xlabel('Imaginary weight value')
plt.ylabel('Accuracy')
plt.show()