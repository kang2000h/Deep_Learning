import matplotlib.pyplot as plt
import numpy as np
ta = [1,2,3,4,5]
va = [1,2,1,1,1]
x = np.arange(len(ta))
y1 = np.array(ta)
y2 = np.array(va)

plt.plot(x, y1, label="Train_Acc")
plt.plot(x, y2, linestyle="--", label="Validation_Acc")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend()
plt.show()
