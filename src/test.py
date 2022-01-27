import matplotlib.pyplot as plt
import numpy as np

trainlossmlp = np.loadtxt("saves/mlp/lossfinal.txt")[:200]
vallossmlp = np.loadtxt("saves/mlp/eval_lossfinal.txt")[:200]
trainlosscnn = np.loadtxt("saves/cnn/lossfinal.txt")[:200]
vallosscnn = np.loadtxt("saves/cnn/eval_lossfinal.txt")[:200]
plt.plot(trainlossmlp,label= "MLP training loss")
plt.plot(vallossmlp,label= "MLP validation loss")
plt.plot(trainlosscnn,label= "CNN training loss")
plt.plot(vallosscnn,label= "CNN validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
ax = plt.gca()
ax.set_ylim(-0.2,4)
plt.legend()
plt.show()
