import matplotlib.pyplot as plt

def visualize(samples):

    fig = plt.figure()
    for ind in range(25):

        y = fig.add_subplot(5, 5, ind+1)
        y.imshow(samples[:,:,ind])
    plt.show()

