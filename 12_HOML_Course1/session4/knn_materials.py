import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'height' : [60, 71, 66, 69, 56, 68, 63, 68, 65, 66],
    'age' : [45, 26, 30, 34, 40, 36, 19, 28, 23, 32],
    'weight' : [77, 47, 55, 59, 72, 60, 40, 60, 45, 58],
}

example_data = pd.DataFrame(data)

def example_plot(arrows=False, radius=False, r_scale=7):
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    new_point = (38, 65)
    nx, ny = new_point
    
    colors = ['b', 'r', 'g', 'g', 'b', 'g', 'r', 'g', 'r', 'g']
    
    for label, x, y in zip(range(10), example_data['age'], example_data['height']):
        plt.annotate(
            label,
            xy=(x, y), xytext=(0, -20),
            textcoords='offset points')
        
        if arrows:
            plt.arrow(nx, ny, x - nx, y - ny, length_includes_head=True, head_width=0.5)

    if radius:
        xs = r_scale * np.cos(np.arange(0, 2*np.pi, 0.1)) + nx + 2
        ys = r_scale * np.sin(np.arange(0, 2*np.pi, 0.1)) + ny - 2.5
        
        plt.plot(xs, ys)
    
    ax.scatter(example_data['age'], example_data['height'], marker='*', c=colors)
    ax.scatter([nx], [ny], marker='*', c=['y'])
    plt.annotate(10, xy=new_point, xytext=(0, -20), textcoords='offset points')
    plt.ylim(45, 78)
    plt.xlabel('Age in years')
    plt.ylabel('Height in inches')