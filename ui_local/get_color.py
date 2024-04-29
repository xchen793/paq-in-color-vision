import numpy as np
from flask import jsonify 

def getRandomColor():
    starting_color = np.random.randint(256, size = 3).tolist()
    direction = np.random.random(size=3).tolist()

    data = {
        'red_direction' : direction[0],
        'green_direction' : direction[1],
        'blue_direction' : direction[2],
        'starting_color' : starting_color
    }
    return data

if __name__ == '__main__':
    data = getRandomColor()