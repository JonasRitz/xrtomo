import numpy as np
import matplotlib.pyplot as plt

unit_square = np.array([[-0.5, 0.5], [-0.5, 0.5]])
ng = 10
xs = np.linspace(unit_square[0,0]+1/(2*ng), unit_square[0,1]-1/(2*ng), ng)
ys = np.linspace(unit_square[1,0]+1/(2*ng), unit_square[1,1]-1/(2*ng), ng)
x_y_d = []

for x in xs:
    for y in ys:
        x_y_d.append(np.array([x, y, 0]))

x_y_d = np.array(x_y_d)
cell_width = 1/ng
cell_height = 1/ng
ns = 10 # Anzahl der Strahlen
nw = 10 # Anzahl der Drehwinkel


def get_cut_pixel(startpkt, endpunkt, x_y_d): # gibt die pixel an, die von der Gerade geschnitten werden
    cut_pixel = []
    for pixel in x_y_d:
        rectangle = np.array([[pixel[0] - cell_width / 2, pixel[1] - cell_height / 2],
                              [pixel[0] - cell_width / 2, pixel[1] + cell_height / 2],
                              [pixel[0] + cell_width / 2, pixel[1] + cell_height / 2],
                              [pixel[0] + cell_width / 2, pixel[1] - cell_height / 2]])
        if( line_intersect(startpkt, endpunkt, rectangle[0], rectangle[1]) \
               or line_intersect(startpkt, endpunkt, rectangle[1], rectangle[2]) \
               or line_intersect(startpkt, endpunkt, rectangle[2], rectangle[3]) \
               or line_intersect(startpkt, endpunkt, rectangle[3], rectangle[0])):
            cut_pixel.append(pixel)
    return cut_pixel

def line_intersect(start_1, ende_1, start_2, ende_2):
    q = (start_1[1] - start_2[1]) * (ende_2[0]-start_2[0]) - (start_1[0]-start_2[0])  * (ende_2[1]-start_2[1])
    d = (ende_1[0] - start_1[0]) * (ende_2[1]-start_2[1]) - (ende_1[1]-start_1[1])  * (ende_2[0]-start_2[0])
    if (d == 0):
        return False
    r = q / d
    q = (start_1[1] - start_2[1]) * (ende_1[0] - start_1[0]) - (start_1[0] - start_2[0]) * (ende_1[1] - start_1[1])
    s = q / d
    if (r < 0 or r > 1 or s < 0 or s > 1):
        return False
    return True

def plot(startpunkt, endpunkt):
    plt.plot([-0.5, -0.5, 0.5, 0.5, -0.5], [-0.5, 0.5, 0.5, -0.5, -0.5], 'black')
    gitter_plot()
    plt.plot(x_y_d[:, 0], x_y_d[:, 1], 'o')
    plt.plot([startpunkt[0], endpunkt[0]], [startpunkt[1], endpunkt[1]])

def gitter_plot():
    xs = np.linspace(unit_square[0, 0], unit_square[0, 1], ng+1)
    ys = np.linspace(unit_square[1, 0], unit_square[1, 1], ng+1)

    for x in xs:
        plt.plot([x, x], [unit_square[1,0], unit_square[1,1]], "black")
    for y in ys:
        plt.plot([unit_square[0,0], unit_square[0,1]], [y,y], "black")


if __name__ == "__main__":
    startpunkt = np.array([-0.5, -0.5])
    endpunkt = np.array([0.5, 0.5])
    plot(startpunkt, endpunkt)
    cut_pixel = get_cut_pixel(startpunkt, endpunkt, x_y_d)
    for elem in cut_pixel:
        plt.plot([elem[0]], [elem[1]], 'or')
    plt.show()


