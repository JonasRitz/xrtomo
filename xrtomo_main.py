import numpy as np
import matplotlib.pyplot as plt

eps = np.finfo(np.float32).eps
unit_square = np.array([[-0.5, 0.5], [-0.5, 0.5]])
ng = 8
xs = np.linspace(unit_square[0,0]+1/(2*ng), unit_square[0,1]-1/(2*ng), ng)
ys = np.linspace(unit_square[1,0]+1/(2*ng), unit_square[1,1]-1/(2*ng), ng)
x_y_d = [] #coordinates of pixel middle and density

for x in xs:
    for y in ys:
        x_y_d.append(np.array([x, y, 0]))

x_y_d = np.array(x_y_d)
cell_width = 1/ng
cell_height = 1/ng
ns = 10 # Anzahl der Strahlen
nw = 10 # Anzahl der Drehwinkel


def get_cut_pixel(startpkt, endpkt, x_y_d): # gibt die pixel an, die von der Gerade geschnitten werden
    cut_pixel = []
    for pixel in x_y_d:
        rectangle = np.array([[pixel[0] - cell_width / 2, pixel[1] - cell_height / 2],
                              [pixel[0] - cell_width / 2, pixel[1] + cell_height / 2],
                              [pixel[0] + cell_width / 2, pixel[1] + cell_height / 2],
                              [pixel[0] + cell_width / 2, pixel[1] - cell_height / 2]])
        schnittpunkte = []
        min_x = rectangle[0][0]-eps
        max_x = rectangle[2][0]+eps
        min_y = rectangle[0][1]-eps
        max_y = rectangle[1][1]+eps
        for i in range(4):
            tmp = seg_intersect(rectangle[i], rectangle[(i+1)%4], startpkt, endpkt)
            tmp_sp = [list(item) for item in schnittpunkte]
            if type(tmp) != type(None) and min_x <= tmp[0] <= max_x and min_y <= tmp[1] <= max_y and list(tmp) not in tmp_sp:
                schnittpunkte.append(tmp)
        if len(schnittpunkte) >= 2:
            len_of_cut = np.linalg.norm(schnittpunkte[0]-schnittpunkte[1])
            if len_of_cut > eps:
                cut_pixel.append((pixel, len_of_cut))
    return cut_pixel

def seg_intersect(a1, a2, b1, b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    eps = 0.00000000000000001
    if np.abs(denom.astype(float)) < eps:
        return None
    else:
        return (num / denom.astype(float))*db + b1

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def plot(startpunkt, endpunkt):
    plt.plot([-0.5, -0.5, 0.5, 0.5, -0.5], [-0.5, 0.5, 0.5, -0.5, -0.5], 'black')
    plt.plot(x_y_d[:, 0], x_y_d[:, 1], 'o')
    plt.plot([startpunkt[0], endpunkt[0]], [startpunkt[1], endpunkt[1]])

def gitter_plot():
    xs = np.linspace(unit_square[0, 0], unit_square[0, 1], ng+1)
    ys = np.linspace(unit_square[1, 0], unit_square[1, 1], ng+1)

    for x in xs:
        plt.plot([x, x], [unit_square[1,0], unit_square[1,1]], "black")
    for y in ys:
        plt.plot([unit_square[0,0], unit_square[0,1]], [y,y], "black")

def get_rot_mat(deg):
    theta = np.radians(-deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

if __name__ == "__main__":
    gitter_plot()
    startpunkt = np.array([-1, 0])
    endpunkt = np.array([0.5, 1.6])
    verbindung = endpunkt-startpunkt
    gesamtwinkel = 100
    teilwinkel = gesamtwinkel/nw
    geraden = []

    for i in range(nw):
        verbindung = np.matmul(get_rot_mat(teilwinkel), verbindung)
        geraden.append([startpunkt, endpunkt])
        endpunkt = startpunkt + verbindung

    for gerade in geraden:
        plot(gerade[0], gerade[1])
        cut_pixel = get_cut_pixel(gerade[0], gerade[1], x_y_d)
        print("Die Gerade:" , gerade)
        print("schneidet" , len(cut_pixel), "Punkte im Gitter")
        for elem in cut_pixel:
            plt.plot([elem[0][0]], [elem[0][1]], 'or')
    plt.show()


