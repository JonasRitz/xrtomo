import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))

eps = np.finfo(np.float32).eps
unit_square = np.array([[-0.5, 0.5], [-0.5, 0.5]])
ng = 10
xs = np.linspace(unit_square[0,0]+1/(2*ng), unit_square[0,1]-1/(2*ng), ng)
ys = np.linspace(unit_square[1,0]+1/(2*ng), unit_square[1,1]-1/(2*ng), ng)
x_y_d = [] #coordinates of pixel middle and density

for x in xs:
    for y in ys:
        x_y_d.append(np.array([x, y, 0]))

x_y_d = np.array(x_y_d)
cell_width = 1/ng
cell_height = 1/ng



def get_cut_pixel(startpkt, endpkt, x_y_d): # gibt die pixel an, die von der Gerade geschnitten werden
    # liste aus (xkoord, ykoord, länge des schnitts, dichte des pixels)
    cut_pixel = []
    for idx, pixel in enumerate(x_y_d):
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
                cut_pixel.append((pixel, len_of_cut, idx))
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
    plt.plot([startpunkt[0], endpunkt[0]], [startpunkt[1], endpunkt[1]], 'r')

def gitter_plot():
    plt.plot(x_y_d[:, 0], x_y_d[:, 1], 'o')
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

def Pseudo(A):
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    #print(U.shape)
    #print(s.shape)
    #print(Vt.shape)
    eps_2 = 10**(-5)
    for idx, sv in enumerate(s):
        if sv/s[0] < eps_2:
            s[idx] = 0
    for idx, sv in enumerate(s):
        if s[idx] != 0:
            s[idx] = 1/s[idx]

    smat = np.zeros((100, 112))
    smat[:100, :100] = np.diag(s)
    tmp = np.matmul(Vt.T, smat)
    Aplus = np.matmul(tmp, U.T)
    #return np.linalg.pinv(A)
    return Aplus

if __name__ == "__main__":
    startpunkt = np.array([-1, 0])
    endpunkt = np.array([0.5, 1.6])
    verbindung = endpunkt-startpunkt
    gitter_plot()
    radius = 1
    winkel_strahlen = 30
    winkel_ink = 10
    ns = 16 #Anzahl der Strahlen
    nw = 7 # Anzahl der Drehwinkel
    teilwinkel = winkel_strahlen/ns
    geraden = []

    mat = np.zeros((ns*nw,ng*ng))
    dichten = np.ones(ng*ng)
    strahlidx = 0
    for i in range(nw):
        erster_strahl = endpunkt-startpunkt
        for i in range(ns):
            verbindung = np.matmul(get_rot_mat(teilwinkel), verbindung)
            geraden.append([startpunkt, endpunkt])
            endpunkt = startpunkt + verbindung
        for gerade in geraden:
            plot(gerade[0], gerade[1])
            cut_pixel = get_cut_pixel(gerade[0], gerade[1], x_y_d)
            for elem in cut_pixel:
                mat[strahlidx, elem[2]] = elem[1]
                plt.plot([elem[0][0]], [elem[0][1]], 'or')
            strahlidx +=1
        verbindung = np.matmul(get_rot_mat(winkel_ink), erster_strahl)
        endpunkt = startpunkt + verbindung
        geraden = []
    plt.show()
    #print(mat)
    print("Kondition:", np.linalg.cond(mat))
    sinogramm = np.matmul(mat, dichten)
    #pinv = np.linalg.pinv(mat)
    pinv = Pseudo(mat)
    dichte = np.matmul(pinv, sinogramm)
    print("Dichte")
    print(dichte)
    print("fehler:", np.linalg.norm(dichte-dichten))
    print("Rel. fehler:", np.linalg.norm(dichte - dichten)/np.linalg.norm(dichten))

    #print(sinogramm)

