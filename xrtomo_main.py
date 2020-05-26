import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams["figure.figsize"] = (10,10)
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))
eps = np.finfo(np.float64).eps
unit_square = np.array([[-0.5, 0.5], [-0.5, 0.5]])
ng = 10  # Anzahl der Pixel pro Achse
ns = 10  # Anzahl der Strahlen
nw = 20  # Anzahl der Drehwinkel
angle = 360


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

def perp(a) :
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

def gen_matrix():
    startpunkt = np.array([-1, 0])
    endpunkt = np.array([0.0, 0.0])
    mat = np.zeros((ns*nw,ng*ng), dtype='float64')
    strahlidx = 0
    geraden = []
    step = angle/nw
    winkel_strahlen = 30
    teilwinkel = winkel_strahlen/ns
    mittelstrahl = endpunkt - startpunkt
    for i in range(nw):
        for i in range(-ns//2, ns//2):
            aktstrahl = np.matmul(get_rot_mat(i*teilwinkel), mittelstrahl)
            aktstrahl = (aktstrahl/np.linalg.norm(aktstrahl)) * 1.7
            endpunkt_tmp = startpunkt + aktstrahl
            geraden.append([startpunkt, endpunkt_tmp])
        for gerade in geraden:
            plot(gerade[0], gerade[1])
            cut_pixel = get_cut_pixel(gerade[0], gerade[1], x_y_d)
            for elem in cut_pixel:
                mat[strahlidx, elem[2]] = elem[1]
                plt.plot([elem[0][0]], [elem[0][1]], 'or')
            strahlidx +=1
        mittelstrahl = np.matmul(get_rot_mat(step), mittelstrahl)
        startpunkt = endpunkt - mittelstrahl
        geraden = []
    plt.show()
    return mat

def get_dichte_homogen():
    dichte = np.ones(ng * ng, dtype='float64')
    return dichte

def get_dichte_inhomogen():
    return np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                     1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                     1, 1, 1, 1, 1, 0, 0, 0, 0, 1])

def TSVD(A, alpha):
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    plt.plot(np.log(s))
    plt.show()
    #for idx, sv in enumerate(s):
    #    if sv < alpha:
    #        s[idx] = 0
    #for idx, sv in enumerate(s):
    #    if s[idx] != 0:
    #        s[idx] = 1/s[idx]
    #smat = np.zeros((ng**2, ns * nw))
    #smat[:ns  * nw, :ng**2] = np.diag(s)
    #tmp = np.matmul(Vt.T, smat)
    #Aplus = np.matmul(tmp, U.T)
    #return Aplus

def Tikhonov(A, alpha, y):
    AtA = np.matmul(A.T, A)
    alphaI = alpha*np.identity(len(AtA))
    ges = AtA + alphaI
    rs = np.matmul(A.T,y)
    x_alpha = np.linalg.solve(ges, rs)
    return x_alpha

def solve_backwards(mat, sinogramm, alpha):
    dichte = Tikhonov(mat, alpha, sinogramm)
    return dichte

def plot_dichte(dichte):
    size = 0.1
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    for i in range(ng):
        for j in range(ng):
            vec = np.array([dichte[i*ng+j], dichte[i*ng+j],dichte[i*ng+j]])
            min = np.min(vec)
            max = np.max(vec)
            if max != min:
                vec = (vec -min )/(max-min)
            elif max < 0:
                vec = np.array([0,0,0])
            elif max > 1:
                vec = np.array([1,1,1])
            print(vec)
            color =  (vec[0], vec[1], vec[2])
            ax2.add_patch(Rectangle((size*i, size*j), size, size, fill=True, color=color))
    plt.show()

if __name__ == "__main__":
    gitter_plot()
    mat = gen_matrix()
    TSVD(mat, 0)
    dichte_original = get_dichte_inhomogen()
    plot_dichte(dichte_original)
    sinogramm = np.matmul(mat, dichte_original)
    sigma_quad = 0.0001
    stoerung = np.sqrt(sigma_quad) * np.random.randn(len(sinogramm))
    sinogramm_gestoert = sinogramm + stoerung
    alpha = np.sqrt(sigma_quad/len(sinogramm))
    dichte_berechnet = solve_backwards(mat, sinogramm_gestoert, alpha)
    #dichte_berechnet_alpha_0 = solve_backwards(mat, sinogramm_gestoert, 1e-16)
    plot_dichte(dichte_berechnet)
    print("Fehler:", np.linalg.norm(dichte_original - dichte_berechnet))
    print("Rel. fehler:", np.linalg.norm(dichte_original - dichte_berechnet) / np.linalg.norm(dichte_berechnet))
    plt.plot(dichte_original, 'b')
    plt.plot(dichte_berechnet, 'r')
    #plt.plot(dichte_berechnet_alpha_0, 'g')
    plt.show()
