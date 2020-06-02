import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Setup:
    def __init__(self, ng, ns, nw, angle, stoerung, plot_unregularisiert):
        self.eps= np.finfo(np.float32).eps
        self.unit_square = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        self.ng = ng
        self.ns = ns
        self.nw = nw
        self.angle = angle
        self.xs = np.linspace(self.unit_square[0,0]+1/(2*ng), self.unit_square[0,1]-1/(2*ng), ng)
        self.ys = np.linspace(self.unit_square[1,0]+1/(2*ng), self.unit_square[1,1]-1/(2*ng), ng)
        self.x_y_d = []
        for x in self.xs:
            for y in self.ys:
                self.x_y_d.append(np.array([x, y, 0]))
        self.x_y_d = np.array(self.x_y_d)
        self.cell_width = 1 / ng
        self.cell_height = 1 / ng
        self.stoerung = stoerung
        self.plot_unregularisiert = plot_unregularisiert

def get_cut_pixel(startpkt, endpkt, x_y_d, setup): # gibt die pixel an, die von der Gerade geschnitten werden
    # liste aus (xkoord, ykoord, länge des schnitts, dichte des pixels)
    cut_pixel = []
    for idx, pixel in enumerate(x_y_d):
        rectangle = np.array([[pixel[0] - setup.cell_width / 2, pixel[1] - setup.cell_height / 2],
                              [pixel[0] - setup.cell_width / 2, pixel[1] + setup.cell_height / 2],
                              [pixel[0] + setup.cell_width / 2, pixel[1] + setup.cell_height / 2],
                              [pixel[0] + setup.cell_width / 2, pixel[1] - setup.cell_height / 2]])
        schnittpunkte = []
        min_x = rectangle[0][0]-setup.eps
        max_x = rectangle[2][0]+setup.eps
        min_y = rectangle[0][1]-setup.eps
        max_y = rectangle[1][1]+setup.eps
        for i in range(4):
            tmp = seg_intersect(rectangle[i], rectangle[(i+1)%4], startpkt, endpkt, setup.eps)
            tmp_sp = [list(item) for item in schnittpunkte]
            if type(tmp) != type(None) and min_x <= tmp[0] <= max_x and min_y <= tmp[1] <= max_y and list(tmp) not in tmp_sp:
                schnittpunkte.append(tmp)
        if len(schnittpunkte) >= 2:
            len_of_cut = np.linalg.norm(schnittpunkte[0]-schnittpunkte[1])
            if len_of_cut > setup.eps:
                cut_pixel.append((pixel, len_of_cut, idx))
    return cut_pixel

def seg_intersect(a1, a2, b1, b2, eps) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
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

def gitter_plot(setup):
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.plot(setup.x_y_d[:, 0], setup.x_y_d[:, 1], 'o')
    xs = np.linspace(setup.unit_square[0, 0], setup.unit_square[0, 1], setup.ng+1)
    ys = np.linspace(setup.unit_square[1, 0], setup.unit_square[1, 1], setup.ng+1)

    for x in xs:
        plt.plot([x, x], [setup.unit_square[1,0], setup.unit_square[1,1]], "black")
    for y in ys:
        plt.plot([setup.unit_square[0,0], setup.unit_square[0,1]], [y,y], "black")

def get_rot_mat(deg):
    theta = np.radians(-deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def gen_matrix(setup):
    startpunkt = np.array([-1, 0])
    endpunkt = np.array([0.0, 0.0])
    mat = np.zeros((setup.ns*setup.nw,setup.ng*setup.ng), dtype='float64')
    strahlidx = 0
    geraden = []
    step = setup.angle/setup.nw
    winkel_strahlen = 30
    teilwinkel = winkel_strahlen/setup.ns
    mittelstrahl = endpunkt - startpunkt
    for i in range(setup.nw):
        for i in range(-setup.ns//2, setup.ns//2):
            aktstrahl = np.matmul(get_rot_mat(i*teilwinkel), mittelstrahl)
            aktstrahl = (aktstrahl/np.linalg.norm(aktstrahl)) * 1.7
            endpunkt_tmp = startpunkt + aktstrahl
            geraden.append([startpunkt, endpunkt_tmp])
        for gerade in geraden:
            plot(gerade[0], gerade[1])
            cut_pixel = get_cut_pixel(gerade[0], gerade[1], setup.x_y_d, setup)
            for elem in cut_pixel:
                mat[strahlidx, elem[2]] = elem[1]
                plt.plot([elem[0][0]], [elem[0][1]], 'or')
            strahlidx +=1
        mittelstrahl = np.matmul(get_rot_mat(step), mittelstrahl)
        startpunkt = endpunkt - mittelstrahl
        geraden = []
    plt.show()
    return mat

def get_dichte_homogen(setup):
    dichte = np.ones(setup.ng * setup.ng, dtype='float64')
    return dichte

def get_dichte_inhomogen():
    return np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                     1, 3, 0, 0, 1, 0, 0, 0, 0, 1,
                     1, 0, 1, 0, 1, 0, 0, 1, 0, 2,
                     1, 0, 2, 0, 1, 0, 0, 0, 5, 3,
                     1, 0, 0, 0, 1, 0, 0, 4, 0, 1,
                     1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                     1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                     1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
                     1, 3, 0, 0, 1, 0, 0, 0, 4, 1,
                     1, 1, 1, 1, 1, 0, 0, 0, 1, 1])

def get_dichte_random(setup):
    return np.random.rand(setup.ng*setup.ng)

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

def plot_dichte(dichte, setup, range_v, titel):
    size = 0.1
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    for i in range(setup.ng):
        for j in range(setup.ng):
            vec = np.array([dichte[i*setup.ng+j], dichte[i*setup.ng+j],dichte[i*setup.ng+j]])
            min = range_v[0]
            max = range_v[1]
            vec = (vec -min )/(max-min)
            #elif max < 0:
            #    vec = np.array([0,0,0])
            #elif max > 1:
            #    vec = np.array([1,1,1])
            color =  (vec[0], vec[1], vec[2])
            ax2.add_patch(Rectangle((size*i, size*j), size, size, fill=True, color=color))
    plt.title(titel)
    plt.show()

def finde_range(o, b, a0):
    return (np.min([o,b, a0]), np.max([o,b,a0]))

def finde_range_ohne_alpha0(o, b):
    return (np.min([o,b]), np.max([o,b]))

def starte_programm(setup):
    gitter_plot(setup)
    mat = gen_matrix(setup)
    dichte_original = get_dichte_inhomogen()#get_dichte_random(setup)
    sinogramm = np.matmul(mat, dichte_original)
    sigma_quad = setup.stoerung
    stoerung = np.sqrt(sigma_quad) * np.random.randn(len(sinogramm))
    sinogramm_gestoert = sinogramm + stoerung
    alpha = np.sqrt(sigma_quad / len(sinogramm))
    dichte_berechnet = solve_backwards(mat, sinogramm_gestoert, alpha)
    if setup.plot_unregularisiert:
        dichte_berechnet_alpha_0 = solve_backwards(mat, sinogramm_gestoert, 0.0)
        range = finde_range(dichte_original, dichte_berechnet, dichte_berechnet_alpha_0)
    else:
        range = finde_range_ohne_alpha0(dichte_original, dichte_berechnet)
    plot_dichte(dichte_original, setup, range, "Original")
    plot_dichte(dichte_berechnet, setup, range, "Regularisiert")
    if setup.plot_unregularisiert:
        plot_dichte(dichte_berechnet_alpha_0, setup, range, "Unregularisiert")
    print("Fehler regularisiert:", np.linalg.norm(dichte_original - dichte_berechnet))
    print("Rel. fehler regularisiert:", np.linalg.norm(dichte_original - dichte_berechnet) / np.linalg.norm(dichte_original))
    if setup.plot_unregularisiert:
        print("Fehler NICHT regularisiert:", np.linalg.norm(dichte_original - dichte_berechnet_alpha_0))
        print("Rel. fehler NICHT regularisiert:", np.linalg.norm(dichte_original - dichte_berechnet_alpha_0) / np.linalg.norm(dichte_original))
    plt.plot(dichte_original, 'b', label="Original")
    plt.plot(dichte_berechnet, 'r', label="Regularisiert")
    if setup.plot_unregularisiert:
        plt.plot(dichte_berechnet_alpha_0, 'g', label="Unregularisiert")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #             Pixel  Strahlen  Drehwinkel  Gesamtwinkel   Stoerung
    setup = Setup(ng=10, ns=10,    nw=10,      angle=360,      stoerung=0.0001, plot_unregularisiert=False)
    starte_programm(setup)

