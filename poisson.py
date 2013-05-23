# Solve Poisson equation on arbitrary 2D domain with RHS f 
# and Dirichlet boundary conditions.
#
# Dale Roberts <dale.o.roberts@gmail.com>

from numpy import *
from numpy.matlib import *
from numpy.linalg import *
from matplotlib.pyplot import *
from matplotlib.delaunay.triangulate import Triangulation

def mesh(xs, ys, npoints):
    # randomly choose some points
    rng = random.RandomState(1234567890)
    rx = rng.uniform(xs[0], xs[1], size=npoints)
    ry = rng.uniform(ys[0], ys[1], size=npoints)
    # only take points in domain
    nx, ny = [], []
    for x,y in zip(rx,ry):
        if in_domain(x,y):
            nx.append(x)
            ny.append(y)
    # Delaunay triangulation
    tri = Triangulation(array(nx), array(ny))
    return tri
            
def A_e(v):
    # take vertices of element and return contribution to A
    G = vstack((ones((1,3)), v.T)).I * vstack((zeros((1,2)),eye(2)))
    return det(vstack((ones((1,3)), v.T))) * G * G.T / 2

def b_e(v):
    # take vertices of element and return contribution to b
    vS = v.sum(axis=0)/3.0 # Centre of gravity
    return f(vS) * ((v[1,0]-v[0,0])*(v[2,1]-v[0,1])-(v[2,0]-v[0,0])*(v[1,1]-v[0,1])) / 6.0

def poisson(tri, boundary):
    # get elements and vertices from mesh
    elements = tri.triangle_nodes
    vertices = vstack((tri.x,tri.y)).T
    # number of vertices and elements
    N = vertices.shape[0]
    E = elements.shape[0]
    #Loop over elements and assemble LHS and RHS 
    A = zeros((N,N))
    b = zeros((N,1))
    for j in range(E):
        index = (elements[j,:]).tolist()
        A[ix_(index,index)] += A_e(vertices[index,:])
        b[index] += b_e(vertices[index,:])
    # find the 'free' vertices that we need to solve for    
    free = list(set(range(len(vertices))) - set(boundary))
    # initialise solution to zero so 'non-free' vertices are by default zero
    u = zeros((N,1))
    # solve for 'free' vertices.
    u[free] = solve(A[ix_(free,free)],b[free])
    return array(u)
    
def f(v):
    # the RHS f
    x, y = v
    f = 2.0*cos(10.0*x)*sin(10.0*y) + sin(10.0*x*y)
    return 1

def in_domain(x,y):
    # is a point in the domain?
    return sqrt(x**2 + y**2) <= 1

xs = (-1.,1.)
ys = (-1.,1.)
npoints = 1000

# generate mesh and determine boundary vertices
tri = mesh(xs, ys, npoints)
boundary = tri.hull

# solve Poisson equation
u  = poisson(tri, boundary).flatten()

# interpolate values and plot a nice image
lpi = tri.linear_interpolator(u)
z = lpi[ys[0]:ys[1]:complex(0,npoints),
        xs[0]:xs[1]:complex(0,npoints)]
z = where(isinf(z), 0.0, z)
extent = (xs[0], xs[1], ys[0], ys[1])
ioff()
clf()
imshow(nan_to_num(z), interpolation='bilinear', extent=extent, origin='lower')
