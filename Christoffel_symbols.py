from sympy import*

#Define the symbols
M,t,r,theta,phi,spin = symbols(" M t r theta phi spin")
#Define any shortenings
Delta = ((r**2)-(2*M*r)+(spin**2))
Sigma = ((r**2)+((spin**2)*(cos(theta)**2)))
#Define the coordinates
coor = [t,r,theta,phi]
#Define the metric
def m(i,j):
    g00 = -(1-((2*M*r)/(Sigma)))
    g01 = 0
    g02 = 0
    g03 = -((2*M*r)/(Sigma))*(spin*(sin(theta)**2))
    g10 = 0
    g11 = (Sigma)/(Delta)
    g12 = 0
    g13 = 0
    g20 = 0
    g21 = 0
    g22 = Sigma
    g23 = 0
    g30 = -((2*M*r)/(Sigma))*(spin*(sin(theta)**2))
    g31 = 0
    g32 = 0
    g33 = ((r**2)+(spin**2)+((2*M*r*(spin**2)*(sin(theta)**2))/(Sigma)))*(sin(theta)**2)
    return ([[g00,g01,g02,g03],
             [g10,g11,g12,g13],
             [g20,g21,g22,g23],
             [g30,g31,g32,g33]])[i][j]
#Define derivative of metric
def dm(i,j,k):
    return diff(m(i,j),coor[k])
#Define inverse of metric
def im(i,j):
    K = Matrix([[m(0,0),m(0,1),m(0,2),m(0,3)],
                [m(1,0),m(1,1),m(1,2),m(1,3)],
                [m(2,0),m(2,1),m(2,2),m(2,3)],
                [m(3,0),m(3,1),m(3,2),m(3,3)]])
    return K.inv(method="LU")[i,j]
#Define christoffel symbol
def gamma(i,j,k):
    s = 0
    for l in range(4):
        s+=0.5*im(i,l)*(dm(k,l,j)+dm(l,j,k)-dm(j,k,l))
    return simplify(s)
#Print non zero elements
for a in range(4):
    for b in range(4):
        for c in range(4):
            if gamma(a,b,c)==0:
                pass
            else:
                print("[",a,b,c,"]",gamma(a,b,c))