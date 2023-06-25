"""

All models are variations of
y = ReLU(W_T dot W dot x + B)

loss(r) = r_weighted MSE
    = Sum((y_pred - y_actual) ** 2 + r(last_y_pred - last_y_actual) ** 2

x is from Uniform random distribution(0,1)
and doesn't include 0


"""


def unweighted_sparse_one_element_expected_loss(
    sparsity: float
):
    """
    r = last_weight
    s = sparsity

    y = ReLU(Z_first * x + s/2 * e1)

    Z_first = [[0, 0]
               [0, I_n-1]]

    I_n-1 = identity  [(10)  with n-1 elements 
                       (01)] 
 
    e1 = basis vector of 1st element
       = [1 0, ...]

    s = probability x1 = 0
    1-s = probability x1 from Uniform (0, 1)

    Z_first * x = 1t = [0, x_n-1]

    x is from Uniform random distribution (0,1)
    and doesn't include 0

    y = [s/2, x_n-1]

    loss(r) = r_weighted MSE = 
      (first_y_pred - first_y_actual) ** 2 
    + Sum((y_pred - y_actual) ** 2 
    + r(last_y_pred - last_y_actual) ** 2

    loss = (s/2 - x) ** 2  

    expected_loss = 
    if 1-s = (1-s)(s/2 - 0) ** 2 = (1-s)(s/2)**2
        s  =  s Integrate(1 to 0)  (s/2 - x) **2 dx 

    = s/3 - s**2/4 
    """

    return sparsity / 3 - sparsity * sparsity /4



def weighted_sparse_one_element_expected_loss(
    last_weight: float,
    sparsity: float
):
    """
    r = last_weight
    s = sparsity

    y = ReLU(Z_last * x + s/2 * e_n-1)

    Z_first = [[I_n-1, 0]
               [0, 0]]

    I_n-1 = identity  [(10)  with n-1 elements 
                       (01)] 
 
    e_n-1= basis vector of 1st element
       = [0, ..., 1]

    s = probability x1 = 0
    1-s = probability x1 from Uniform (0, 1)

    Z_first * x = 1t = [x_n-1, 0]
    y = [x_n-1, s/2]

    loss(r) = r_weighted MSE = 
      (first_y_pred - first_y_actual) ** 2 
    + Sum((y_pred - y_actual) ** 2 
    + r(last_y_pred - last_y_actual) ** 2

    loss = r (s/2 - x) ** 2  

    expected_loss = r * first_element_zero_expected_loss(r,s)

    if 1-s = r(1-s)(s/2 - 0) ** 2 = (1-s)(s/2)**2
    ir    s  =  rs Integrate(1 to 0)  (s/2 - x) **2 dx 
    = r(s/3 - s**2/4)
    """

    return last_weight * (sparsity/3 - sparsity * sparsity/4)


def unweighted_sparse_two_element_superposition_expected_loss(
    sparsity: float
):
    """
    r = last_weight
    s = sparsity

    y = ReLU(super_first * x )

    dipole superposition
    super_2t = [[1, -1]
                [-1, 1]]

    super_first = [[super_2t,    0  ],
                        0   ,  I_n-2]]

    I_n-2 = identity  [(10)  with n-2 elements 
                       (01)] 
 
    s = probability x1 = 0
    1-s = probability x1 from Uniform (0, 1)

    y = super_first * x = 1t = [x1-x2, x2-x1, x_n-2]

    loss(r) = r_weighted MSE = 
      (first_y_pred - first_y_actual) ** 2 
    + Sum((y_pred - y_actual) ** 2 
    + r(last_y_pred - last_y_actual) ** 2

    loss = (max(x1-x2, 0) - x1) ** 2  
         + (max(x2-x1, 0) - x2) ** 2

         = (-min(x2, x1)) ** 2    
         + (-min(x1, x2)) ** 2

         = 2(min(x1, x2)) ** 2

    expected_loss = 

    if x1 or x2 is zero   = 2(min(x1,x2)) ** 2 = 0
    if x1 and x2 not zero  
     = s * s * 2 Integrate(1,0)(1,0) (min(x1,x2))**2 dx1 dx2
    
        if x2 > x1 = Integrate(1,0) (Integrate(x2,0) (x1)**2 dx1 dx2)
        if x1 > x2 = Integrate(1,0) (Integrate(x1,0) (x2)**2 dx2 dx1)
        = change dx2 dx1 to dx1 dx2
        = case x2 > x1 same integral form with case x1 > x2

     = 2 s ** 2 Integrate(1,0) (x2,0) x1 ** 2 dx1 dx2
     = s ** 2 / 3

    """
    return sparsity * sparsity / 3


def weighted_sparse_two_element_superposition_expected_loss(
    last_weight: float,
    sparsity: float
):
    """
    r = last_weight
    s = sparsity

    y = ReLU(super_last * x )

    dipole superposition
    super_2t = [[1, -1]
                [-1, 1]]

    super_last = [[I_n-2,     0   ],
                      0  , super_2t]]

    I_n-2 = identity  [(10)  with n-2 elements 
                       (01)] 
 
    s = probability x1 = 0
    1-s = probability x1 from Uniform (0, 1)

    y = super_last * x = 1t = [x_n-2, x_n-1-x_n, x_n-1-x_n]

    loss(r) = r_weighted MSE = 
      (first_y_pred - first_y_actual) ** 2 
    + Sum((y_pred - y_actual) ** 2 
    + r(last_y_pred - last_y_actual) ** 2

    loss = (max(x1-x2, 0) - x1) ** 2  
         + r (max(x2-x1, 0) - x2) ** 2

         = (-min(x2, x1)) ** 2    
         + r(-min(x1, x2)) ** 2

         = (1+r)(min(x1, x2)) ** 2

    expected_loss = 

    if x1 or x2 is zero   = (1+r)(min(x1,x2)) ** 2 = 0
    if x1 and x2 not zero  
     = s * s * (1+r) Integrate(1,0)(1,0) (min(x1,x2))**2 dx1 dx2
    
        if x2 > x1 = Integrate(1,0) (Integrate(x2,0) (x1)**2 dx1 dx2)
        if x1 > x2 = Integrate(1,0) (Integrate(x1,0) (x2)**2 dx2 dx1)
        = change dx2 dx1 to dx1 dx2
        = case x2 > x1 same integral form with case x1 > x2
     = (1+r) s ** 2 Integrate(1,0) (x2,0) x1 ** 2 dx1 dx2
     = (1+r) s ** 2 / 6 

    """
    return (1 + last_weight) * sparsity * sparsity / 6

