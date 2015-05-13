import math


def vector_from_angle(alpha, beta, mag=1.0):
    """
    Return vector given alpha and beta in degrees based on ESDU definition
    """
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    vec = [0.0, 0.0, 0.0]
    vec[0] = mag*math.cos(alpha)*math.cos(beta)
    vec[1] = mag*math.sin(beta)
    vec[2] = mag*math.sin(alpha)*math.cos(beta)
    return vec


def angle_from_vector(vec):
    """
    Return vector given alpha and beta in degrees based on ESDU definition
    """
    mag = math.sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2])

    beta = math.asin(vec[1]/mag)
    alpha = math.acos(vec[0]/(mag*math.cos(beta)))
    alpha = math.degrees(alpha)
    beta = math.degrees(beta)
    return (alpha, beta)


def rotate_vector(vec, alpha_degree, beta_degree):
    """
    Rotate vector by alpha and beta based on ESDU definition
    """
    alpha = math.radians(alpha_degree)
    beta = math.radians(beta_degree)
    rot = [0.0, 0.0, 0.0]

    rot[0] = math.cos(alpha)*math.cos(beta)*vec[0] + \
        math.sin(beta)*vec[1] + \
        math.sin(alpha)*math.cos(beta)*vec[2]
    rot[1] = -math.cos(alpha)*math.sin(beta)*vec[0] + \
        math.cos(beta)*vec[1] - \
        math.sin(alpha)*math.sin(beta)*vec[2]
    rot[2] = -math.sin(alpha)*vec[0] + \
        math.cos(alpha)*vec[2]
    return rot


def feet_to_meters(val):
    return val*0.3048


def pressure_from_alt(alt):
    """
    Calculate pressure in Pa from altitude in m
    using standard atmospheric tables
    """
    return 101325.0 * math.pow((1.0-2.25577e-5*alt), 5.25588)


def to_kelvin(rankine):
    return rankine*0.555555555


def non_dim_time(dim_time):
    speed = 0.2*math.sqrt(1.4*287.0*277.77)
    non_dim_speed = 0.2*math.sqrt(0.2)
    return dim_time*speed/non_dim_speed


def dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]


def mag(vec):
    return math.sqrt(dot(vec, vec))
