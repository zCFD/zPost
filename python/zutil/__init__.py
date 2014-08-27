import math
import sys

def angle_from_vector(vec):
    """
    Return vector given alpha and beta in degrees based on ESDU definition
    """
    mag = math.sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2])
    
    beta = math.asin(vec[1]/mag)
    alpha = math.acos(vec[0]/(mag*math.cos(beta))) 
    alpha = math.degrees(alpha)
    beta = math.degrees(beta)
    return (alpha,beta)

def rotate_vector(vec,alpha_degree,beta_degree):
    """
    Rotate vector by alpha and beta based on ESDU definition
    """
    alpha = math.radians(alpha_degree)
    beta = math.radians(beta_degree)
    rot = [0.0,0.0,0.0]
    rot[0] =  math.cos(alpha)*math.cos(beta)*vec[0] + math.sin(beta)*vec[1]  + math.sin(alpha)*math.cos(beta)*vec[2]
    rot[1] = -math.cos(alpha)*math.sin(beta)*vec[0] + math.cos(beta)*vec[1]  - math.sin(alpha)*math.sin(beta)*vec[2]
    rot[2] = -math.sin(alpha)*               vec[0]                          + math.cos(alpha)*               vec[2]
    return rot

def dot(vec1,vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

def mag(vec):
    return math.sqrt(dot(vec,vec))

