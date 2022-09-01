"""
Arena Builder Helper Function
Author: Thinh Nguyen

Quick Note
----------------
Use these helper functions to create arrays containing cartesian coordinates of objects
List of functions:
build_walls(), build_wall()
    --> straight wall with equal spacing for objects, from start to end
build_circles(), build_cicle()
    --> Enter center, radius, spacing --> circle --> Use this to make quick donuts
build_arc()
    --> similar to build_wall() but use arc, NOTE: input `spin` to choose direction of the arc.
    Default `spin` is 1 for counterclockwise
"""
import numpy as np


def build_walls(starts, ends, spacings):
    walls = np.array([]).reshape(0,2)
    for (start, end, spacing) in zip(starts, ends, spacings):
        walls = np.vstack((walls, build_wall(start, end, spacing)))
    return walls


def build_wall(start, end, spacing):
    x0, y0 = start
    x1, y1 = end
    alpha = np.arctan2(y1-y0, x1-x0)
    sx = spacing * np.cos(alpha)
    sy = spacing * np.sin(alpha)
    xls = np.arange(x0,x1,sx) if x0!=x1 else np.array([])
    yls = np.arange(y0,y1,sy) if y0!=y1 else np.array([])
    if len(xls) + len(yls) == 0:
        wall = start.reshape(-1,2)
        return wall
    elif len(xls) == 0 and len(yls) != 0 :
        xls = x0*np.ones(len(yls))
    elif len(yls) == 0 and len(xls) != 0 :
        yls = y0*np.ones(len(xls))
    wall = np.hstack((xls.reshape(-1,1),yls.reshape(-1,1))).reshape(-1,2)
    return wall


def build_circles(centers, radii, spacings, rotates=None, random_rotates=False):
    if rotates is None:
        rotates = 0.1*np.pi*np.randome.rand(len(centers)) if random_rotates else np.zeros(len(centers))
    circles = np.array([]).reshape(0,2)
    for (center, radius, spacing, rotate) in zip(centers, radii, spacings, rotates):
        circles = np.vstack((circles, build_circle(center, radius, spacing, rotate)))
    return circles


def build_circle(center, radius, spacing, rotate=0.):
    x0, y0 = center
    angular_spacing = calc_angular_spacing(radius, spacing)
    phi = np.arange(-np.pi, np.pi, angular_spacing)
    rho = radius*np.ones(len(phi))
    phi = wrap2pi(phi + rotate)
    x, y = pol2cart(rho,phi)
    x, y = (x+x0, y+y0)
    circle = np.hstack(( x.reshape(-1,1), y.reshape(-1,1) ))
    return circle


def build_arc(start, end, radius, spacing, spin=1):
    if type(start) is not np.ndarray: start = np.asarray(start)
    if type(end) is not np.ndarray: end = np.asarray(end)
    euclid_dist = np.linalg.norm(end-start)
    x0, y0 = start
    x1, y1 = end
    yaw = np.arctan2(y1 - y0, x1 - x0)
    midpoint = (start + end)/2
    if 2*radius < euclid_dist: radius = euclid_dist
    phi_arc = 2*np.arcsin(euclid_dist/(2*radius))
    d = np.sqrt(radius**2 - (euclid_dist/2)**2)
    center = midpoint + [d*np.cos(yaw+spin*np.pi/2), d*np.sin(yaw+spin*np.pi/2)]
    angular_spacing = 2*np.arcsin(spacing/(2*radius))
    phi_start = np.arctan2(y0-center[1], x0-center[0])
    phi = np.arange(0., spin*(phi_arc), spin*angular_spacing)
    phi = wrap2pi(phi+phi_start)
    rho = radius*np.ones(len(phi))
    x, y = pol2cart(rho,phi)
    x, y = (x+center[0], y+center[1])
    arc = np.hstack(( x.reshape(-1,1), y.reshape(-1,1) ))
    return arc
    

def calc_angular_spacing(radius, spacing):
    phi_s = 2*np.arcsin(spacing/(2*radius))
    n = np.round(2*np.pi/phi_s)
    phi_spacing = 2*np.pi/n
    return phi_spacing

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def wrap2pi(a):
    if type(a) == np.ndarray:
        a[a >=np.pi] -= 2*np.pi
        a[a <-np.pi] += 2*np.pi
    else:
        a += -2*np.pi if a >= np.pi else 2*np.pi if a <-np.pi else 0.
    return a
