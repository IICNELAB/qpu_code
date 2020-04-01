import numpy as np

def random_unit_quaternion():
    q = np.random.rand(4)
    q = (q - 0.5) * 2  # [-1, 1)
    norm = np.sqrt(np.sum(np.square(q)))
    if norm == 0.:
        return random_unit_quaterion()  # try again
    else:
        q = q / norm
        return q


def normalize(v):
    """Normalize a vector along last axis.
    Args:
        v: np.array(..., d)
    Return:
        v normalized: np.array(..., d)
    """
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def q_angle_axis(theta, axis):
    axis = axis / np.linalg.norm(axis) * np.sin(theta)
    return np.concatenate([[np.cos(theta)], axis])


def q_conj(q):
    """Compute quaternion conjugation
    Args:
        q: np.array(..., 4)
    Return:
        q_conj: np.array(..., 4)
    """
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)    


def q_power(q, h):
    if np.abs(q[0]) > 1. - 1e-6:
        return q 
    theta = np.arccos(q[0])
    theta = theta * h
    q = q.copy()
    mul = np.sin(theta) / (np.sqrt(1 - q[0] * q[0]) + 1e-6)
    q[0] = np.cos(theta)
    q[1] = q[1] * mul
    q[2] = q[2] * mul
    q[3] = q[3] * mul
    return q
    

def q_geo_distance(q1, q2):
    return 1 - np.abs(np.sum(q1 * q2))


def q_angle(v1, v2):
    """Compute quaternion angle of shortest arc from vector v1 to v2.
    Args:
        v1: np.array(..., 3)
        v2: np.array(..., 3)
    Return:
        q: np.array(..., 4)
    References:
        https://bitbucket.org/sinbad/ogre/src/9db75e3ba05c/OgreMain/include/OgreVector3.h?fileviewer=file-view-default#cl-651
        https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    Formular
        q = 2|v1||v2|cos(phi)*(cos(phi) + mu*sin(phi))
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    q = np.repeat(np.zeros_like(v1[..., :1]), 4, axis=-1)
    dot = np.sum(v1 * v2, axis=-1)
    # Inverse vectors
    inv_ind = dot < (1e-6 - 1.)
    if np.sum(inv_ind) > 0:
        # Get an axis and rotate pi
        c = np.cross([1, 0, 0], v1[inv_ind])
        # If colinear, pick another axis
        col_ind = np.linalg.norm(c, axis=-1) < 1e-6  
        c[col_ind] = np.cross([0, 1, 0], v1[inv_ind][col_ind])
        q[inv_ind] = np.concatenate([np.zeros_like(c[..., :1]), c], axis=-1)

    # Normal case
    norm_ind = ~inv_ind
    if np.sum(norm_ind) > 0:
        c = np.cross(v1[norm_ind], v2[norm_ind])  # 2 * mu * sin(phi) * cos(phi)   
        s = np.sqrt((1 + dot[norm_ind]) * 2)  # 2 * |cos(phi)|
        s = np.expand_dims(s, axis=-1)
        q_xyz = c / s
        q_w = s / 2.
        q[norm_ind] = np.concatenate([q_w, q_xyz], axis=-1)

    # Handle when v1 or v2 is zero
    q = normalize(q)
    return q


def q_mul(q1, q2):
    """Compute Hamilton product between q1 and q2.
    Args:
        q1: np.array(..., 4)
        q2: np.array(..., 4)
    Return:
        q1xq2: (..., 4)
    """
    # Compute outer product
    r1, i1, j1, k1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    r2, i2, j2, k2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    r_out, i_out, j_out, k_out = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2, \
                                 r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2, \
                                 r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2, \
                                 r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
    
    return np.stack([r_out, i_out, j_out, k_out], axis=-1)


def q_rel_angle(v1, v2):
    """
    First move reference frame so that v1 = x with minimum rotation. Then compute quaternion angle from x to v2.
    Args:
        v1: np.array(..., 3)
        v2: np.array(..., 3)
    Return:
        q: np.array(..., 4)
    """
    x = np.zeros_like(v1)
    x[..., 0] = 1

    return q_mul(q_angle(v1, x), q_angle(x, v2))


def q_rotate(q, v):
    """
    Apply quaternion rotation q to v.
    Args:
        q: np.array(..., 4)
        v: np.array(..., 3)
    Return:
        v rotated: np.array(..., 3)
    """
    v = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)  # 3D vector to pure quaternion
    return q_mul(q, q_mul(v, q_conj(q)))[..., 1:]


if __name__ == '__main__':
    # Tests for q_angle
    # Normal
    v1 = np.array([0, 0.1, 0])
    v2 = np.array([0.1, 0, 0])
    print('v1:', v1, 'v2:', v2)
    print('q_angle:', q_angle(v1 ,v2), 'expected: [0.707, 0, 0, -0.707]')
    print('-')
    # Same direction
    v1 = np.array([0.1, 0, 0])
    v2 = np.array([0.2, 0, 0])
    print('v1:', v1, 'v2:', v2)
    print('q_angle:', q_angle(v1, v2), 'expected: [1, 0, 0, 0]')
    print('-')
    # Inverse 1
    v1 = np.array([0.1, 0, 0])
    v2 = np.array([-0.1, 0, 0])
    print('v1:', v1, 'v2:', v2)
    print('q_angle:', q_angle(v1, v2), 'expected: [0, 0, 0, -1]')
    print('-')
    # Inverse 2
    v1 = np.array([0, 0.1, 0])
    v2 = np.array([0, -0.1, 0])
    print('v1:', v1, 'v2:', v2)
    print('q_angle:', q_angle(v1, v2), 'expected: [0, 0, 0, 1]')
    # Batched
    v1 = np.array([[0, 0.1, 0], [0, 0.1, 0]])
    v2 = np.array([[0.1, 0, 0], [0, -0.1, 0]])
    print('v1:', v1, 'v2:', v2)
    print('q_angle:', q_angle(v1, v2), 'expected: [[0.707, 0, 0, -0.707], [0, 0, 0, 1]]')

    # Tests for q_mul
    q1 = np.array([2.5, 5, -2.1, 2])
    q2 = np.array([1, 3, -6.1, 1])
    print('q1:', q1, 'q2:', q2)
    print('q_mul:', q_mul(q1, q2), 'expected: [-27.31, 22.6, -16.35, -19.7]')
    print('-')

    # Test for q_power
    q = random_unit_quaternion()
    q2 = q_power(q, -1)
    print('q', q, 'q^-1', q2)