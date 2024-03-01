import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class DH:
    def __init__(self, n_dof, name="", ee_mask=None):
        """
        Create a table for all DH table params
        @type  n_dof: int
        @param n_dof: manipualator degrees of freedom
        """
        self.n_dof = n_dof
        self.dh_params = np.empty([n_dof, 5])
        self.param = {'THETA': 0, 'D': 1, 'A': 2, 'ALPHA': 3, 'IS_ANGULAR': 4}
        self.fig_index = 0
        self.limits = np.zeros((n_dof, 2))
        self.name = name
        self.ee_mask = ee_mask
        self.ee_dof = np.count_nonzero(ee_mask)

    def plot(self, save_fig=False):
      """
      Plot using matplotlib a 3D scheme of the robot
      When using Jupyter notebook or Conda uncomment %matplotlib notebook line to interat with the plots.
      """
      x = [0]
      y = [0]
      z = [0]
      tf = np.identity(4)
      for i in range(self.n_dof):
          tf = np.dot(tf, self.get_tf_dof(i))
          x.append(tf[0, 3])
          y.append(tf[1, 3])
          z.append(tf[2, 3])

      ax = plt.axes(projection='3d')
      ax.plot3D(x, y, z)
      ax.plot3D(x, y, z, '*')

      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')

      set_axes_equal(ax)
      if save_fig:
          plt.savefig("pf_{:04d}.png".format(self.fig_index))
          self.fig_index += 1
      else:
          plt.show()
    
    def set_dof(self, i, theta, d, a, alpha, is_angular=True):
        """
        Store DH params for joint i.

        @type  i: int
        @param i: joint id from 0 to n-1

        @type  theta: float
        @param theta: theta params in radians

        @type  d: float
        @param d: d param in m

        @type  a: float
        @param a: a param in m

        @type  alpha: float
        @param alpha: alpha param in radians

        @type  is_angular: bool
        @param is_angular: true if the joint is angular false if it is prismatic
        """

        assert i <= self.n_dof
        self.dh_params[i, self.param['THETA']] = theta
        self.dh_params[i, self.param['D']] = d
        self.dh_params[i, self.param['A']] = a
        self.dh_params[i, self.param['ALPHA']] = alpha
        self.dh_params[i, self.param['IS_ANGULAR']] = is_angular
      
    def get_tf_dof(self, i):
        """
        Builds the transformation matrix for joint i according to DH.

        @type  i: int
        @param i: joint id from 0 to n-1

        @rtype:   numpy array
        @return:  the transformation matrix for joint i.
        """

        assert i <= self.n_dof
        ct = np.cos(self.dh_params[i, self.param['THETA']])
        st = np.sin(self.dh_params[i, self.param['THETA']])
        ca = np.cos(self.dh_params[i, self.param['ALPHA']])
        sa = np.sin(self.dh_params[i, self.param['ALPHA']])
        d = self.dh_params[i, self.param['D']]
        a = self.dh_params[i, self.param['A']]

        tf = np.array([[ct, -st*ca, st*sa,  a*ct],
                     [st, ct*ca,  -ct*sa, a*st],
                     [0,  sa,     ca,     d],
                     [0,  0,      0,      1]])
        return tf

    def get_tf(self):
        """
        Composes all the transformation matrix for joints 0 to n-1 according to DH.

        @rtype:   numpy array
        @return:  the R^T_H matrix for the manipulator.
        """

        tf = np.identity(4)
        for i in range(self.n_dof):
            tf = np.dot(tf, self.get_tf_dof(i))
        
        return tf
    

    def set_joints(self, q_values):
        """
        Sets the angular position (theta) or the prismatic position (d) for each joint.
        @type  q_values: list()
        @param q_values: list of floats where each value is the value of a joint
        """
        if len(q_values) != self.n_dof:
            print("Invalid position. {} DoF are required.".format(self.n_dof))
        else:
            for i in range(self.n_dof):    
                if self.dh_params[i, self.param['IS_ANGULAR']]:
                    self.dh_params[i, self.param['THETA']] = wrap_angle(q_values[i])
                else:
                    self.dh_params[i, self.param['D']] = q_values[i]

    def get_joints_value(self):
        """
        Return the value for each joint. This value will be stored in dh_params THETA or D depending
        if the joint is angular or prismatic.

        @rtype: numpy array
        @return: numpy array with the value for each joint (in radians or meters)
        """
        ret = np.zeros(self.n_dof)
        for i in range(self.n_dof):
            if self.dh_params[i, self.param['IS_ANGULAR']]:
                ret[i] = self.dh_params[i, self.param['THETA']]
            else:
                ret[i] = self.dh_params[i, self.param['D']]
        return ret

    def get_TE_pose(self):
        """
        Return the position (x, y, z) and angle (\phi_n, \phi_o, \phi_a) for the terminal element (TE).
        Use the R^T_H matrix that relates the base with the TE and the equations seen at class to obtain
        \phi_n, \phi_o and \phi_a from a rotation matrix.

        @rtype: numpy array
        @return: numpy array with the values x, y, z, angle_normal, angle_orientation, angle_approach for the TE
        """

        rth = self.get_tf()
        R = rth[0:3, 0:3]
        a_z = math.atan2(R[1, 0], R[0, 0])
        a_y = math.atan2(-R[2, 0], R[0, 0]*math.cos(a_z) + R[1, 0] * math.sin(a_z))
        a_x = math.atan2(-R[1, 2] * math.cos(a_z) + R[0, 2] * math.sin(a_z), R[1, 1] * math.cos(a_z) - R[0, 1] * math.sin(a_z))
        return np.array([rth[0, 3], rth[1, 3], rth[2, 3], a_x, a_y, a_z])  


    def set_limits(self, joint_id, min_value, max_value):
        """
        Set joint limits.
        """
        assert joint_id < self.n_dof
        assert joint_id >= 0
        assert type(joint_id) == int
        assert min_value < max_value
        
        self.limits[joint_id, 0] =  min_value
        self.limits[joint_id, 1] =  max_value
    
    def check_limits(self, q_values):
        assert len(q_values) == self.n_dof
        for i, q in enumerate(q_values):
            if self.limits[i, 0] < self.limits[i, 1]:
                if self.dh_params[i, self.param['IS_ANGULAR']]:
                    v = wrap_angle(q)
                else:
                    v = q
                if v < self.limits[i, 0] or v > self.limits[i, 1]:
                    return False
        return True

def set_axes_equal(ax):
      x_limits = ax.get_xlim3d()
      y_limits = ax.get_ylim3d()
      z_limits = ax.get_zlim3d()

      x_range = abs(x_limits[1] - x_limits[0])
      x_middle = np.mean(x_limits)
      y_range = abs(y_limits[1] - y_limits[0])
      y_middle = np.mean(y_limits)
      z_range = abs(z_limits[1] - z_limits[0])
      z_middle = np.mean(z_limits)

      # The plot bounding box is a sphere in the sense of the infinity
      # norm, hence I call half the max range the plot radius.
      plot_radius = 0.5 * max([x_range, y_range, z_range])

      ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
      ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
      ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def wrap_angle(angle):
    """ Wraps angle between 0 and 2 pi """
    return (angle + ( 2.0 * math.pi * math.floor( ( math.pi - angle ) / ( 2.0 * math.pi ) ) ) )