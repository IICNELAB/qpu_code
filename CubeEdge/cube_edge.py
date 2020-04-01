import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from utils.quaternion import q_angle, q_rotate

class CubeEdge(torch.utils.data.Dataset):
    """Generate CubeEgde dataset on the fly.
    1. Sample a cube.
    2. Add noise to vertices if training
    2. Apply random shear on xy plane
    Vertices ind :
      7----6   
     /|   /|
    3----2 |   
    | 4--|-5 
    |/   |/
    0----1 
    """
    def __init__(self, train, num_edges=4, num_samples=2000, use_quaternion=True, random_rotation=False, sigma=0.1):
        """
        Params:
            train: (bool) whether is training set
            num_edges: (int) number of edges
            num_samples: (int) number of samples
            use_quaternion: (bool) whether convert to quaternion
            random_rotation: (bool) whether apply random rotation
            sigma: (float) standard deviation of the noise applied to vertices
        """
        self.seed_is_set = False  # multi threaded loading
        self.num_samples = num_samples
        self.train = train
        self.use_quaternion = use_quaternion
        self.random_rotation = random_rotation
        self.sigma = sigma 
        self.vertices = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype="float32") * 2 - 1.
       
        # All possible shapes
        if num_edges == 4:
            self.num_shapes = 4
            self.shapes = np.array([self._make_shape(0, 1, 2, 3, 0),
                                    self._make_shape(0, 1, 2, 3, 7),
                                    self._make_shape(0, 1, 2, 6, 7),
                                    self._make_shape(0, 1, 2, 6, 5)])    
        elif num_edges == 5:
            self.num_shapes = 8
            self.shapes = np.array([self._make_shape(0, 1, 2, 3, 0, 1),
                                    self._make_shape(0, 1, 2, 3, 0, 4),
                                    self._make_shape(0, 1, 2, 3, 7, 4),
                                    self._make_shape(0, 1, 2, 3, 7, 6),
                                    self._make_shape(0, 1, 2, 6, 7, 4),
                                    self._make_shape(0, 1, 2, 6, 7, 3),
                                    self._make_shape(0, 1, 2, 6, 5, 4),
                                    self._make_shape(0, 1, 2, 6, 5, 1)])
        elif num_edges == 6:
            self.num_shapes = 16
            self.shapes = np.array([self._make_shape(0, 1, 2, 3, 0, 1, 2),
                                    self._make_shape(0, 1, 2, 3, 0, 1, 5),
                                    self._make_shape(0, 1, 2, 3, 0, 4, 7),
                                    self._make_shape(0, 1, 2, 3, 0, 4, 5),                                    
                                    self._make_shape(0, 1, 2, 3, 7, 4, 0),
                                    self._make_shape(0, 1, 2, 3, 7, 4, 5),
                                    self._make_shape(0, 1, 2, 3, 7, 6, 2),
                                    self._make_shape(0, 1, 2, 3, 7, 6, 5),
                                    self._make_shape(0, 1, 2, 6, 7, 4, 0),
                                    self._make_shape(0, 1, 2, 6, 7, 4, 5),
                                    self._make_shape(0, 1, 2, 6, 7, 3, 2),
                                    self._make_shape(0, 1, 2, 6, 7, 3, 0),
                                    self._make_shape(0, 1, 2, 6, 5, 4, 0),
                                    self._make_shape(0, 1, 2, 6, 5, 4, 7),                                    
                                    self._make_shape(0, 1, 2, 6, 5, 1, 0),
                                    self._make_shape(0, 1, 2, 6, 5, 1, 2)])    

        elif num_edges == 7:
            self.num_shapes = 32
            self.shapes = np.array([self._make_shape(0, 1, 2, 3, 0, 1, 2, 3),
                                    self._make_shape(0, 1, 2, 3, 0, 1, 2, 6),
                                    self._make_shape(0, 1, 2, 3, 0, 1, 5, 4),
                                    self._make_shape(0, 1, 2, 3, 0, 1, 5, 6),
                                    self._make_shape(0, 1, 2, 3, 0, 4, 7, 3),
                                    self._make_shape(0, 1, 2, 3, 0, 4, 7, 6),
                                    self._make_shape(0, 1, 2, 3, 0, 4, 5, 1),  
                                    self._make_shape(0, 1, 2, 3, 0, 4, 5, 6),                                  
                                    self._make_shape(0, 1, 2, 3, 7, 4, 0, 1),
                                    self._make_shape(0, 1, 2, 3, 7, 4, 0, 3),
                                    self._make_shape(0, 1, 2, 3, 7, 4, 5, 1),
                                    self._make_shape(0, 1, 2, 3, 7, 4, 5, 6),
                                    self._make_shape(0, 1, 2, 3, 7, 6, 2, 1),
                                    self._make_shape(0, 1, 2, 3, 7, 6, 2, 3),
                                    self._make_shape(0, 1, 2, 3, 7, 6, 5, 1),
                                    self._make_shape(0, 1, 2, 3, 7, 6, 5, 4),
                                    self._make_shape(0, 1, 2, 6, 7, 4, 0, 1),
                                    self._make_shape(0, 1, 2, 6, 7, 4, 0, 3),
                                    self._make_shape(0, 1, 2, 6, 7, 4, 5, 1),
                                    self._make_shape(0, 1, 2, 6, 7, 4, 5, 6),
                                    self._make_shape(0, 1, 2, 6, 7, 3, 2, 1),
                                    self._make_shape(0, 1, 2, 6, 7, 3, 2, 6),
                                    self._make_shape(0, 1, 2, 6, 7, 3, 0, 1),
                                    self._make_shape(0, 1, 2, 6, 7, 3, 0, 4),
                                    self._make_shape(0, 1, 2, 6, 5, 4, 0, 1),
                                    self._make_shape(0, 1, 2, 6, 5, 4, 0, 3),
                                    self._make_shape(0, 1, 2, 6, 5, 4, 7, 3),
                                    self._make_shape(0, 1, 2, 6, 5, 4, 7, 6),                                    
                                    self._make_shape(0, 1, 2, 6, 5, 1, 0, 3),
                                    self._make_shape(0, 1, 2, 6, 5, 1, 0, 4),
                                    self._make_shape(0, 1, 2, 6, 5, 1, 2, 3),
                                    self._make_shape(0, 1, 2, 6, 5, 1, 2, 6)])    

    def _make_shape(self, *vertice_ind):
        shape = []
        for ind in vertice_ind:
            shape.append(self.vertices[ind])
        return np.array(shape)
    
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Return:
            shape: (np.array) (num_edges + 1, 3) if use_quaternion=True, 
                              (num_edges, 4) otherwise.  
            shape_ind: (int) which shape is sampled
        """
        if self.train:
            self.set_seed(index)
        else:
            self.set_seed(index + self.num_samples)
        shape_ind = np.random.randint(self.num_shapes)
        shape = self.shapes[shape_ind]
        if not self.train:
            shape = self.add_noise(shape, self.sigma)
        shape = self.apply_random_shear(shape)
        if self.random_rotation:
            shape = self.apply_random_rotation(shape)
        if self.use_quaternion:
            shape = self.pos2qu(shape)
        return np.float32(shape), shape_ind

    @staticmethod
    def add_noise(shape, sigma):
        shape = shape.copy()
        for i in range(shape.shape[0]):
            shape[i] += np.random.randn(3) * sigma
        return shape        

    @staticmethod
    def apply_random_shear(shape):
        """random shear on xy plane
        """
        shape = shape.copy()
        hxy, hxz, hyx, hyz, hzx, hzy = np.random.randn(6)
        # Keep axis z unchanged
        hzx = 0
        hzy = 0
        for i in range(shape.shape[0]):
            shape[i][0], shape[i][1], shape[i][2] = shape[i][0] + hxy * shape[i][1] + hxz * shape[i][2], \
                                                    shape[i][1] + hyx * shape[i][0] + hyz * shape[i][2], \
                                                    shape[i][2] + hzx * shape[i][0] + hzy * shape[i][1]
        return shape

    @staticmethod
    def apply_random_rotation(shape):
        shape = shape.copy()
        angle_x, angle_y, angle_z = np.random.rand(3) * 2 * np.pi 
        for i in range(shape.shape[0]):
            # x roll
            shape[i][1], shape[i][2] = np.cos(angle_x) * shape[i][1] - np.sin(angle_x) * shape[i][2], \
                                       np.sin(angle_x) * shape[i][1] + np.cos(angle_x) * shape[i][2]
            # y roll
            shape[i][0], shape[i][2] = np.cos(angle_y) * shape[i][0] - np.sin(angle_y) * shape[i][2], \
                                       np.sin(angle_y) * shape[i][0] + np.cos(angle_y) * shape[i][2]
            # z roll
            shape[i][0], shape[i][1] = np.cos(angle_z) * shape[i][0] - np.sin(angle_z) * shape[i][1], \
                                       np.sin(angle_z) * shape[i][0] + np.cos(angle_z) * shape[i][1]
        return shape

    @staticmethod
    def pos2qu(shape):
        """Convert 3D coordinates to quaternion rotation
        """
        num_vertices = shape.shape[0]
        q_shape = np.zeros([num_vertices - 1, 4])
        q_shape[0] = q_angle(shape[1] - shape[0], shape[1] - shape[0])
        for i in range(1, num_vertices - 1):
            q_shape[i] = q_angle(shape[i - 1] - shape[i], shape[i + 1] - shape[i])
        return q_shape

    @staticmethod
    def plot_cube(ax, shape, name):
        for i in range(shape.shape[0]):
            ax.scatter(shape[i][0], shape[i][1], shape[i][2], marker='o')
        for i in range(shape.shape[0] - 1):
            ax.plot([shape[i][0], shape[i + 1][0]], [shape[i][1], shape[i + 1][1]], [shape[i][2], shape[i + 1][2]])
        ax.set_title(name)
        x_max, x_min = np.max(shape[:, 0]), np.min(shape[:, 0])
        y_max, y_min = np.max(shape[:, 1]), np.min(shape[:, 1])
        z_max, z_min = np.max(shape[:, 2]), np.min(shape[:, 2])
        max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) / 2.0
        mid_x = (x_max + x_min) / 2.0
        mid_y = (y_max + y_min) / 2.0
        mid_z = (z_max + z_min) / 2.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


if __name__ == "__main__":
    cube_edges = CubeEdge(train=True, num_edges=4, use_quaternion=False)
    rows = 4
    cols = 6
    imsize = 2.5
    fig = plt.figure('samples', figsize=(imsize * cols, imsize * rows))
    for i in range(rows * cols):
        shape, label = cube_edges[i]    
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        CubeEdge.plot_cube(ax, shape, label)
    plt.show()