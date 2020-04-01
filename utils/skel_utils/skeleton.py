import numpy as np
from utils.quaternion import q_angle, q_rel_angle, q_rotate, normalize

class Skeleton:
    """
    Abstraction for a skeleton.
    """
    def __init__(self, parents):
        self.num_joints = len(parents)
        self.children = {i:[] for i in range(-1, self.num_joints)}  # {i : [child1, child2, ...], ...}
        self.build_tree(parents)
        self.dfs = self.DFS()

    def build_tree(self, parents):
        """Build a tree from parent of each joint
        parents: [parent(0), parent(1), ...] -1 for the parent of root
        """ 
        self.parents = parents
        for i, p in enumerate(parents):
            self.children[p].append(i)
        return self.children

    def compute_bone_lens(self, xyz):
        """Compute bone lengths from joint coordinates.
        Args:
            xyz: np.array(3, J)
        """
        bone_lens = []
        for i in range(len(self.parents)):
            if self.parents[i] != -1:
                bone_lens.append(np.linalg.norm(xyz[..., i] - xyz[..., self.parents[i]], axis=-1))
            else:
                bone_lens.append(0.)
        return bone_lens

    def DFS(self):
        """Depth first traversal
        """
        dfs = [-1]
        self.DFS_rec(dfs, -1)
        return dfs

    def DFS_rec(self, dfs, root):
        if len(self.children[root]) == 0:
            return
        for child in self.children[root]:
            dfs.append(child)
            self.DFS_rec(dfs, child)

    def xyz_rotate(self,xyz,y_only=False,new_rotate=True):
        """
        Random Rotate xyz, the sample_num is expected to be 1.
        Args:
            xyz: np.array(..., 3)
        Return:
            xyz: np.array(..., 3)
        """
        s = list(xyz.shape)
        s[-1] = 4
        if new_rotate:
            self.rotate = np.zeros(s)
            theta = np.random.rand()*np.pi  # 二分之一转动角
            alpha = np.random.rand()*np.pi  # 转轴与z轴夹角
            beta = np.random.rand()*np.pi*2 # 转轴xOy投影与x轴夹角
            if y_only:
                alpha,beta = np.pi/2,np.pi/2
            z = np.cos(alpha)*np.sin(theta)
            x = np.sin(alpha)*np.cos(beta)*np.sin(theta)
            y = np.sin(alpha)*np.sin(beta)*np.sin(theta)
            self.rotate[...,:] = np.array([np.cos(theta),x,y,z])
        return q_rotate(self.rotate,xyz)

    def xyz2qrel(self, xyz, return_base=False):
        """Convert xyz to quaternion.
        Args:
            xyz: np.array(..., 3, num_joint)
        Return:
            q: np.array(..., 4, num_joint)            
        """
        s = list(xyz.shape)
        s[-2] = 4
        q = np.zeros(s)

        unit_z = np.zeros_like(xyz[..., 0])
        unit_z[..., 2] = 1

        for pivot in range(len(self.parents)):
            begin = self.parents[pivot]
            for j in range(len(self.children[pivot])):
                end = self.children[pivot][j]
                if begin == -1:
                    q[..., 0, pivot] = 1.  # The root
                    q[..., end] = q_angle(xyz[..., pivot] - xyz[..., self.children[pivot][0]], xyz[..., end] - xyz[..., pivot])
                    base = xyz[..., pivot] - xyz[..., self.children[pivot][0]]
                else:
                    q[..., end] = q_angle(xyz[..., pivot] - xyz[..., begin], xyz[..., end] - xyz[..., pivot])
        if return_base:
            return q,base
        else:
            return q

    def qrel2xyz(self, q, bone_lens, base=None):
        """Convert quaternion (w.r.t. parent) to xyz.
        Args:
            q: np.array(..., 4, num_joint)
            bone_lens: [i: [len(i, child_j), ...], ...]
        Return:
            xyz: np.array(..., 4, num_joint)            
        """

        s = list(q.shape)
        s[-2] = 3
        xyz = np.zeros(s)
        if base is None:
            unit_z = np.zeros_like(xyz[..., 0])
            unit_z[..., 2] = 1
        else:
            unit_z = base
        # Compute offsets between two adjacent joints
        for i in range(1, len(self.dfs)):
            pivot = self.dfs[i]
            begin = self.parents[pivot]
            print(begin, pivot)
            for j in range(len(self.children[pivot])):
                end = self.children[pivot][j]
                if begin == -1:  
                    v = q_rotate(q[..., end], unit_z)
                else:
                    v = q_rotate(q[..., end], xyz[..., pivot] - xyz[..., begin])
                xyz[..., end] = normalize(v) * bone_lens[end] + xyz[..., pivot]
        return xyz

    def xyz2qabs(self, xyz):
        """Convert xyz to quaternion w.r.t. axis z.
        Args:
            xyz: np.array(..., 3, num_joint)
        Return:
            q: np.array(..., 4, num_joint)            
        """
        s = list(xyz.shape)
        s[-2] = 4
        q = np.zeros(s)

        unit_z = np.zeros_like(xyz[..., 0])
        unit_z[..., 2] = 1

        for joint in range(len(self.parents)):
            parent = self.parents[joint]
            if parent == -1: # for root v is the vector from root to its first child joint
                v = xyz[..., joint] - xyz[..., self.children[joint][0]]
                q[..., joint] = q_angle(unit_z, v)
            else:
                v = xyz[..., joint] - xyz[..., parent]
                q[..., joint] = q_angle(unit_z, v)
        return q

    def qabs2xyz(self, q, bone_lens):
        """Convert quaternion (w.r.t. axis z) to quaternion.
        Args:
            q: np.array(..., 4, num_joint)
            bone_lens: [i: [len(i, child_j), ...], ...]
        Return:
            xyz: np.array(..., 4, num_joint)            
        """
        s = list(q.shape)
        s[-2] = 3
        xyz = np.zeros(s)

        unit_z = np.zeros_like(xyz[..., 0])
        unit_z[..., 2] = 1
        for i in range(1, len(self.dfs)):
            joint = self.dfs[i]
            parent = self.parents[joint]
            if parent != -1:
                v = q_rotate(q[..., joint], unit_z)
                xyz[..., joint] = normalize(v) * bone_lens[joint] + xyz[..., parent]
        return xyz


if __name__ == '__main__':
    import ntu_info
    sk = Skeleton(parents=ntu_info.PARENTS)
    print(sk.children)
    from feeder import Feeder
    train_data_path = '/home/xuan/data/NTU-RGB-D/xyz/xsub/train_data.npy'
    train_label_path = '/home/xuan/data/NTU-RGB-D/xyz/xsub/train_label.pkl'
    train_data = Feeder(train_data_path, train_label_path, num_samples=-1, num_frames=20, mmap=True)
    xyz, _ = train_data[0]
    xyz = xyz[:, 0, :]
    print(xyz.shape)
    print(sk.compute_bone_lens(xyz))
