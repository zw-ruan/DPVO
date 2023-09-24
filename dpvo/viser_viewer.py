import time

from multiprocessing import Value
import torch
import torch.multiprocessing as mp
import viser
import viser.transforms as tf


class ViserViewer(mp.Process):
    def __init__(self, image:torch.Tensor, poses:torch.Tensor, points:torch.Tensor,
                 colors:torch.Tensor, intrinsics: torch.Tensor):
        super().__init__()
        self.image_ = image 
        self.poses_ = poses

        self.points_ = points
        self.colors_ = colors
        self.intrinsics_ = intrinsics

        # share ammong process 
        self.redraw = Value('b', False)
        self.lock = mp.Lock()
        self.frame_cnt = Value('i', 0)

        self.start()

    def update_image(self, keyframe_num=0):
        self.lock.acquire()
        self.frame_cnt.value = keyframe_num
        self.redraw.value = True
        # self.cur_rgb_image = image.permute((1, 2, 0)).to('cpu')
        self.lock.release()

    def setup_gui(self, server):
        with server.add_gui_folder("Control"):
            gui_camera_scale = server.add_gui_slider(
                "Camera size", min=0, max=0.2, step=0.01, initial_value=0.05
            )
            gui_point_scale = server.add_gui_slider(
                "Point size", min=0, max=0.1, step=0.01, initial_value=0.03
            )

        lock = self.lock
        redraw = self.redraw
        @gui_camera_scale.on_update
        def _(_) -> None:
            lock.acquire()
            redraw.value = True
            lock.release()

        @gui_point_scale.on_update
        def _(_) -> None:
            lock.acquire()
            redraw.value = True
            lock.release()

        return gui_camera_scale, gui_point_scale

    def run(self):
        port = 8080
        server = viser.ViserServer(port=port)

        camera_controller, point_controller = self.setup_gui(server)

        while True:
            self.lock.acquire()
            if not self.redraw.value:
                self.lock.release()
                time.sleep(0.5)
                continue

            self.redraw.value = False
            valid_cnt = self.frame_cnt.value
            self.lock.release()
            # server.reset_scene()

            if valid_cnt < 10:
                continue

            # draw poses
            cur_poses = self.poses_[:valid_cnt].cpu().numpy()
            for idx in range(cur_poses.shape[0]):
                wxyz = cur_poses[idx][[-1, 3, 4, 5]]
                T_current2world = tf.SE3.from_rotation_and_translation(
                    tf.SO3(wxyz), cur_poses[idx][[0, 1, 2]]).inverse()

                # aspect = (self.image_[idx].shape[1] / self.image_[idx].shape[0]).item()
                aspect = 1920 / 1080
                fov = torch.arctan(self.intrinsics_[idx][3] / self.intrinsics_[idx][1]).item()
                server.add_camera_frustum(f'dpvo/frame_{idx}', 90, aspect,
                                          color=(255, 0, 0),
                                          wxyz=T_current2world.wxyz_xyz[:4],
                                          position=T_current2world.wxyz_xyz[4:],
                                          scale=float(camera_controller.value))

            # draw points
            cur_colors = self.colors_[:valid_cnt].cpu().numpy().reshape(-1, 3)
            # weird self.points_ and self.colors_ have different shape
            cur_points = self.points_[:cur_colors.shape[0]].cpu().numpy().reshape(-1, 3)
            server.add_point_cloud('dpvo/pointcloud', points=cur_points,
                                   colors=cur_colors, point_size=float(point_controller.value))

