from nuscenes.nuscenes import NuScenes as V2XSimDataset
from nuscenes.utils.data_classes import LidarPointCloud
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from mmseg.apis import inference_model, show_result_pyplot
from Scene.Scene import Scene
from .utils import mask2image
from PIL import Image

class Vehicle:
    def __init__(self, dataset:V2XSimDataset, vehicle_id:int, scene_id:int=0):
        self.dataSet = dataset
        self.vehicle_id = vehicle_id

        #RBG cameras
        self.tokensCamera_Stream ={} 
        self.channelsCamera = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for ch in self.channelsCamera:
            self.tokensCamera_Stream[ch] = dataset.field2token('sample_data', 'channel', f'{ch}_id_{vehicle_id}')

        #Segmentation of RGB camera
        self.tokensSeg_Stream ={}  
        self.channelsSeg = ['SEG_FRONT', 'SEG_FRONT_LEFT', 'SEG_FRONT_RIGHT', 'SEG_BACK', 'SEG_BACK_LEFT', 'SEG_BACK_RIGHT'] 
        for ch in self.channelsSeg:
            self.tokensSeg_Stream[ch] = dataset.field2token('sample_data', 'channel', f'{ch}_id_{vehicle_id}')
        
        #Lidar tokens
        self.tokensLidar_Stream = dataset.field2token('sample_data', 'channel', f'LIDAR_TOP_id_{vehicle_id}')

        self.egoTranslation_Stream = []
        self.egoRotation_Stream = []
        for t in self.tokensCamera_Stream['CAM_FRONT']:
            self.egoTranslation_Stream += [self.dataSet.get('ego_pose', t)['translation']]
            self.egoRotation_Stream += [self.dataSet.get('ego_pose', t)['rotation']]
        
        #Calibration parameters(intrinsic & extrinsic) for cameras and lidar
        self.calibParam ={}    
        for ch in (self.channelsCamera + ['LIDAR']):
            if ch != 'LIDAR':
                token = self.tokensCamera_Stream[ch][0]
            else:
                token = self.tokensLidar_Stream[0]
            tokenCalibList = self.dataSet.get('sample_data', token)['calibrated_sensor_token']
            data = self.dataSet.get('calibrated_sensor', tokenCalibList)
            self.calibParam[ch] = data

        return


    def _getData(self, token:str, key:str=None):
        data = self.dataSet.get('sample_data', token)
        if key in data.keys():
            return data[key]
        else:
            return data
        
    def readData(self, token:str):
        data = self.dataSet.get('sample_data', token)
        filePath, fileforamt = data['filename'], data['fileformat']
        
        previous_directory = os.getcwd()
        os.chdir(self.dataSet.dataroot)
        
        if fileforamt not in ['jpg', 'npz', 'pcd']:
            raise Exception("wrong format")
        
        if fileforamt =='jpg':
            img = cv2.imread(filePath)
            os.chdir(previous_directory)
            return img
        elif fileforamt =='npz':
            data = np.load(filePath)
            os.chdir(previous_directory)
            for item in data.files:
                return data[item]
        elif fileforamt == 'pcd':
            pc = LidarPointCloud.from_file(filePath)
            os.chdir(previous_directory)
            return pc   
        
    def _showVideo_Cam(self, ch:str, delay:int):
        for t in self.tokensCamera_Stream[ch]:
            img = self.readData(t)
            cv2.imshow("image", img)
            cv2.waitKey(delay)
        cv2.destroyAllWindows()
        return

    def _showVideo_Seg(self, ch:str, delay:int):
        for t in self.tokensSeg_Stream[ch]:
            segIMG = self.readData(t)
            segIMG_scaled = np.interp(segIMG, (0, 31), (0, 255)).astype(np.uint8)
            cv2.imshow('image', segIMG_scaled)
            cv2.waitKey(delay)
        cv2.destroyAllWindows()
        return
    
    def _showVideo_Lidar2D(self, delay:int, axes_limit:float = 40, segModel=None):
        '''
        dont work in ipynb files
        '''
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

        for indx,t in enumerate(self.tokensLidar_Stream):
            plt.cla()
            if segModel == None:
                points = self.readData(t).points[:3,:]
                # points = self.transform_Lidar2Cam('CAM_FRONT', t).points[:3,:]
                dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
                colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            else:
                points, _ = self.segmentPoints(model=segModel,timeStamp=indx)
                colors = points[3,:]

            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=3.0)
            ax.plot(0, 0, 'x', color='red') # Show ego vehicle.
            
            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
            ax.axis('off')
            ax.set_aspect('equal')
            
            plt.pause(delay)

        plt.show()
        return

    def showVideo(self, ch:str, delay:int=1):
        if ch in self.channelsCamera:
            self._showVideo_Cam(ch, delay)
        elif ch in self.channelsSeg:
            self._showVideo_Seg(ch, delay)
        elif ch == 'LIDAR':
            self._showVideo_Lidar2D(delay)
        return
    
    def transform_Lidar2Cam(self, camChannel:str, lidarToken:str):
        '''
        Returns:\n
        1)List of 2D points visible from camera\n
        2)Indices of PointCloud that are visible from camara list[Bool] (1xn)\n
        3)List of depths(not exactly) from each point in PC to camera  (1xn)
        '''
        pc = self.readData(lidarToken)
        #           [x1, x2, ...]
        #  points = [y1, y2, ...]
        #           [z1, z2, ...]

        lidar_pos = np.array(self.calibParam['LIDAR']['translation'])
        lidar_rot = Quaternion(self.calibParam['LIDAR']['rotation'])
        pc.translate(lidar_pos)
        pc.rotate(lidar_rot.rotation_matrix)

        cam_pos = np.array(self.calibParam[camChannel]['translation'])
        cam_rot = Quaternion(self.calibParam[camChannel]['rotation'])
        pc.translate(-cam_pos)
        pc.rotate(cam_rot.rotation_matrix.T)

        depthsAll = pc.points[2, :]

        points = self._project_PC2Image(camChannel, pc)
        
        # points = view_points(pc.points[:3, :], np.array(self.calibParam[camChannel]['camera_intrinsic']), normalize=True)
        visible_pointsUV, indices = self._filter_visible_points(camChannel, points, depthsAll)
        
        # scatter = plt.scatter(visible_pointsUV[0, :], visible_pointsUV[1, :],c=depthsAll[indices], s=3.0)
        # plt.show()
        return visible_pointsUV, depthsAll, indices

    def _filter_visible_points(self, camChannel:str, points, depths):
        image_width = np.array(self.calibParam[camChannel]['camera_intrinsic'])[0,2] * 2
        image_hight = np.array(self.calibParam[camChannel]['camera_intrinsic'])[1,2] * 2

        mask = np.ones(points.shape[1], dtype=bool)
        mask = np.logical_and(mask, depths > 0.1)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < image_width - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < image_hight - 1)
        #points = points[:, mask]

        visible_lidar_points = points[:, mask]

        return visible_lidar_points, mask 

    def _project_PC2Image(self, camChannel:str, pc:LidarPointCloud):
        view = np.array(self.calibParam[camChannel]['camera_intrinsic'])

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = pc.points.shape[1]
        points = pc.points[:3,:]
        points = np.concatenate((points, np.ones((1, nbr_points))))

        projected_points = np.dot(viewpad, points)
        projected_points /= projected_points[2, :]
        
        return projected_points[:2,:]
    
    def _segmentImage(self, model, img):
        result = inference_model(model, img)

        mask = result.pred_sem_seg.data
        mask = mask.squeeze(0).cpu().numpy()
        return mask
    
    def segmentPoints(self, model, timeStamp:int):
        """
        Segment 3D points using image segmentation for every camera channel
        """
        if timeStamp<0 or timeStamp>len(self.tokensLidar_Stream)-1:
            raise IndexError('Wrong time stamp')
        
        lidarToken = self.tokensLidar_Stream[timeStamp]
        pc = self.readData(lidarToken).points[:3,:]

        allPoints = np.array([]).reshape(5, 0) #xyz, label, index (for all channels)
        masks = []

        for ch in self.channelsCamera:
            imgToken = self.tokensCamera_Stream[ch][timeStamp]

            img = self.readData(imgToken)
            pointsUV, _, ind = self.transform_Lidar2Cam(ch, lidarToken)
            pointsUV = pointsUV.astype(int)
            indices = np.where(ind == True)[0]
            points3D = pc[:,indices]

            mask = self._segmentImage(model, img)
            masks.append(mask)
            labels = mask[pointsUV[1,:], pointsUV[0,:]]

            tempPoints = np.vstack((points3D, labels, indices))
            allPoints = np.hstack((allPoints, tempPoints))
            
        return allPoints, masks
    
    def showSegImages(self, model, timeStamp:int, opacity:float=0.7, masks=None):
        """
        show all 6 images with segmetation maskes at once
        """
        rows = 2
        cols = 3
        fig, axs = plt.subplots(rows, cols, figsize=(15, 6))

        for i, (ch, _ax) in enumerate(zip(self.channelsCamera, axs.flatten())):
            imgToken = self.tokensCamera_Stream[ch][timeStamp]
            img = self.readData(imgToken)
            if masks==None:
                result = inference_model(model, img)
                segIMG = show_result_pyplot(model, img, result, show=False, withLabels=False, opacity=opacity)
            else:
                maskIMG = mask2image(masks[i].astype(int), model.dataset_meta['palette']).convert('RGB')
                img = Image.fromarray(img[:,:,::-1])
                segIMG = Image.blend(img, maskIMG, opacity)
            _ax.imshow(segIMG)
            _ax.title.set_text(f'{ch}_id_{self.vehicle_id}')
            _ax.axis('off') 

        # plt.tight_layout()
        plt.show()
        return
    
    def generateScene(self, seg_Model, timeStamp:int):
        labeledPoints, masks = self.segmentPoints(seg_Model, timeStamp) #xyz, label, index in initial PC

        palette = seg_Model.dataset_meta['palette']
        classes = seg_Model.dataset_meta['classes']
        egoScene = Scene(labeledPoints[:3,:], labeledPoints[3,:], palette, classes)

        return egoScene, masks