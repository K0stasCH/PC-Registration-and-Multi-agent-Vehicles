from Scene.Scnene import Scene
from Vehicle.Vehicle import Vehicle
from nuscenes.nuscenes import NuScenes as V2XSimDataset
import mmseg
from mmseg.apis import inference_model, init_model, show_result_pyplot
import numpy as np
import pickle

if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
        print('Running on the GPU<-----------')

    datapath = "D:\\Dataset-Thesis\\temp\\V2X Sim Mini\\V2X-Sim-2.0-mini"
    v2x_sim = V2XSimDataset(version='v2.0-mini', dataroot=datapath, verbose=True)

    # config_file = f'{mmseg.__path__[0]}\\.mim\\configs\\segformer\\segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py'
    # checkpoint_file = '.\\checkpoints\\seg\\segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'

    # config_file = f'{mmseg.__path__[0]}\\.mim\\configs\\resnest\\resnest_s101-d8_fcn_4xb2-80k_cityscapes-512x1024.py'
    # checkpoint_file = '.\\checkpoints\\seg\\fcn_s101-d8_512x1024_80k_cityscapes_20200807_140631-f8d155b3.pth'

    # config_file = f'{mmseg.__path__[0]}\\.mim\\configs\\pspnet\\pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
    # checkpoint_file = '.\\checkpoints\\seg\\pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

    # config_file = "C:\\Users\\konst\Downloads\\myModel\\pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
    # checkpoint_file = "C:\\Users\\konst\Downloads\\myModel\\iter_200.pth"

    config_file = '.\\checkpoints\\pspNET\\pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
    checkpoint_file = '.\\checkpoints\\pspNET\\iter_1000.pth'

    model = init_model(config_file, checkpoint_file, device='cuda:0')

    v1 = Vehicle(v2x_sim,1)
    v4 = Vehicle(v2x_sim,4) 
    # v.transform_Lidar2Cam('CAM_FRONT', v.tokensLidar_Stream[0])
    # v.showVideo('LIDAR')
    # points = v.transform_Lidar2Cam('CAM_FRONT', v.tokensLidar_Stream[1])

    # test = v.segmentPoints(model, 1)
    time = 1
    # v1.showSegImages(model, time)
    tempScene1 = v1.generateScene(model,model, time)
    tempScene4 = v4.generateScene(model,model, time)

    # tempScene1.visualizePCD()
    # tempScene4.visualizePCD()

    # x = tempScene.visualizePCD()

    g1 = tempScene1.generateGraph(classNames=['person', 'vehicle'])
    g1.plotGraph()
    
    g4 = tempScene4.generateGraph()
    g4.plotGraph()
    graphs = [g1, g4]

    # with open("graphs.pickle", "wb") as file:
    #     pickle.dump(graphs, file)

    # for t in v.tokensCamera['CAM_BACK']:
    #     calibtoken = v.getData(t)['ego_pose_token']
    #     data = v2x_sim.get('ego_pose', calibtoken)['translation']
    #     print(data)

    print(0)