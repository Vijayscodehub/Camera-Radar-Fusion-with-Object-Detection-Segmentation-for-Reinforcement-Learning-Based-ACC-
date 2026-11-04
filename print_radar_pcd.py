from nuscenes.utils.data_classes import RadarPointCloud
p = RadarPointCloud.from_file('data/nuscenes_dataset/v1.0-mini/samples/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178.pcd')
print(p.points.shape)
print(p.points[:, :5])