# CPM_landmark
基于CPM网络结构的关键点检测模型

# 版本要求
py2 or py3<br>
tensorflow>=1.2<br>

opencv>=3.2


# 文件说明
bt.py 生成带有heatmap的训练数据的tfrecords<br>
bv.py 生成带有heatmap的验证数据的tfrecords<br><br>

blouse.py 训练模型<br>
blouse_test.py 测试数据<br><br>

visual.py 可视化

# 参考文献
[Wei S E, Ramakrishna V, Kanade T, et al. Convolutional Pose Machines[J]. 2016:4724-4732.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.pdf)<br>
[Howard A G, Zhu M, Chen B, et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. 2017.](https://arxiv.org/pdf/1704.04861.pdf)

