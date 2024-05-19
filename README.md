# Algorithm based on SAM for plasma hole structure segmentation

SAM使用了[segment anything](https://github.com/facebookresearch/segment-anything.git)
### 1. 基于SAM的分割算法代码（tools/data_deal_publish-v3.ipynb）
包含了分割算法以及非惯性系参数计算的相关函数。
##### 实现分割的函数
```python
get_mask(f_trap,delta_f, predic-tor,multi_points=True,fix_cneter_x=True,radias=25,num_points=4,num_candidate=10,thre=0.6,div=40,input_point=None, show=False, save=False, save_dir=None) -> mask, is_construct
```
* `f_trap`: 粒子数密度分布  
* `delta_f`: 数密度分布减平衡数密度  
* `predictor`: 使用的SAM基础模型  
* `multi_points`: 是否采用多点分割  
* `fix_cneter_x`: 是否固定参考点的平坐标  
* `radias`: 多点分割时的取样半径  
* `num_points`: 多点分割取点个数  
* `num_candidate`: 动态选取参考点的候选点数量
* `thre`: 基于掩码分数的过滤阈值
* `div`: 基于掩码特征过滤的阈值
* `input_point`: 手动选取点的输入
* `show`: 是否可视化
* `save`: 是否保存图片
* `save_dir`: 保存路径
* `mask`: 分割掩码
* `is_construct`: 是否为hole结构
##### 非惯性系参数计算函数
```python
get_alpha(f_trap, delta,mask_origin, mgL,x_range=[0,2*np.pi]) -> alpha
```
* `f_trap`: 数密度分布
* `delta`: 相位变换
* `mask_origin`: 与f_trap的shape相同的掩码
* `mgL`: 相关参数
* `x_range`: 积分区间
### 2. 电流计算代码（tools/current_publish.ipynb）
#### 计算束缚电流和漂移电流的函数
```python
current(f_trap,mask_origin,J,Pi,omega_ce,ks,x_range=[0,2*np.pi],y_range=[-0.1,0.1])
```
* `f_trap`: 数密度分布
* `mask_origin`: 与f_trap的shape相同的掩码
* `J`, `Pi`, `omega_ce`, `ks`: 相关物理参数
* `x_range`，`y_range`: 积分区间
##### 计算冷等离子体电流的函数
```python
current2(gyro,A)
```
##### 束缚电流和漂移电流的比例拟合与可视化
##### 两个方向电流分量的计算与可视化
##### 电流与矢势相位差的计算与可视化
#####  垂直速度计算程序（current.ipynb）
### 3. 电场与电流的拟合以及波速、增长率的计算代码（tools/fit.ipynb）
##### 拟合电场与电流的函数
```python
fit(time,cut=(0,-1),show=False)
```
* `time`: 用于拟合的时间点
* `cut`: 用于拟合的空间范围
* `show`: 是否展示可视化
##### 增长率与波速的计算
```python
# 增长率 r=ar-bi*(dphi/ds)
gamma=a_lis.real[:,None]-b_lis.imag[:,None]*partial_phi_lis
# 新波速 v'=v-br
v2=v_lis-b_lis.imag[:,None]
```
### 4. hole结构的可视化代码（utils/data2img.ipynb）
##### 可视化任一时空位置的hole结构
##### 可视化结构随时间的变化图像
### 5. 单摆哈密顿量代码（utils/单摆.ipynb，utils/单摆-常力矩.ipynb）
##### 单摆的哈密顿量等高线图以及摆长、外力对哈密顿量影响的研究
