<<<<<<< HEAD
## File tree

### Mscripts
    E:.
    │  demo*.m 				# demo文件，用于
    │  example_script.m		# 生成各种预测子	
    │  readme.md 			# 
    │  step_01.m 			# 挑选通信模型的预测子
    │  step_02.m 			# 使用预测子（euc, cmc, navplMS）和SC预测FC
    │  step_02_2.m 			# 统计模型，spearman 相关
    │  step_03.m 			# 可视化，map on brain
    │  step_04.m 			# Unimodal-Heteromodal map
    │  step_05.m 			# 结构/功能梯度分析
    │  step_06.m 			# pattern，方法的相似性
    │  step_07.m 			# developmental pattern [GAM analysis]
    │
    └─fcn
        │  diffusion_maps.m     # 功能梯度计算函数
        │  myboxplot.m          # boxplot
        │  offsetAxes.m         # xy轴分离
        │  Violin.m             
        │  violinplot.m
        │
        └─fcn                   # 计算图论参数

### DATA
- File path: [./DATA/ENVS]
- stps.mat <Stimulating Parent Scale>
    - VariableNames: ['participant_id','interview_age','score','idx_in_feature_dataset']
    - data_FT: [331*4]
    - data_PT1: [92*4]
=======
## File tree

### Mscripts
    E:.
    │  demo*.m 				# demo文件，用于
    │  example_script.m		# 生成各种预测子	
    │  readme.md 			# 
    │  step_01.m 			# 挑选通信模型的预测子
    │  step_02.m 			# 使用预测子（euc, cmc, navplMS）和SC预测FC
    │  step_02_2.m 			# 统计模型，spearman 相关
    │  step_03.m 			# 可视化，map on brain
    │  step_04.m 			# Unimodal-Heteromodal map
    │  step_05.m 			# 结构/功能梯度分析
    │  step_06.m 			# pattern，方法的相似性
    │  step_07.m 			# developmental pattern [GAM analysis]
    │
    └─fcn
        │  diffusion_maps.m     # 功能梯度计算函数
        │  myboxplot.m          # boxplot
        │  offsetAxes.m         # xy轴分离
        │  Violin.m             
        │  violinplot.m
        │
        └─fcn                   # 计算图论参数

### DATA
- File path: [./DATA/ENVS]
- stps.mat <Stimulating Parent Scale>
    - VariableNames: ['participant_id','interview_age','score','idx_in_feature_dataset']
    - data_FT: [331*4]
    - data_PT1: [92*4]
>>>>>>> 0b398be6c1e52edfa4e8c764a541a3b1173aa15b
    - data_PT2: [83*4]