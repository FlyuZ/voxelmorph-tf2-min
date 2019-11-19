根据kaggle[1]，写的一个无监督的小Test，省去无关代码，方便刚入手voxelmoroph的同学理清程序逻辑，后续再自行在源代码上修改

环境为tensorflow2.x

源码根据voxelmorph在Github上的源码做修改，只保留运行这个小Test所需要的所有代码，基本只做了删没有改

安装tensorflow2，修改路径后，应该可以直接跑起来并看到结果

提供jupyter与python两个运行文件


首先用两张图像（可以是2d或3d）做输入，分别为Moving Image和Fixed Image，我们的任务就是把moving往fixed上配。  
进入网络后，先获得速度场（velocity field）（也就是VecInt层，在这之前还有UNet、ConV，Sample等层），  
再根据速度场通过积分（integration layer）计算出对应的形变场（deformation field），  
然后对Moving Image做形变获得Moved Image也就是结果，最后输出也是两个图像，Moved Image和deformation field。  
（此库省略了velocity field，unet后接conv层然后就是SpatialTransformer层，详细的看论文）

参考资料（1）：https://www.kaggle.com/adalca/learn2reg  （数据集从这里下载的）  
参考资料（2）：https://www.dazhuanlan.com/2019/08/24/5d612d006c171/  （讲的是NeurIPS 2019那篇）