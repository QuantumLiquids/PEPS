# 如何安装本库

本库为头文件库。换句话说，安装本库不需要任何编译。只需要在`git clone`后将`include/`文件夹拷贝到您可以找到的地方即可。
或者，如果你认为这个git仓库你可以找到这个`include/`文件夹。那么你不需要做任何的操作。

如果你想用一种优雅的方式来完成安装，那么cmake提供了标准化的安装方式：
```bash
mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make install
```
如果没有设置cmake选项`CMAKE_INSTALL_PREFIX`，那么类Unix系统默认的编译路径一般在`/usr/local/`下面。
你需要sudo 权限来执行`make install`。否则请指定一个你的帐户有写权限的目录，这也常常是HPC的场景。


如果你不想编译测试，那么这一文档可以结束了。

------
## 编译测试
待写