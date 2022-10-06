## Docker简介

#### 一、课堂知识点

什么是Docker，docker容器与虚拟机有什么区别

#### 二、Docker的概念

![img](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080654.jpeg)

Docker 是一种开源的虚拟进程技术，基于GO语言进行开发的。

Docker 可以让开发者自己打包我们自己的应用或依赖包到一个轻量级的，可移植的容器中去。然后，再一发布，那么就可以在所有的Linux上进行运行，一次构建，到处使用

Docker容器技术，一定是依托Linux操作系统存在！

镜像产生出来的程序，我们叫“容器”。容器完成使用一种“沙箱”机制。

#### 三、沙箱的概念

沙箱是一种虚拟进程技术，它的特点：沙箱之间相互独立，互不干扰。并且对现有的Linux系统不会产生任何的影响。

1、搭建测试，开发，生产环境

2、发布自己的应用程序

#### 四、使用Docker的原因

1、保证开发，测试，上线环境统一

2、更快速的交付和部署

3、提供比虚拟机更为高效的虚拟技术

4、可以更轻松的迁移或者横向扩展

#### 五、Docker 和Vmware虚拟机的区别

VM 是一个运行在宿主机之上的完整的操作系统，VM运行时 ，需要消耗宿主机大量系统资源，例如：CPU,内存，硬盘等一系列的资源。Docker容器与VM不一样，它只包含了应用程序以及各种依赖库。

正因为它占用系统资源非常少，所以它更加的轻量。它在启动时，可能只需要简单的一个命令就可以了。启动仅仅只需要几秒或几十秒钟就可以完成。对于宿主机来讲，承担VM可能5-10个就已经非常厉害了 ，但是Docker容器很轻松就承担几千个。而且网络配置相对而言也比较简单，主要以桥接方式为主。

<img src="https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080659.png" alt="img" style="zoom:150%;" />



## Docker安装

#### 一、安装Docker

Docker官网https://docs.docker.com/engine/install/centos/有详细安装说明。

![image-20210621110939756](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080719.png)

官网使用的是国外镜像地址，下载慢，可以使用阿里云镜像https://developer.aliyun.com/mirror/。

![image-20210621111224842](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080723.png)

执行如下命令，安装成功。

```linux
# step 1: 安装必要的一些系统工具
yum install -y yum-utils device-mapper-persistent-data lvm2
# Step 2: 添加软件源信息
yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
# Step 3: 更新并安装Docker-CE
yum makecache fast
yum -y install docker-ce
# Step 4: 开启Docker服务
service docker start
```

查看Docker版本，安装成功。

```tex
docker -v
```

![image-20210621112335672](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080731.png)





## Docker阿里云镜像加速器配置

#### 一、配置Docker阿里云镜像加速器

登录阿里云：https://www.aliyun.com/，访问阿里容器服务。

![image-20210621113420658](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080734.png)

查看镜像加速，每个账号会分配自己的镜像加速地址，依次执行下图的命令即可。

![image-20210621113549736](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080738.png)

1. sudo mkdir -p /etc/docker
2. sudo tee /etc/docker/daemon.json <<-'EOF'
   {
    "registry-mirrors": ["![img](file:///C:\Users\于科伟\AppData\Roaming\Tencent\QQTempSys\%W@GJ$ACOF(TYDYECOKVDYB.png)https://pr0bscab.mirror.aliyuncs.com"]
   }
   EOF
3. sudo systemctl daemon-reload
4. sudo systemctl restart docker

## Docker三大核心概念

#### 一、Docker镜像

Docker镜像是由文件系统叠加而成（是一种文件的存储形式）。最底端是一个文件引导系统，即bootfs，这很像典型的Linux/Unix的引导文件系统。Docker用户几乎永远不会和引导系统有什么交互。实际上，当一个容器启动后，它将会被移动到内存中，而引导文件系统则会被卸载，以留出更多的内存供磁盘镜像使用。Docker容器启动是需要的一些文件，而这些文件就可以称为Docker镜像。

<img src="https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080742.png" alt="image-20210621121209160" style="zoom:150%;" />

镜像是构建Docker的基石。用户基于镜像来运行自己的容器。镜像也是Docker生命周期中的“构建”部分。镜像是基于联合文件系统的一种层式结构，由一系列指令一步一步构建出来。例如：添加一个文件；执行一个命令；打开一个窗口。也可以将镜像当作容器的“源代码”。镜像体积很小，非常“便携”，易于分享、存储和更新。

#### 二、Docker容器

Docker可以帮助你构建和部署容器，你只需要把自己的应用程序或者服务打包放进容器即可。容器是基于镜像启动起来的，容器中可以运行一个或多个进程。我们可以认为，镜像是Docker生命周期中的构建或者打包阶段，而容器则是启动或者执行阶段。 容器基于镜像启动，一旦容器启动完成后，我们就可以登录到容器中安装自己需要的软件或者服务。

所以Docker容器就是：

一个镜像格式；

一些列标准操作；

一个执行环境。

Docker借鉴了标准集装箱的概念。标准集装箱将货物运往世界各地，Docker将这个模型运用到自己的设计中，唯一不同的是：集装箱运输货物，而Docker运输软件。

和集装箱一样，Docker在执行上述操作时，并不关心容器中到底装了什么，它不管是web服务器，还是数据库，或者是应用程序服务器什么的。所有的容器都按照相同的方式将内容“装载”进去。

Docker也不关心你要把容器运到何方：我们可以在自己的笔记本中构建容器，上传到Registry，然后下载到一个物理的或者虚拟的服务器来测试，在把容器部署到具体的主机中。像标准集装箱一样，Docker容器方便替换，可以叠加，易于分发，并且尽量通用。

使用Docker，我们可以快速的构建一个应用程序服务器、一个消息总线、一套实用工具、一个持续集成（CI）测试环境或者任意一种应用程序、服务或工具。我们可以在本地构建一个完整的测试环境，也可以为生产或开发快速复制一套复杂的应用程序栈。

#### 三、Docker仓库

专门用于放置镜像的地方，镜像可以从网络上下载，也可以自己去产生。最大的公开仓库是 Docker Hub(https://hub.docker.com/)。



## Docker原理

![image-20210902231732667](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080815.png)

#### 一、课堂知识点

容器转镜像，Dockerfile构建镜像

#### 二、Dockerfile构建镜像

镜像的来源，一般可以有3种途径：从平台上获取，容器逆向生成镜像，以及DockerFile文件构建镜像

##### 1、Dockerfile组成

Dockerfile 分为四部分：基础镜像信息、维护者信息、镜像操作指令和容器启动时执行指令

##### 2、Dockerfile构建Tomcat

```
#设置基础镜像
FROM centos
MAINTAINER 11553176@qq.com
#复制压缩包到镜像中，并进行解压
ADD ./jdk-8u11-linux-x64.tar.gz /usr/local/java
ADD ./apache-tomcat-8.5.20.tar.gz /usr/local/tomcat
#ADD ./woniu.war /usr/local/tomcat/apache-tomcat-8.5.20/webapps 
#set environment variable
ENV JAVA_HOME /usr/local/java/jdk1.8.0_11
ENV JRE_HOME $JAVA_HOME/jre  
ENV CLASSPATH .:$JAVA_HOME/lib:$JRE_HOME/lib  
ENV PATH $PATH:$JAVA_HOME/bin
#EXPOSE 映射端口
EXPOSE 8080
#ENTRYPOINT 配置容器启动时，需要执行的文件
ENTRYPOINT /usr/local/tomcat/apache-tomcat-8.5.20/bin/startup.sh && tail -F /usr/local/tomcat/apache-tomcat-8.5.20/logs/catalina.out
```

##### 3、Dockerfile文件讲解

基础镜像信息

```
#设置基础镜像FROM centos
```

维护者信息

```
MAINTAINER 11553176@qq.com
```

镜像操作指令

```
#复制压缩包到镜像中，并进行解压
ADD ./jdk-8u11-linux-x64.tar.gz /usr/local/java
ADD ./apache-tomcat-8.5.20.tar.gz /usr/local/tomcat
#set environment variable
ENV JAVA_HOME /usr/local/java/jdk1.8.0_11
ENV JRE_HOME $JAVA_HOME/jre  
ENV CLASSPATH .:$JAVA_HOME/lib:$JRE_HOME/lib  
ENV PATH $PATH:$JAVA_HOME/bin
#EXPOSE 映射端口
EXPOSE 8080
```

容器启动时执行指令

```
#ENTRYPOINT 配置容器启动时，需要执行的文件
ENTRYPOINT /usr/local/tomcat/apache-tomcat-8.5.20/bin/startup.sh && tail -F /usr/local/tomcat/apache-tomcat-8.5.20/logs/catalina.out
```

##### 4、Dockerfile命令详细介绍

**FROM :** 指定基础镜像，要在哪个镜像建立

**MAINTAINER：**指定维护者信息

**RUN：**在镜像中要执行的Linux命令

**ADD:** 相当于 COPY，但是比 COPY 功能更强大(如果是压缩包，会自动解压)

**COPY ：**复制本地主机的 （为 Dockerfile 所在目录的相对路径）到容器中的

**ENV：**定义环境变量

**WORKDIR：**指定当前工作目录，相当于 cd ，为后续的 RUN 、 CMD 、 ENTRYPOINT 指令配置工作目录。

**EXPOSE：**指定容器要打开的端口

**VOLUME：**挂载目录，创建一个可以从本地主机或其他容器挂载的挂载点，一般用来存放数据库和需要保持的数据等。

格式为VOLUME [“/data”]

**ENTRYPOINT** 指定容器在运行时，需要执行的文件，或者需要命令的操作的文件

**CMD** 指定容器在运行时，需要运行的命令

##### 5、构建镜像

在具有Dockerfile的目录中，执行如下命令：

```
docker build -t woniu/tomcat:8.5 --rm=true .
```

##### 6、启动Tomcat

```
docker run -d -p 8080:8080 --name woniutomcat woniu/tomcat:8.5
```

![image-20210210192612591](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080828.png)

##### 7、复制文件到Docker容器中

```
docker cp /root/web-demo/boots.war woniutomcat:/usr/local/tomcat/apache-tomcat-8.5.20/webapps
```

#### 三、容器转镜像

Docker除了支持Dockerfile构建镜像之外，还支持从容器逆向生成镜像

##### 1、使用docker commit 提交镜像

```
docker commit -m "根据容器自定义镜像" -a "woniuxy" ee20a94d39e4 woniuxy/tomcat:1.0
docker commit -m "包含了woniu.war的镜像" -a "woniuxy" 556f92e623a1 woniuxy/tomcat:8.5
```

-m 备注消息

-a 作者

ee20a94d39e4 容器的ID

##### 2、检查镜像

```
docker images
```

![image-20210210193209328](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080845.png)

##### 3、开启容器

```
docker run -d -p 8080:8080 --name mytomcat woniuxy/tomcat:8.5
```



## Docker拉取tomcat镜像并部署应用

#### 一、Docker拉取Tomcat镜像

##### 1、下载镜像

docker pull 镜像名

```
docker pull tomcat
```

##### 2、启动容器

```
docker run -d -p 8080:8080 --name mytomcat tomcat
```

##### 3、参数说明

-i：表示运行容器

-t：表示容器启动后会进入其命令行。加入这两个参数后，容器创建就能登录进去。即分配一个伪终端。

—name :为创建的容器命名。

-v：表示目录映射关系（前者是宿主机目录，后者是映射到宿主机上的目录），可以使用多个－v做多个目录或文件映射。注意：最好做目录映射，在宿主机上做修改，然后共享到容器上。

-d：在run后面加上-d参数,则会创建一个守护式容器在后台运行（这样创建容器后不会自动登录容器，如果只加-i -t两个参数，创建后就会自动进去容器）。

-p：表示端口映射，前者是宿主机端口，后者是容器内的映射端口。可以使用多个－p做多个端口映射

##### 4、测试Tomcat容器，正常启动。

![image-20210621151426531](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080901.png)

导入外部的war包。

将任意一个JavaWeb向打成war包，在Linux系统中创建/root/web-demo目录，将war包上传到该目录下（映射，不映射不会访问到war包的内容）。

```Linux
docker run -d -p 8080:8080 --name mytomcat --privileged=true -v /root/web-demo:/usr/local/tomcat/webapps tomcat
```

#### 二、Docker拉取MySQL镜像

##### 1、载镜像

```
docker pull mysql:5.7
```

##### 2、启动容器

```
docker run --name mysql01 -p 3306:3306 --privileged=true -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
```

参数说明：

-e MYSQL_ROOT_PASSWORD= 设置MySQL登录密码



## Docker commit命令重点讲解

#### 一、Docker常用命令

##### 1、查看当前正在运行的容器实例

```
docker ps
```

![image-20210820153149689](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080908.png)

##### 2、查看当前正在运行，以及曾经运行过的容器实例

```
docker ps -a
```

![image-20210820153218478](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080911.png)

##### 3、启动|重启|停止某一个容器

```
docker start|restart|stop [容器的名称/容器的ID]
```

![image-20210820153611306](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080914.png)

##### 4、从docker环境中，移除某一个容器

```
docker rm [容器的名称/容器的ID]
```

##### 5、查看docker的日志输出

```
docker logs [容器的名称/容器的ID]
```

##### 6、从宿主机进入到Docker容器内部

```
docker exec -it [容器的名称/容器的ID] /bin/bash
```

##### 7、查看容器的详细信息

```
docker inspect [容器的名称/容器的ID]
```

##### 8、查看容器的内置IP地址

```
docker inspect --format='{{.NetworkSettings.IPAddress}}' [容器的名称/容器的ID]
```

![image-20210820153923605](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080918.png)

##### 9、删除镜像

```
docker rmi [镜像的名称:版本号/镜像的ID]
```

| docker命令                                        | 命令描述                                   |
| ------------------------------------------------- | ------------------------------------------ |
| docker -v                                         | 查看docker版本信息                         |
| docker info                                       | 查看docker详细信息                         |
| docker --help                                     | 查看docker命令的帮助信息                   |
| docker images                                     | 列出本地仓库中的镜像                       |
| docker search 镜像名字                            | 在远程仓库上查找镜像                       |
| docker pull 镜像名字[:版本]                       | 从远程仓库中拉取镜像                       |
| docker rmi 镜像名字                               | 根据镜像名字删除镜像                       |
| docker save -o 目标文件名字.tar 镜像名字[:版本号] | 导出docker镜像                             |
| docker load -i 压缩包名字.tar                     | 导入docker镜像                             |
| docker run 镜像名字                               | 根据指定镜像创建容器并启动                 |
| docker run -d 容器id                              | 后台运行容器，无需任何交互                 |
| docker ps                                         | 查看正在运行中的容器                       |
| docker ps -a                                      | 查询所有容器，无论是运行中的还是停止运行的 |
| docker start 容器id                               | 启动一个已经停止运行的容器                 |
| docker restart 容器id                             | 重启容器                                   |
| docker stop 容器id                                | 停止容器                                   |
| docker kill 容器id                                | 立即停止容器                               |
| docker rm 容器id                                  | 删除已经停止的容器                         |
| docker rm -f 容器id                               | 删除容器，无论容器是否正在运行             |
| docker logs 容器id                                | 查看容器日志                               |
| docker cp 文件 容器id:容器内路径                  | 将指定文件拷贝到容器中的指定路径           |

| exit                    | 退出容器，同时停止容器               |
| ----------------------- | ------------------------------------ |
| ctrl+P+Q                | 退出容器，但是不停止容器             |
| docker attach 容器id    | 再次进入正在运行的容器               |
| docker exec 容器id 命令 | 在容器外面发送命令给容器，让容器执行 |



## Docker数据卷

docker数据卷

我们已经知道，docker镜像中包含着应用以及运行应用所需的环境。如此，根据docker镜像所产生的容器，也就包含着应用以及运行应用所需的环境。

 问题是：容器在运行中所产生的数据，会随着容器被删除而丢失。

 有两种办法可以把容器运行时所产生的数据永久地保存下来（持久化）。 1.通过docker commit命令来根据容器生成新的镜像; 2.使用docker容器数据卷，简称为卷。 下面我们就详细地学习一下docker容器数据卷的内容！

 卷就是linux中的目录或文件，由docker挂载到容器中，可以把同一个文件同时挂在到多个不同的容器中。设计卷的目的，就是完成容器中数据的持久化操作。卷是完全独立于容器的，不会随着容器的删除而被删除。

 卷能的作用，就是能将容器的数据传递给主机，也能将主机的数据传递到容器中。

 不废话了，直接使用一把数据卷！

docker run -it -v /主机目录:/容器中的目录 镜像名，当执行了以上命令以后，会在主机和容器中，自动创建相应的目录！

此时，无论是修改了主机上的/foo目录中的内容还是容器中的/foo的内容，对方都能感知! 

甚至，在容器关闭后，修改了主机中的/foo目录内容，当再次启动容器时，容器中/foo的内容仍然会与主机上/foo的内容一致！以下是测试时的一些相关命令。

 docker run -it -v /主机目录:/容器内目录:ro 镜像名（ro表示readonly），主机目录中可读可写，容器目录中是只读的



 

## Docker应用

#### 一、部署项目到Docker

##### 1、先准备好SpringBoot项目，打成jar包。

![image-20210621161234620](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080931.png)

##### 2、将项目上传至Linux系统。

##### 3、编写Dockerfile文件。

```
FROM java:8
MAINTAINER 11553176@qq.com
ADD eureka01.jar eureka01.jar
EXPOSE 9090
ENTRYPOINT ["java","-jar","eureka01.jar"]
```

##### 4、生成镜像。

```
docker build -t woniuxy/eureka:1.0
```

##### 5、运行容器。

```
docker run -d --name eureka01 -p 9090:9090 woniuxy/eureka:1.0
```

##### 6、测试浏览器访问，成功。

![image-20210621161450154](https://woniumd.oss-cn-hangzhou.aliyuncs.com/java/panfeng/20210903080947.png)