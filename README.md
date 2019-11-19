# spark-movie-recommendation
四缺一斯国一

## 部署文档

### 前端

`git clone https://github.com/anonymz233/douban_movie_vue_front_end.git`

安装 nodejs / npm / yarn

`npm install` 安装依赖

`npm run build` 编译

修改 `nginx/default.conf` 中的后端地址

`sh run.sh` 启动 docker

### 后端

`git clone https://github.com/anonymz233/flask_douban_moive_web.git`

在工程根目录下添加 .env 并至少填入以下配置参数，其中数据库的相关参数需调整

```
REDIS_HOST=39.98.136.173
REDIS_PASSWORD=hehebugaosuni

MONGODB_DB=movie
MONGODB_HOST=39.98.136.173
MONGODB_PORT=9089
MONGODB_USERNAME=user
MONGODB_PASSWORD=hehebugaosuni

FLASK_ENV=development
```

在 `docker-compose.yml` 的第 24 行处可以修改后端监听的端口号

在根目录下执行 `docker-compose -f docker-compose.yml -d up` 启动工程

### 数据库

#### MongoDB

将mongodata解压

`docker run -v <mongodata目录>:/data/db -p 9089:27017 -itd --name mongo mongo mongod --auth`

#### Redis

`docker-compose -f redis.yml up -d`

## Spark

### Foreplay

开放7077-7087，8080-8090，9089-9099端口

在运行 `docker run` 命令之前请务必更改主机ssh默认端口，22端口将被转发至容器内

###  在1服务器上运行

```
cat server1.tar | docker import - master
docker run -itd \
-h 'master' \
--name master \
--add-host=slave1:<slave1 IP> \
--add-host=slave2:<slave2 IP> \
--add-host=slave3:<slave3 IP> \
-p 22:22 -p 7077-7087:7077-7087 -p 8080-8090:8080-8090 \
master2 \
bash
```

### 在2服务器上启动spark-hadoop镜像

```
cat server2-4.tar | docker import - slave
docker run -itd \
-h 'slave1' \
--add-host=master:<master IP> \
--add-host=slave1:<slave1 IP> \
--add-host=slave2:<slave2 IP> \
--add-host=slave3:<slave3 IP> \
--name slave \
-p 22:22 -p 7077-7087:7077-7087 -p 8080-8090:8080-8090 \
slave \
bash
```

### 在3号服务器上启动spark-hadoop镜像

```
cat server2-4.tar | docker import - slave
docker run -itd \
-h 'slave2' \
--add-host=master:<master IP> \
--add-host=slave1:<slave1 IP> \
--add-host=slave2:<slave2 IP> \
--add-host=slave3:<slave3 IP> \
--name slave \
-p 22:22 -p 7077-7087:7077-7087 -p 8080-8090:8080-8090 \
slave \
bash
```

### 在4号服务器上启动spark-hadoop镜像

```
cat server2-4.tar | docker import - slave
docker run -itd \
-h 'slave3' \
--add-host=master:<master IP> \
--add-host=slave1:<slave1 IP> \
--add-host=slave2:<slave2 IP> \
--add-host=slave3:<slave3 IP> \
--name slave \
-p 22:22 -p 7077-7087:7077-7087 -p 8080-8090:8080-8090 \
slave \
bash
```

### 在1号服务器上启动spark-hadoop集群

```
start-yarn.sh
start-dfs.sh
/usr/local/spark/sbin/start-master.sh
sbin/start-slaves.sh
```
