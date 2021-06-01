# Expriement Manager

We utilize Sacred for managing experiments (both training and testing). 

If the users want to re-train PEMP, we provide two Sacred observers for recording the experiments. 

# 1. MongoObserver

### 1.1 MongoDB + Omniboard (Recommended)

*   Create database directory

```bash
mkdir -p /data/db
```

#### 1.1.1 Install with Docker (Recommended)

*   Install Docker following the [instruction](https://docs.docker.com/engine/install/ubuntu/).
*   Run following commands.

```bash
#-------------------------------------- Install MongoDB ---------------------------------------
# Pull the image
docker pull mongo:4.4.1-bionic

# Create network (for Omniboard connection)
docker network create sacred

# Start a container.
docker run -p 7000:27017 -v /data/db:/data/db --name sacred_mongo --network sacred -d mongo:4.4.1-bionic

#-------------------------------------- Install Omniboard ---------------------------------------
# Pull the image
docker pull vivekratnavel/omniboard

# Start a container. Notice that the `PEMP` is the experiment name, which must match the gloabl variable defined in `./entry/*.py` files.
docker run -p 17001:9000 --name omniboard --network sacred -d vivekratnavel/omniboard -m sacred_mongo:27017:PEMP
```

*   Open the address http://localhost:17001 

#### 1.1.2 Manually Install

*   Download a proper version of [MongoDB](https://www.mongodb.com/download-center/community), and install with following commands.

```bash
curl -O https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu1804-4.4.4.tgz
tar -zxvf mongodb-linux-x86_64-ubuntu1804-4.4.4.tgz

# Replace <path_to_mongodb> to the absolute path of mongodb
mv  mongodb-linux-x86_64-ubuntu1804-4.4.4 mongodb
export PATH=/<path_to_mongodb>/bin:$PATH

# Start MongoDB
mongod --dbpath /data/db --port 27107
```

*   Install Nodejs

```bash
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.34.0/install.sh | bash
source　~/.bashrc
nvm install node
```

*   Install Omniboard using npm

```bash
npm install -g omniboard

# Start Omniboard
export PORT=17001
omniboard -m localhost:27107:PEMP
```

*   Open the address http://localhost:17001 



# 2. FileStorageObserver

*   Record experiment details into files. 
*   No UI tools.

