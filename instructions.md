# Wine Wuality Prediction

### Phase 1: Pre-requisites
get the AWS account, credentials and pem file ready (same as last assignment)

### Phase 2: Copy data to S3
We will copy all the datasets given and also the python scripts to s3 in our bucket

### Phase 3: Establish an EMR Cluster
Following the Final_Guide.pdf we will create the EMR cluster and give sufficient permissions to ssh into master node

### Phase 4: Configuring Master Node
login to the master node
```
ssh -i /location/for/your_key.pem hadoop@master_node_public_dns
```
configure the aws credentials (similar to last assignment)
```agsl
mkdir ~/.aws
```
```
nano ~/.aws/credentials
```
```
[default]
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
aws_session_token=YOUR_AWS_SESSION_TOKEN
```
```
ctrl+X , Y, Enter
```
to save the file

### Phase 5: Execute Training Code via Spark-submit
We will get all the files from s3
``` 
aws s3 sync s3://myprobucket001/ /path/to/files
```
We will run the training code using spark-submit
```
spark-submit --master yarn /path/to/files/training.py
```
- Spark will begin, perform model training in parallel using the core nodes and store the optimally trained model in the S3 bucket - s3://myprobucket001/saved_lr_model

### Phase 6: Docker Image creation
1. we will login to docker from terminal
```
docker login
```
we will build the docker image with the tag in the format of dockerhub image
```
docker build -t <username>/wine-pred .
```
here <username> is the docker hub user name.

we will Upload the image to Docker Hub
```
docker push  <username>/wine-pred
```

### Phase 7: Perform Prediction using docker
we will launch a EC2 machine, ssh into it and configure the aws credentials.

We will ssh into the machine, install ans start docker
```
sudo yum install docker
sudo service docker start
```
we will pull the docker image 
```
docker pull  <username>/wine-pred
```
we will run the docker image for prediction on the ValidationDataset.csv
```
docker run <username>/wine-pred
```
If we have another test file for testing (Testdataset.csv), we will mount the directory having the file to docker image and run it with this file as argument
```
sudo docker run -v $(pwd):/files <username>/wine-pred spark-submit inference.py /files/TrainingDataset.csv
```

### Phase 8: Perform Prediction without docker
we will launch a EC2 machine, ssh into it and configure the aws credentials and install spark and pyspark into that machine.

We will copy all the required files and the saved model using ```aws s3 sync''' ( similar to above).

nowm, we will run the spark submit command to run the inference script for prediction
```
spark-submit inference.py testDataset.csv
```

