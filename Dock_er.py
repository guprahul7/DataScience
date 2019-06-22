

#Vagrant
#Tt works on my machine but it doesn't work on other one.
#Use of Virtual Machine - will implement whatever necessary, different boxes...we dump our code into each box

#DOCKER
#Docker runs off of containers
#Docker file builds --> Docker image (which contains: all project code, installation(s) of node.js etc. basically complete application)
#It's not a machine, but this image is designed to sit on top of a machine, 
#From that image we can run as many containers on that machine - depending on processing power, RAM
#Docker image runs as a container - and we can run many many containers 

#Instead of Vagrant, where we put project code INTO an environment, 
#with docker we are going to build our enviroment, and run that environment anywhere.
#Then you can push your docker image to docker-hub/quay.io repository - kinda like github

#Docker has the environment by itself so it will run on any machine
#Self contained 
#You build your container, and push it out nad can run it anywhere
#Really Awesome for Cloud Computing, and CLuster Computing
#