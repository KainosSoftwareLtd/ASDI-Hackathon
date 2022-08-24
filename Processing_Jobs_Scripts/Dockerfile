#processing job Sagemaker
#https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html

#ZScalar SSL Certificate issues
#https://kainossoftwareltd.sharepoint.com/systems/SitePages/Docker---ZScaler-Fixes.aspx
#if need to add certificates to Docker image, we don't want to then uploads this to the cloud as potential security threat
#typical workaround is building image in CI/CD pipeline instead where there are no ssl certificate errors

#Dockerfile - blueprint for building images
#Image - template for running containers
#Container - the actual running process where we have packaged project

#Order is important

#Specifying the base image
FROM python:3.10
#here the dockerfile is pulling the python 3.10 from docker hub which already has python installed so we have all the things we need to have python in our container.

#add script to '.', i.e. current directory
ADD Scripts/main.py .
#Here we added the python file that we want to run in docker and define its location.

# NOTE:
# If you use requirements.txt, then everytime you rebuild your container, it will have to download your requirements again. 
# This is because the results of the command aren't cached because they're after the add command. This could bake rebuilding take quite some time.
# If you install them individually, they will be cached. Rebuilds will be fast. 
# However, if requirements.txt changes and you don't notice, you'll have the wrong packages installed.
# The “Combo” solution is to do both. Install them all individually, then do an install from requirements.txt. 
# If all the packages are installed before installing from requirements.txt, then that part will go quickly. 
# If there is a new package in the file that you missed, then it will still be installed and you'll know you have to update your container. 
# It will be slightly slower until you update, but still faster than installing all packages that way, and your application will still work.

# #pip3 install -r requirements.txt only runs if requirements.txt changes
# COPY requirements.txt /ASDI-Hackathon/requirements.txt
# #set working directory to...
# WORKDIR /ASDI-Hackathon/Scripts
# RUN pip3 install -r requirements.txt
# #Here we installed the dependencies from requirements.txt
# #or e.g. 'pip3 install pandas numpy tqdm'
# COPY . /ASDI-Hackathon/Scripts

ADD requirements.txt /
RUN pip install -r requirements.txt

CMD [ "python3", "Scripts/main.py" ]
#Lastly we specified the entry command this line is simply running python ./create_final_df.py in our container terminal

#DOES NOT WORK macos so easily, need to use the dedicated Docker app, have it open, then run again
#build Docker image and name it in current directory ('.')
#make sure you cd into the directory Dockerfile is in
#-t = tag or name of docker image
#then run in terminal 'docker build -t asdi-pipeline . '
#or run build command in a parent directory (need to cd if not already there) and specify path to Dockerfile with -f
#this switches context of Docker to parent directory
#e.g. 'docker build -t asdi-pipeline -f Scripts/Dockerfile .'

#to run container, run the image 'docker run create_final_df'