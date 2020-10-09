#!/bin/bash
# ref: https://github.com/awslabs/ec2-spot-labs/blob/master/ec2-spot-deep-learning-training/user_data_script.sh
# Made a few changes to avoid causing problems with instances not being deleted due to a hang on shutdown. Also added
# the closing of open ports because I would rather not have open ports on an expensive instance.
curl --local-port 4000-4200 "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
apt install unzip -y
unzip awscliv2.zip
./aws/install

INSTANCE_ID=$(curl --local-port 4000-4200 -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl --local-port 4000-4200 -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
AWS_REGION=us-west-2

VOLUME_ID=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=DeepL" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=DeepL" --query "Volumes[].AvailabilityZone" --output text)

if [ $VOLUME_ID ]; then
		# Check if the Volume AZ and the instance AZ are same or different.
		# If they are different, create a snapshot and then create a new volume in the instance's AZ.
		if [ $VOLUME_AZ != $INSTANCE_AZ ]; then
				SNAPSHOT_ID=$(aws ec2 create-snapshot \
						--region $AWS_REGION \
						--volume-id $VOLUME_ID \
						--description "`date +"%D %T"`" \
						--tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=DeepL-snapshot}]' \
						--query SnapshotId --output text)

				aws ec2 wait --region $AWS_REGION snapshot-completed --snapshot-ids $SNAPSHOT_ID
				# Wait before deleting to make sure multiple copies arent created due to volume hang on instance shutdown
				aws ec2 wait volume-available --region $AWS_REGION --volume-id $VOLUME_ID
				aws ec2 --region $AWS_REGION  delete-volume --volume-id $VOLUME_ID

				VOLUME_ID=$(aws ec2 create-volume \
						--region $AWS_REGION \
								--availability-zone $INSTANCE_AZ \
								--snapshot-id $SNAPSHOT_ID \
						--volume-type gp2 \
						--tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=DeepL}]' \
						--query VolumeId --output text)
				aws ec2 wait volume-available --region $AWS_REGION --volume-id $VOLUME_ID
		fi
		# Attach volume to instance
		VOLUME_NAME=$(aws ec2 attach-volume \
			--region $AWS_REGION --volume-id $VOLUME_ID \
			--instance-id $INSTANCE_ID --device /dev/xvdf --query Device --output text)
		sleep 10

		# Mount volume and change ownership, since this script is run as root
		mkdir /Data
		mount $VOLUME_NAME /Data
		chown -R ubuntu: /Data/
		cd /home/ubuntu

		# Get training code
		curl --local-port 4000-4200 "https://raw.githubusercontent.com/fhaltmayer/DeepL/master/ProGan/Progan_aws.py" -o "Progan_aws.py"
		
		# Close up open ports 
		aws ec2  modify-instance-attribute --instance-id $INSTANCE_ID --groups sg-0d197adc3de283346

		# Initiate training using the tensorflow_36 conda environment
		sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate pytorch_latest_p36; python3 Progan_aws.py -1"
		#source activate pytorch_latest_p36; python3 Progan_aws.py -1
fi

# SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $AWS_REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)
# aws ec2 cancel-spot-fleet-requests --region $AWS_REGION --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID --terminate-instances

# sudo aws ec2 attach-volume --volume-id vol-0a664142b86891348 --instance-id $myID --device /dev/xvdh
# sudo lsblk
# sudo mkdir /Data
# sudo mount /dev/xvdh /Data
# sleep 5
# sudo chown -R ubuntu: /Data
# cd ~
# sudo wget https://raw.githubusercontent.com/fhaltmayer/DeepL/master/ProGan/Progan_aws.py
# # sudo source activate pytorch_latest_p36
