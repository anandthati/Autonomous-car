#!/bin/bash
clear
ssh -i ~/Downloads/Jenkins.pem ec2-user@ec2-13-126-47-57.ap-south-1.compute.amazonaws.com "sudo yum install -y jq && curl --silent http://169.254.169.254/latest/dynamic/instance-identity/document && curl --silent http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .instanceId"r