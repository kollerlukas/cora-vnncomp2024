#!/bin/bash

echo post_install.sh running..

# -------------------------------------------------------------------------
# SETTINGS

USER_NAME=enp0s5
LICENSE_URL='https://drive.google.com/uc?export=download&id=<file-id>'

MATLAB_RELEASE=2024a
EXISTING_MATLAB_LOCATION=$(dirname $(dirname $(readlink -f $(which matlab))))
ADDITIONAL_PRODUCTS="Symbolic_Math_Toolbox Optimization_Toolbox Statistics_and_Machine_Learning_Toolbox Deep_Learning_Toolbox Deep_Learning_Toolbox_Converter_for_ONNX_Model_Format Parallel_Computing_Toolbox"

CURR_DIR=$(pwd)

# -------------------------------------------------------------------------
# ECHO

echo ${USER_NAME}
echo ${LICENSE_URL}
echo ${EXISTING_MATLAB_LOCATION}
echo ${CURR_DIR}
ls -al

# -------------------------------------------------------------------------
# INITIAL GENERAL INSTALLATION

# check if everything is up to date
export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends --yes \
    wget \
    unzip \
    ca-certificates \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*
	
# -------------------------------------------------------------------------
# MATLAB PACKAGE INSTALLATION

wget -q https://www.mathworks.com/mpm/glnxa64/mpm \
    && chmod +x mpm \
    && sudo HOME=${HOME} ./mpm install \
        --destination=${EXISTING_MATLAB_LOCATION} \
        --release=r${MATLAB_RELEASE} \
        --products ${ADDITIONAL_PRODUCTS}	
	
# -------------------------------------------------------------------------
# CORA INSTALLATION

# download license file
curl --retry 100 --retry-connrefused  -L ${LICENSE_URL} -o license.lic

ls -al

# copy to license folder and delete other license info
cp -f license.lic "${EXISTING_MATLAB_LOCATION}/licenses"
# cp -f license.lic "/root/.matlab/R${MATLAB_RELEASE}_licenses"
# rm "${EXISTING_MATLAB_LOCATION}/licenses/license_info.xml"

# run installCORA non-interactively; requires write access to startup
matlab -nodisplay -r "cd ${CURR_DIR}; which startup"
# sudo chmod 777 /home/matlab/Documents/MATLAB/startup.m
matlab -nodisplay -r "cd ${CURR_DIR}; installCORA(false,true,'/home/matlab'); savepath"

# reading ONNX networks within docker can cause exceptions
# due to some gui issue (see neuralNetwork/readONNXNetwork)
# fixing it on-the-fly requires writing permission
matlab -nodisplay -r "cd ${CURR_DIR}; which +nnet/+internal/+cnn/+onnx/+fcn/ModelTranslation.m"
sudo chmod 777 /home/matlab/Documents/MATLAB/SupportPackages/R$MATLAB_RELEASE/toolbox/nnet/supportpackages/onnx/+nnet/+internal/+cnn/+onnx/+fcn/ModelTranslation.m \
   && sudo chmod 777 /home/matlab/Documents/MATLAB/SupportPackages/R$MATLAB_RELEASE/toolbox/nnet/supportpackages/onnx/+nnet/+internal/+cnn/+onnx/CustomLayerManager.m

# install CORA
# matlab -nodisplay -r "cd /home/ubuntu/toolkit/code; addpath(genpath('.')); installCORA(false,true,'/home/ubuntu/toolkit/code');"

# -------------------------------------------------------------------------
# DONE
echo post_install.sh done
