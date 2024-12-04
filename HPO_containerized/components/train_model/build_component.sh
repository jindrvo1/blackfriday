#!/bin/sh

DIRNAME=$(dirname $(realpath $0))
source $DIRNAME/../../.env

COMPONENT_NAME=$(basename $DIRNAME)_component
TARGET_IMAGE=$KFP_COMPONENT_TARGET_IMAGE_BASE/$COMPONENT_NAME:latest

echo "KFP_BASE_IMAGE=\"$KFP_BASE_IMAGE\"" > $DIRNAME/.env
echo "TARGET_IMAGE=\"$TARGET_IMAGE\"" >> $DIRNAME/.env

kfp component build $DIRNAME --component-filepattern component.py --platform linux/amd64 $@