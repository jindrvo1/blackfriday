#!/bin/sh

DIRNAME=$(dirname $(realpath $0))
COMPONENT_DIR=$DIRNAME/components

for DIR in $COMPONENT_DIR/*
do
    if [ -d $DIR ]; then
        BUILDFILE=$DIR/build_component.sh
        COMPONENT=$(basename $DIR)_component

        if [ ! -f $BUILDFILE ]; then
            echo "No build script found for component $COMPONENT"
            continue
        fi

        echo "Building component $COMPONENT"
        $BUILDFILE
        echo "Done building component $COMPONENT"
    fi
done