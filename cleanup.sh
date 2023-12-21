#!/bin/sh

rm temp/*
rm .history/* -rf
rm bad.list
rm predictions.jpg
rm -rf dist/ build/ *.spec
git add . 
sudo docker build -t alex1075/machine-code .
