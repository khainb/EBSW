SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH
gdown https://drive.google.com/uc?id=1BIgSWmYfJov7ov0p3v7zq1QUVlZvpWr8
unzip shapenet_chair.zip
rm shapenet_chair.zip