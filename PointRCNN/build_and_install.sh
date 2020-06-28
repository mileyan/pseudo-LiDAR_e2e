cd pointnet2_lib/pointnet2
rm -rf build
python setup.py install
cd ../../

cd lib/utils/iou3d/
rm -rf build
python setup.py install

cd ../roipool3d/
rm -rf build
python setup.py install

cd ../simple_roipool3d/
rm -rf build
python setup.py install

cd ../../../tools
