SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/7-scenes-redkitchen.zip --no-check-certificate
unzip 7-scenes-redkitchen.zip
rm 7-scenes-redkitchen.zip
mv 7-scenes-redkitchen kitchen

wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1.zip --no-check-certificate
unzip sun3d-home_at-home_at_scan1_2013_jan_1.zip
rm sun3d-home_at-home_at_scan1_2013_jan_1.zip
mv sun3d-home_at-home_at_scan1_2013_jan_1 home1

wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_md-home_md_scan9_2012_sep_30.zip --no-check-certificate
unzip sun3d-home_md-home_md_scan9_2012_sep_30.zip
rm sun3d-home_md-home_md_scan9_2012_sep_30.zip
mv sun3d-home_md-home_md_scan9_2012_sep_30 home2

wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_uc-scan3.zip --no-check-certificate
unzip sun3d-hotel_uc-scan3.zip
rm sun3d-hotel_uc-scan3.zip
mv sun3d-hotel_uc-scan3 hotel1

wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel1.zip --no-check-certificate
unzip sun3d-hotel_umd-maryland_hotel1.zip
rm sun3d-hotel_umd-maryland_hotel1.zip
mv sun3d-hotel_umd-maryland_hotel1 hotel2

wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3.zip --no-check-certificate
unzip sun3d-hotel_umd-maryland_hotel3.zip
rm sun3d-hotel_umd-maryland_hotel3.zip
mv sun3d-hotel_umd-maryland_hotel3 hotel3

wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_76_studyroom-76-1studyroom2.zip --no-check-certificate
unzip sun3d-mit_76_studyroom-76-1studyroom2.zip
rm sun3d-mit_76_studyroom-76-1studyroom2.zip
mv sun3d-mit_76_studyroom-76-1studyroom2 study

wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika.zip --no-check-certificate
unzip sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika.zip
rm sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika.zip
mv sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika lab