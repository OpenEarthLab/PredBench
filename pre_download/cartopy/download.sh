#!/usr/bin/env bash

# https://github.com/nvkelso/natural-earth-vector/tree/master
# downloaded from: https://www.naturalearthdata.com/downloads/
# Cultural vectors: https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
# Physical vectors: https://www.naturalearthdata.com/downloads/110m-physical-vectors/

mkdir pre_download/cartopy/shapefiles
mkdir pre_download/cartopy/shapefiles/natural_earth
mkdir pre_download/cartopy/shapefiles/natural_earth/cultural
mkdir pre_download/cartopy/shapefiles/natural_earth/physical

wget -c "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_boundary_lines_land.zip" \
    -O pre_download/cartopy/shapefiles/natural_earth/cultural/ne_110m_admin_0_boundary_lines_land.zip
unzip -d pre_download/cartopy/shapefiles/natural_earth/cultural/ \
    pre_download/cartopy/shapefiles/natural_earth/cultural/ne_110m_admin_0_boundary_lines_land.zip 
rm pre_download/cartopy/shapefiles/natural_earth/cultural/ne_110m_admin_0_boundary_lines_land.zip


wget -c https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip  \
    -O pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_coastline.zip
unzip -d pre_download/cartopy/shapefiles/natural_earth/physical/ \
    pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_coastline.zip
rm pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_coastline.zip


wget -c https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip \
    -O pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_land.zip
unzip -d pre_download/cartopy/shapefiles/natural_earth/physical/ \
    pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_land.zip
rm pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_land.zip


wget -c https://naciscdn.org/naturalearth/110m/physical/ne_110m_ocean.zip \
    -O pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_ocean.zip
unzip -d pre_download/cartopy/shapefiles/natural_earth/physical/ \
    pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_ocean.zip
rm pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_ocean.zip


wget -c https://naciscdn.org/naturalearth/110m/physical/ne_110m_rivers_lake_centerlines.zip \
    -O pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_rivers_lake_centerlines.zip
unzip -d pre_download/cartopy/shapefiles/natural_earth/physical/ \
    pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_rivers_lake_centerlines.zip
rm pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_rivers_lake_centerlines.zip


wget -c https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip \
    -O pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_lakes.zip
unzip -d pre_download/cartopy/shapefiles/natural_earth/physical/ \
    pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_lakes.zip
rm pre_download/cartopy/shapefiles/natural_earth/physical/ne_110m_lakes.zip

