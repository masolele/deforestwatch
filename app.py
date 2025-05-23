import streamlit as st
import ee
import geemap
from datetime import date
import geemap.foliumap as geemap
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_origin
#from model_utils import get_region_from_roi, load_region_model
from preprocess_utils import preprocess_planet
from huggingface_hub import hf_hub_download
import os
import json
from google.oauth2 import service_account
from streamlit_folium import st_folium
import folium
from Unet_RES_Att_models_IV import Attention_UNetFusion3I, Attention_UNetFusion3I_Sentinel

# Authenticate Earth Engine (Streamlit Cloud will use secrets.toml)
# service_account = st.secrets["earthengine"]["EE_SERVICE_ACCOUNT"]
# private_key = st.secrets["earthengine"]["EE_PRIVATE_KEY"]
# credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
# ee.Initialize(credentials)

# Earth Engine initialization with fail-safe
#########
# SERVICE_ACCOUNT_KEY = 'landuseJSON/land-use-292522-392b955456aa.json'
# ee.Initialize(ee.ServiceAccountCredentials(None, SERVICE_ACCOUNT_KEY))
####
json_data = st.secrets["json_data"]

# Preparing values
json_object = json.loads(json_data, strict=False)
service_account = json_object['client_email']
json_object = json.dumps(json_object)

# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

from model_utils import get_region_from_roi, load_region_model

st.set_page_config(layout="wide")
st.title("🌍 Deforestation Land Use Prediction App")

# Add date selectors to sidebar
st.sidebar.markdown("### 🗓️ Select Image Date Range")
start_date = st.sidebar.date_input("Start date", value=date(2024, 1, 1))
end_date = st.sidebar.date_input("End date", value=date(2024, 12, 30))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()


# Upload or draw ROI
st.sidebar.subheader("Upload ROI")
uploaded_file = st.sidebar.file_uploader("Upload GeoJSON or zipped Shapefile", type=["geojson", "zip"])
roi = None

if uploaded_file:
    import geopandas as gpd
    import tempfile
    import zipfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.endswith(".geojson"):
            gdf = gpd.read_file(file_path)
        elif uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            gdf = gpd.read_file(tmpdir)

        import geemap
        roi = geemap.geopandas_to_ee(gdf)

# Fallback to draw ROI
# if roi is None:
#     st.subheader("Or draw ROI on the map")
#     m = geemap.Map()
#     m.add_basemap("SATELLITE")
#     m.add_draw_control()
#     m.to_streamlit(height=500)
#     roi = m.user_roi

st.subheader("Or draw ROI on the map")

# Create map centered on Africa
m = folium.Map(location=[0, 20], zoom_start=3)
draw = folium.plugins.Draw(export=True)
draw.add_to(m)

output = st_folium(m, height=500, width=700)

roi = None
if output and "all_drawings" in output and output["all_drawings"]:
    from shapely.geometry import shape
    import geopandas as gpd

    geojson_geom = output["all_drawings"][-1]["geometry"]
    shp = shape(geojson_geom)
    gdf = gpd.GeoDataFrame(index=[0], geometry=[shp], crs="EPSG:4326")

    import geemap
    roi_fc = geemap.geopandas_to_ee(gdf)
    roi = roi_fc.geometry()

if roi:
    region = get_region_from_roi(roi)
    if not region:
        st.error("ROI not in supported regions (Africa, Asia, Latin America).")
        st.stop()
    st.success(f"Detected region: {region}")

    if st.button("Run Prediction"):
        with st.spinner("Processing and predicting..."):
            x_img = preprocess_planet(roi, str(start_date), str(end_date))  # Should return shape (H, W, 17)
            model = load_region_model(region)
            input_tensor = np.expand_dims(x_img, axis=0)
            pred = model.predict(input_tensor)[0]
            pred_classes = np.argmax(pred, axis=-1).astype(np.uint8)

            # Color map
            # color_map = {
            #    0: (153, 153, 153), 1: (255, 215, 0), 2: (34, 139, 34),
            #    3: (31, 120, 180), 4: (176, 139, 109), 5: (227, 168, 87), 6: (204, 204, 204)
            #}

            color_map = {1: (255, 235, 190),    # Large-scale cropland
                         2: (227, 168, 87),     # Pasture
                         3: (150, 75, 0),       # Mining
                         4: (255, 204, 153),    # Small-scale cropland
                         5: (128, 128, 128),    # Roads
                         6: (140, 198, 63),     # Other-land with tree cover
                         7: (0, 100, 0),        # Plantation forest
                         8: (111, 64, 152),     # Coffee
                         9: (153, 153, 153),    # Built-up
                         10: (31, 120, 180),     # Water
                         11: (255, 102, 102),   # Oil palm
                         12: (5, 136, 179),     # Rubber
                         13: (128, 64, 0),      # Cacao
                         14: (204, 255, 204),   # Avocado
                         15: (255, 153, 51),    # Soy
                         16: (255, 255, 102),   # Sugar
                         17: (255, 229, 204),   # Maize
                         18: (255, 255, 153),   # Banana
                         19: (255, 218, 185),   # Pineapple
                         20: (230, 230, 250),   # Rice
                         21: (160, 82, 45),     # Logging
                         22: (255, 204, 229),   # Cashew
                         23: (186, 85, 211),    # Tea
                         24: (102, 153, 153),   # Others
                         #24: (192, 192, 192),   # Unknown / fallback
                        }

            rgb_image = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
            for cls, rgb in color_map.items():
                rgb_image[pred_classes == cls] = rgb

            st.image(rgb_image, caption="Predicted Land Use (Masked)", use_column_width=True)

            if st.button("Export GeoTIFF"):
                transform = from_origin(-180, 90, 0.01, 0.01)  # You can adjust this
                with rasterio.open(
                    "prediction.tif", "w",
                    driver="GTiff",
                    height=pred_classes.shape[0],
                    width=pred_classes.shape[1],
                    count=1,
                    dtype=rasterio.uint8,
                    crs="+proj=latlong",
                    transform=transform
                ) as dst:
                    dst.write(pred_classes, 1)
                with open("prediction.tif", "rb") as f:
                    st.download_button("Download GeoTIFF", f, "prediction.tif", "image/tiff")
