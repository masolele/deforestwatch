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
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from production_ready_script import LargeImagePredictor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



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
tile = folium.TileLayer(
        tiles = 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
       ).add_to(m)
# Load Hansen Global Forest Loss dataset (2000-2023)
hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')

# Select the 'lossyear' band (forest loss year, 0-23 representing 2000-2023)
forest_loss = hansen.select('lossyear')
loss = forest_loss#.max()

# Define visualization parameters (red color for loss)
vis_params = {
    'min': 1,
    'max': 23,
    'palette': ['FF0000'],  # Red color for loss
    'opacity': 0.7
}
# Get GEE tile URL
map_id = forest_loss.getMapId(vis_params)
tile_url = map_id['tile_fetcher'].url_format

# Add to folium.Map
folium.TileLayer(
    tiles=tile_url,
    attr='Google Earth Engine',
    name='Forest Loss',
    overlay=True,
    control=True
).add_to(m)
# Add the GEE layer to the Folium map
# Add the GEE layer to the map
#m.add_ee_layer(forest_loss, vis_params, "Forest Loss (2000-2023)")
# geemap.add_ee_layer(
#     m,
#     forest_loss,
#     vis_params,
#     'Forest Loss (2000-2023)'
# )
draw = folium.plugins.Draw(export=True)
draw.add_to(m)

output = st_folium(m, height=500, width=700)

patch_size = 64
n_classes = 25
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
            print(x_img.shape, 'image shape')
            model = load_region_model(region)
            #input_tensor = np.expand_dims(x_img, axis=0)
            #pred = model.predict(input_tensor)[0]
            # predictor = LargeImagePredictor(
            #     model,
            #     patch_size=patch_size,
            #     overlap =32,
            #     batch_size = 8
            # )
            pred = predict_img_with_smooth_windowing(
                x_img,
                window_size=patch_size,
                subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                nb_classes=n_classes,
                pred_func=(
                    lambda img_batch_subdiv: model.predict_on_batch(img_batch_subdiv)
                )
            )
            #pred = predictor.predict_large_image(x_img)
            pred_classes = np.argmax(pred, axis=-1).astype(np.uint8)
            if loss:
                pred_classes = np.where(loss == 0, 0, pred_classes)
                pred_classes = np.where(pred_classes == 0, np.nan, pred_classes)
            #print("Prediction completed. Output shape:", pred_classes.shape)
            print(np.unique(pred_classes))

            # Color map
            # color_map = {
            #    0: (153, 153, 153), 1: (255, 215, 0), 2: (34, 139, 34),
            #    3: (31, 120, 180), 4: (176, 139, 109), 5: (227, 168, 87), 6: (204, 204, 204)
            #}

            color_map = {1: (201, 160, 220),    # Large-scale cropland
                         2: (227, 168, 87),     # Pasture
                         3: (255, 186, 186),       # Mining
                         4: (178, 132, 190),    # Small-scale cropland
                         5: (255, 123, 123),    # Roads
                         6: (158, 189, 110),     # Other-land with tree cover
                         7: (78, 124, 78),        # Plantation forest
                         8: (165, 11, 94),     # Coffee
                         9: (255, 82, 82),    # Built-up
                         10: (31, 120, 180),     # Water
                         11: (255, 0, 169),   # Oil palm
                         12: (93, 156, 236),     # Rubber
                         13: (0, 255, 255),      # Cacao
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
            
            # Plot using matplotlib
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb_image)
            ax.set_title("Predicted Land Use Map")
            ax.axis('off')
            
            # Create legend
            class_labels = {
                1: "Large-scale cropland",
                2: "Pasture",
                3: "Mining",
                4: "Small-scale cropland",
                5: "Roads",
                6: "Other-land with tree cover",
                7: "Plantation forest",
                8: "Coffee",
                9: "Built-up",
                10: "Water",
                11: "Oil palm",
                12: "Rubber",
                13: "Cacao",
                14: "Avocado",
                15: "Soy",
                16: "Sugar",
                17: "Maize",
                18: "Banana",
                19: "Pineapple",
                20: "Rice",
                21: "Logging",
                22: "Cashew",
                23: "Tea",
                24: "Others"
            }
            
            patches = [mpatches.Patch(color=np.array(color_map[c])/255.0, label=class_labels[c]) for c in sorted(color_map)]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            st.pyplot(fig)


            #st.image(rgb_image, caption="Predicted Land Use (Masked)", use_container_width=True)


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
