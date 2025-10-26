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
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb
from shapely.geometry import box, mapping



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
start_date = st.sidebar.date_input("Start date", value=date(2021, 1, 1))
end_date = st.sidebar.date_input("End date", value=date(2021, 12, 31))

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
#loss = forest_loss#.max()

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
    
    forest_loss = forest_loss.resample('bilinear').reproject(crs='EPSG:4326', scale=10)
    loss_dict = forest_loss.clip(roi.bounds())
    loss = np.array(loss_dict.get('lossyear').getInfo())

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
                x_img.astype(np.float32),
                window_size=patch_size,
                subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                nb_classes=n_classes,
                pred_func=(
                    lambda img_batch_subdiv: model.predict_on_batch(img_batch_subdiv)
                )
            )
            #pred = predictor.predict_large_image(x_img)
            pred_classes = np.argmax(pred, axis=-1)#.astype(np.uint8)
            #if loss:
                #mask = loss > 0
                #pred_classes = np.where(mask, pred_classes, np.nan)
            #print("Prediction completed. Output shape:", pred_classes.shape)
            print(np.unique(pred_classes[~np.isnan(pred_classes)], return_counts=True))

            # Color map
            # color_map = {
            #    0: (153, 153, 153), 1: (255, 215, 0), 2: (34, 139, 34),
            #    3: (31, 120, 180), 4: (176, 139, 109), 5: (227, 168, 87), 6: (204, 204, 204)
            #}

            # Hex color map
            hex_color_map = {
                0: "#FFFFFF",
                1: "#C9A0DC",
                2: "#E3A857",
                3: "#FFBABA",
                4: "#B284BE",
                5: "#FF7B7B",
                6: "#9EBD6E",
                7: "#4E7C4E",
                8: "#A50B5E",
                9: "#FF5252",
                10: "#1F78B4",
                11: "#FF00A9",
                12: "#5D9CEC",
                13: "#00FFFF",
                14: "#E0F3DB",
                15: "#FF9F1C",
                16: "#FED976",
                17: "#FFFFBE",
                18: "#FFFF99",
                19: "#FFDAB9",
                20: "#E6E6FA",
                21: "#A0522D",
                22: "#FFCCE5",
                23: "#BA55D3",
                24: "#669999"
            }


            # Map predicted classes to hex color codes
            hex_image = np.full(pred_classes.shape, "#000000", dtype=object)  # Default to black
            for cls, hex_color in hex_color_map.items():
                hex_image[pred_classes == cls] = hex_color
            
            # Convert hex to RGB for imshow (needed since imshow requires RGB values)
            rgb_image = np.array([[mcolors.to_rgb(c) for c in row] for row in hex_image])
            
            # Plot with matplotlib
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb_image)
            ax.set_title("Predicted Land Use Map")
            ax.axis('off')
            
            # Define class labels
            class_labels = {
                0: "Background", 1: "Large-scale cropland", 2: "Pasture", 3: "Mining", 4: "Small-scale cropland",
                5: "Roads", 6: "Other-land with tree cover", 7: "Plantation forest", 8: "Coffee", 9: "Built-up",
                10: "Water", 11: "Oil palm", 12: "Rubber", 13: "Cacao", 14: "Avocado", 15: "Soy", 16: "Sugar",
                17: "Maize", 18: "Banana", 19: "Pineapple", 20: "Rice", 21: "Logging", 22: "Cashew", 23: "Tea", 24: "Others"
            }
            
            # Create legend from hex colors
            patches = [mpatches.Patch(color=hex_color_map[c], label=class_labels[c]) for c in sorted(hex_color_map)]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            # Display in Streamlit
            st.pyplot(fig)


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
