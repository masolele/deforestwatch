import streamlit as st
import ee
import geemap.foliumap as geemap
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_origin
from model_utils import get_region_from_roi, load_region_model
from preprocess_utils import preprocess_planet

# Authenticate Earth Engine (Streamlit Cloud will use secrets.toml)
service_account = st.secrets["earthengine"]["EE_SERVICE_ACCOUNT"]
private_key = st.secrets["earthengine"]["EE_PRIVATE_KEY"]
credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
ee.Initialize(credentials)

st.set_page_config(layout="wide")
st.title("🌍 Deforestation Land Use Prediction App")

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
if roi is None:
    st.subheader("Or draw ROI on the map")
    m = geemap.Map()
    m.add_basemap("SATELLITE")
    m.add_draw_control()
    m.to_streamlit(height=500)
    roi = m.user_roi

if roi:
    region = get_region_from_roi(roi)
    if not region:
        st.error("ROI not in supported regions (Africa, Asia, Latin America).")
        st.stop()
    st.success(f"Detected region: {region}")

    if st.button("Run Prediction"):
        with st.spinner("Processing and predicting..."):
            x_img = preprocess_planet(roi)  # Should return shape (H, W, 17)
            model = load_region_model(region)
            input_tensor = np.expand_dims(x_img, axis=0)
            pred = model.predict(input_tensor)[0]
            pred_classes = np.argmax(pred, axis=-1).astype(np.uint8)

            # Color map
            color_map = {
                0: (153, 153, 153), 1: (255, 215, 0), 2: (34, 139, 34),
                3: (31, 120, 180), 4: (176, 139, 109), 5: (227, 168, 87), 6: (204, 204, 204)
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
