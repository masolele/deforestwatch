# 🌲 deforestwatch

**AI-powered land use monitoring tool following deforestation**  
Using Earth observation data + deep learning to map what comes *after* the forest is gone.

---

## 🌍 About

**deforestwatch** is an open-source tool for monitoring land use change in areas affected by deforestation. It combines:
- Satellite imagery from **Sentinel-1**, **Sentinel-2**, and **DEM**
- Region-specific deep learning models (UNet variants)
- Forest loss detection from **Hansen Global Forest Change**
- Interactive user interface via **Streamlit**

---

## 🚀 Features

- 🖼️ Draw or upload a Region of Interest (ROI)
- 🧠 Automatically selects AI model based on location (Africa, Asia, Latin America)
- 🛰️ Preprocesses Sentinel-1 + Sentinel-2 + elevation + indices
- 🌾 Predicts land use categories over deforested areas only
- 🗺️ Side-by-side map of RGB imagery + land use classification
- 📤 Export predictions as GeoTIFF
- ☁️ Built on [Google Earth Engine](https://earthengine.google.com/) and [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🔗 Live Demo

👉 [Launch the App on Streamlit](https://your-username-deforestwatch.streamlit.app)  
*(To be replaced with the actual Streamlit Cloud link)*

---

## 🧠 Model Regions

The app supports three specialized models trained on region-specific post-deforestation land use patterns:

| Region        | Model                         |
|---------------|-------------------------------|
| **Africa**    | `sentAfrica.hdf5`             |
| **Asia**      | `sentAsia.hdf5`               |
| **Latin America** | `sentLatinAmerica.hdf5`   |

---

## 📂 Folder Structure
deforestwatch/
├── app.py ← Main Streamlit app
├── preprocess_utils.py ← Sentinel pre-processing + indices
├── model_utils.py ← Auto region detection & model loading
├── Unet_RES_Att_models_IV.py← Custom UNet model architecture
├── models/ ← Pretrained region-specific models
├── requirements.txt
└── .streamlit/


---

## 🛠️ How to Run Locally

1. Clone the repo  
   `git clone https://github.com/masolele/deforestwatch.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Authenticate with Earth Engine  
   `earthengine authenticate`

4. Run the app  
   `streamlit run app.py`

---

## 📡 Data Sources

- **[Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR)**
- **[Sentinel-1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)**
- **[SRTM DEM](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)**
- **[Hansen Forest Loss](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2023_v1_11)**

---

## 📖 License

This project is open-source under the [MIT License](LICENSE).  
Model weights provided for research and non-commercial use.

---

## 🤝 Acknowledgments

Built using:
- [Google Earth Engine](https://earthengine.google.com/)
- [Streamlit](https://streamlit.io/)
- [TensorFlow / Keras](https://www.tensorflow.org/)

---

## 📬 Contact

Created by [@masolele](https://github.com/masolele)  
Need help setting up your own region or models? Open an issue or start a discussion.

---


