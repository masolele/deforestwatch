import ee
import geemap
import os
import rasterio
from rasterio.merge import merge
import numpy as np

def chunked_export(image, roi, output_path, scale=10, max_size_mb=50):
    """
    Export large images by splitting into chunks < max_size_mb
    Args:
        image: ee.Image to export
        roi: ee.Geometry region of interest
        output_path: Final output file path
        scale: Resolution in meters
        max_size_mb: Maximum chunk size in MB (default 50MB for EE)
    """
    # Create temp directory
    os.makedirs('temp_chunks', exist_ok=True)
    
    # 1. Calculate optimal grid division
    bounds = roi.bounds().getInfo()['coordinates'][0]
    min_x = min(p[0] for p in bounds)
    max_x = max(p[0] for p in bounds)
    min_y = min(p[1] for p in bounds)
    max_y = max(p[1] for p in bounds)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Estimate pixel count that would stay under size limit
    pixel_size_bytes = 2  # Assuming 16-bit integers (int16)
    #pixel_size_bytes = 4  # Assuming 32-bit floats
    max_pixels = (max_size_mb * 1024 * 1024) / (pixel_size_bytes * image.bandNames().size().getInfo())
    area_per_chunk = max_pixels * (scale ** 2)
    
    # Calculate grid divisions
    x_chunks = max(1, int(np.ceil(width / np.sqrt(area_per_chunk))))
    y_chunks = max(1, int(np.ceil(height / np.sqrt(area_per_chunk))))
    
    # 2. Export chunks
    chunk_files = []
    for i in range(x_chunks):
        for j in range(y_chunks):
            x_start = min_x + (i * width / x_chunks)
            x_end = min_x + ((i+1) * width / x_chunks)
            y_start = min_y + (j * height / y_chunks)
            y_end = min_y + ((j+1) * height / y_chunks)
            
            chunk_roi = ee.Geometry.Rectangle([x_start, y_start, x_end, y_end])
            chunk_path = f'temp_chunks/chunk_{i}_{j}.tif'
            
            print(f'Exporting chunk {i},{j} ({x_start:.2f},{y_start:.2f})-({x_end:.2f},{y_end:.2f})')
            geemap.ee_export_image(
                image.clip(chunk_roi),
                filename=chunk_path,
                scale=scale,
                region=chunk_roi
            )
            chunk_files.append(chunk_path)
    
    # 3. Mosaic chunks
    print('Mosaicking chunks...')
    src_files = [rasterio.open(f) for f in chunk_files]
    mosaic, transform = merge(src_files)
    
    # Write mosaic
    with rasterio.open(output_path, 'w',
                     driver='GTiff',
                     height=mosaic.shape[1],
                     width=mosaic.shape[2],
                     count=mosaic.shape[0],
                     dtype=mosaic.dtype,
                     crs=src_files[0].crs,
                     transform=transform) as dst:
        dst.write(mosaic)
    
    # Cleanup
    for f in src_files:
        f.close()
    for f in chunk_files:
        os.remove(f)
    os.rmdir('temp_chunks')
    
    print(f'Saved mosaicked image to {output_path}')
