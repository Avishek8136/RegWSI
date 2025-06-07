"""
Complete OME-TIFF Registration Pipeline using DeepHistReg

This module provides a complete pipeline for registering OME-TIFF files,
with fixes for OpenCV limitations and large image handling.
"""

### Ecosystem Imports ###
from typing import Union, Optional, Tuple, List, Dict
import pathlib
import shutil
import os
import warnings
warnings.filterwarnings('ignore')

### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
from skimage.transform import resize
import SimpleITK as sitk
import tifffile
import xml.etree.ElementTree as ET

### DeeperHistReg Imports ###
import deeperhistreg
from deeperhistreg.dhr_input_output.dhr_loaders import pil_loader, tiff_loader
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid

# Set OpenCV environment variable to handle large images
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**63 - 1)

def load_ome_tiff(filepath: Union[str, pathlib.Path], 
                  channel: Optional[int] = None,
                  z_slice: Optional[int] = None,
                  series: int = 0,
                  downsample_factor: int = 1) -> Tuple[np.ndarray, dict]:
    """
    Load OME-TIFF file with metadata and optional downsampling
    
    Args:
        filepath: Path to OME-TIFF file
        channel: Specific channel to load (None for all channels)
        z_slice: Specific z-slice to load (None for all slices)
        series: Series index for multi-series files
        downsample_factor: Factor to downsample image (1 = no downsampling)
    
    Returns:
        image array and metadata dictionary
    """
    with tifffile.TiffFile(filepath) as tif:
        # Get metadata
        metadata = {}
        if tif.ome_metadata:
            metadata['ome'] = tif.ome_metadata
        
        # Get image data
        if tif.series:
            series_data = tif.series[series]
            image = series_data.asarray()
            
            # Store shape info
            metadata['original_shape'] = image.shape
            metadata['axes'] = series_data.axes
            
            # Handle specific channel/z-slice selection
            if len(image.shape) > 2:
                # Common axes orders: TZCYX, ZCYX, CYX, TYX
                axes = series_data.axes.upper()
                
                # Extract specific z-slice
                if 'Z' in axes and z_slice is not None:
                    z_idx = axes.index('Z')
                    image = np.take(image, z_slice, axis=z_idx)
                    axes = axes.replace('Z', '')
                elif 'Z' in axes:
                    # Take middle slice if no specific slice requested
                    z_idx = axes.index('Z')
                    z_middle = image.shape[z_idx] // 2
                    image = np.take(image, z_middle, axis=z_idx)
                    axes = axes.replace('Z', '')
                
                # Extract specific channel
                if 'C' in axes and channel is not None:
                    c_idx = axes.index('C')
                    image = np.take(image, channel, axis=c_idx)
                    axes = axes.replace('C', '')
                
                # Remove time dimension if present
                if 'T' in axes:
                    t_idx = axes.index('T')
                    image = np.take(image, 0, axis=t_idx)
                    axes = axes.replace('T', '')
                
                # Ensure we have 2D or 3D (RGB) image
                if len(image.shape) > 3:
                    # Take first slices of remaining dimensions
                    while len(image.shape) > 3:
                        image = image[0]
                
                metadata['processed_axes'] = axes
        else:
            # Simple TIFF
            image = tif.asarray()
        
        # Apply downsampling if requested
        if downsample_factor > 1:
            if len(image.shape) == 2:
                new_shape = (image.shape[0] // downsample_factor, 
                            image.shape[1] // downsample_factor)
            else:
                new_shape = (image.shape[0] // downsample_factor, 
                            image.shape[1] // downsample_factor,
                            image.shape[2])
            
            image = resize(image, new_shape, preserve_range=True, anti_aliasing=True)
            image = image.astype(image.dtype if hasattr(image, 'dtype') else np.uint8)
            metadata['downsampled'] = True
            metadata['downsample_factor'] = downsample_factor
            
        return image, metadata

def save_ome_tiff(image: np.ndarray, 
                  filepath: Union[str, pathlib.Path],
                  metadata: Optional[dict] = None,
                  pixel_size: Optional[Tuple[float, float]] = None,
                  channel_names: Optional[List[str]] = None,
                  preserve_dtype: bool = True):
    """
    Save image as OME-TIFF with metadata
    
    Args:
        image: Image array
        filepath: Output path
        metadata: Optional metadata dictionary
        pixel_size: Pixel size in microns (x, y)
        channel_names: Names for each channel
        preserve_dtype: If True, preserves original data type (e.g., uint16)
    """
    # Prepare metadata
    if pixel_size or channel_names:
        metadata_dict = {
            'axes': 'YXC' if len(image.shape) == 3 else 'YX',
            'PhysicalSizeX': pixel_size[0] if pixel_size else 1.0,
            'PhysicalSizeY': pixel_size[1] if pixel_size else 1.0,
            'PhysicalSizeXUnit': 'µm' if pixel_size else 'pixel',
            'PhysicalSizeYUnit': 'µm' if pixel_size else 'pixel',
        }
        if channel_names and len(image.shape) == 3:
            metadata_dict['Channel'] = {'Name': channel_names}
    else:
        metadata_dict = metadata if metadata else {}
    
    # Determine photometric interpretation based on data
    if len(image.shape) == 3 and image.shape[2] == 3:
        photometric = 'rgb'
    else:
        photometric = 'minisblack'
    
    # Save with tifffile, preserving data type
    if preserve_dtype:
        tifffile.imwrite(
            filepath,
            image,
            photometric=photometric,
            metadata=metadata_dict,
            compression='lzw'
        )
    else:
        # Convert to 8-bit if not preserving
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        tifffile.imwrite(
            filepath,
            image,
            photometric=photometric,
            metadata=metadata_dict,
            compression='lzw'
        )

def safe_save_image(image: np.ndarray, filepath: Union[str, pathlib.Path], 
                   use_pil: bool = True) -> bool:
    """
    Safely save image avoiding OpenCV limitations
    
    Args:
        image: Image array to save
        filepath: Output path
        use_pil: Use PIL instead of OpenCV for saving
    
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = pathlib.Path(filepath)
        
        if use_pil:
            # Use PIL for safer large image handling
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # RGB image
                    pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
                elif image.shape[2] == 1:
                    # Single channel
                    pil_image = Image.fromarray(image[:,:,0].astype(np.uint8), 'L')
                else:
                    # Convert multi-channel to RGB
                    rgb_image = image[:,:,:3] if image.shape[2] >= 3 else np.stack([image[:,:,0]]*3, axis=2)
                    pil_image = Image.fromarray(rgb_image.astype(np.uint8), 'RGB')
            else:
                # Grayscale
                pil_image = Image.fromarray(image.astype(np.uint8), 'L')
            
            # Save with PIL
            pil_image.save(str(filepath), format='TIFF', compression='lzw')
            return True
            
        else:
            # Use tifffile for better TIFF support
            if len(image.shape) == 3 and image.shape[2] == 3:
                photometric = 'rgb'
            else:
                photometric = 'minisblack'
            
            tifffile.imwrite(
                str(filepath),
                image.astype(np.uint8),
                photometric=photometric,
                compression='lzw'
            )
            return True
            
    except Exception as e:
        print(f"Failed to save image with primary method: {e}")
        # Fallback: try with basic tifffile
        try:
            tifffile.imwrite(str(filepath), image.astype(np.uint8))
            return True
        except Exception as e2:
            print(f"Failed to save image with fallback: {e2}")
            return False

def convert_ome_to_standard_tiff(ome_path: Union[str, pathlib.Path],
                                 output_path: Union[str, pathlib.Path],
                                 channel: Optional[int] = None,
                                 z_slice: Optional[int] = None,
                                 max_dimension: int = 8192,
                                 preserve_bit_depth: bool = False) -> Optional[pathlib.Path]:
    """
    Convert OME-TIFF to standard TIFF for DeepHistReg compatibility
    
    Args:
        ome_path: Path to OME-TIFF file
        output_path: Output directory
        channel: Specific channel to extract
        z_slice: Specific z-slice to extract
        max_dimension: Maximum allowed dimension (will downsample if larger)
        preserve_bit_depth: If True, preserves original bit depth
    
    Returns:
        Path to converted TIFF file or None if failed
    """
    try:
        # Calculate downsample factor based on file size and max dimension
        with tifffile.TiffFile(ome_path) as tif:
            if tif.series:
                shape = tif.series[0].shape
                print(f"Original image shape: {shape}")
                
                # Find height and width dimensions
                axes = tif.series[0].axes.upper()
                h_idx = axes.index('Y') if 'Y' in axes else -2
                w_idx = axes.index('X') if 'X' in axes else -1
                
                height = shape[h_idx]
                width = shape[w_idx]
                max_current = max(height, width)
                
                downsample_factor = 1
                if max_current > max_dimension:
                    downsample_factor = int(np.ceil(max_current / max_dimension))
                    print(f"Downsampling by factor {downsample_factor} to fit within {max_dimension}px")
        
        # Load OME-TIFF with downsampling
        image, metadata = load_ome_tiff(ome_path, channel=channel, z_slice=z_slice, 
                                       downsample_factor=downsample_factor)
        original_dtype = image.dtype
        
        print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
        
        # For registration preprocessing, we need 8-bit
        if not preserve_bit_depth:
            if image.dtype != np.uint8:
                # Normalize to 0-255 for registration only
                if np.issubdtype(image.dtype, np.floating):
                    image = np.clip(image * 255, 0, 255).astype(np.uint8)
                else:
                    # Scale from current bit depth
                    image_min, image_max = image.min(), image.max()
                    if image_max > image_min:
                        image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)
        
        # Ensure RGB format for DeepHistReg
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            if image.dtype == np.uint16:
                # For 16-bit, we need special handling
                image_8bit = ((image / 65535.0) * 255).astype(np.uint8)
                image_rgb = np.stack([image_8bit, image_8bit, image_8bit], axis=-1)
                if preserve_bit_depth:
                    # Convert back to 16-bit RGB
                    image = np.stack([image, image, image], axis=-1)
                else:
                    image = image_rgb
            else:
                image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] > 3:
            # Take first 3 channels if more than 3
            image = image[:, :, :3]
        
        # Save as standard TIFF
        output_file = pathlib.Path(output_path) / f"{pathlib.Path(ome_path).stem}_converted.tiff"
        
        success = safe_save_image(image, output_file, use_pil=True)
        
        if success and output_file.exists():
            print(f"Successfully converted to: {output_file}")
            return output_file
        else:
            print("Failed to save converted image")
            return None
            
    except Exception as e:
        print(f"Error converting OME-TIFF: {e}")
        import traceback
        traceback.print_exc()
        return None

def safe_load_image(image_path: Union[str, pathlib.Path]) -> Optional[np.ndarray]:
    """
    Safely load image using multiple backends
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image array or None if failed
    """
    image_path = pathlib.Path(image_path)
    
    # Try PIL first (better for large images)
    try:
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Remove alpha channel
            image = image[:, :, :3]
        
        print(f"Loaded with PIL: {image.shape}")
        return image
        
    except Exception as e:
        print(f"PIL loading failed: {e}")
    
    # Try tifffile
    try:
        image = tifffile.imread(str(image_path))
        
        # Handle different formats
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        
        # Ensure uint8
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        print(f"Loaded with tifffile: {image.shape}")
        return image
        
    except Exception as e:
        print(f"Tifffile loading failed: {e}")
    
    # Try OpenCV as last resort with error handling
    try:
        image = cv2.imread(str(image_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Loaded with OpenCV: {image.shape}")
            return image
        else:
            print("OpenCV returned None")
            
    except Exception as e:
        print(f"OpenCV loading failed: {e}")
    
    return None

def resize_to_match(source, target, preserve_range=True):
    """
    Resize source image to match target dimensions
    """
    target_shape = target.shape[:2]
    if source.shape[:2] != target_shape:
        print(f"Resizing source from {source.shape} to match target {target.shape}")
        source_resized = resize(source, target_shape, preserve_range=preserve_range, anti_aliasing=True)
        if preserve_range:
            source_resized = source_resized.astype(source.dtype)
        return source_resized
    return source

def preprocess_images_advanced(source, target):
    """
    Advanced preprocessing for better alignment between different modalities
    """
    # First resize source to match target dimensions
    source = resize_to_match(source, target)
    
    # Convert to grayscale for processing
    if source.ndim == 3:
        source_gray = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        source_gray = source.astype(np.uint8)
    
    if target.ndim == 3:
        target_gray = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        target_gray = target.astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    source_blur = cv2.GaussianBlur(source_gray, (5, 5), 1.0)
    target_blur = cv2.GaussianBlur(target_gray, (5, 5), 1.0)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    source_enhanced = clahe.apply(source_blur)
    target_enhanced = clahe.apply(target_blur)
    
    # Edge enhancement for better feature detection
    source_edges = cv2.Canny(source_enhanced, 30, 100)
    target_edges = cv2.Canny(target_enhanced, 30, 100)
    
    # Combine enhanced and edge information
    source_combined = cv2.addWeighted(source_enhanced, 0.7, source_edges, 0.3, 0)
    target_combined = cv2.addWeighted(target_enhanced, 0.7, target_edges, 0.3, 0)
    
    # Convert back to RGB for DeepHistReg
    source_final = np.stack([source_combined, source_combined, source_combined], axis=-1)
    target_final = np.stack([target_combined, target_combined, target_combined], axis=-1)
    
    return source_final, target_final

def create_robust_registration_params():
    """
    Create more robust registration parameters for difficult cases
    """
    params = default_initial_nonrigid()
    
    # More aggressive initial alignment
    params['initial_alignment_params'] = {
        'type': 'feature_based',
        'detector': 'superpoint',
        'matcher': 'superglue',
        'ransac_threshold': 10.0,
        'max_features': 10000,
        'match_ratio': 0.9,
        'use_mutual_best': False,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    
    # Adjust nonrigid parameters for large deformations
    params['nonrigid_params'] = {
        'type': 'demons',
        'iterations': [200, 150, 100, 50],
        'smoothing_sigma': 3.0,
        'update_field_sigma': 2.0,
        'max_step_length': 5.0,
        'use_histogram_matching': True,
        'use_symmetric_forces': True,
        'use_gradient_type': 'symmetric',
    }
    
    # Multi-resolution with more levels
    params['multiresolution_params'] = {
        'levels': 5,
        'shrink_factors': [16, 8, 4, 2, 1],
        'smoothing_sigmas': [8.0, 4.0, 2.0, 1.0, 0.5],
    }
    
    # More robust optimization
    params['optimization_params'] = {
        'metric': 'mattes_mutual_information',
        'number_of_bins': 32,
        'optimizer': 'gradient_descent',
        'learning_rate': 2.0,
        'min_step': 0.001,
        'iterations': 500,
        'relaxation_factor': 0.8,
        'gradient_magnitude_tolerance': 1e-6,
        'metric_sampling_strategy': 'random',
        'metric_sampling_percentage': 0.1,
    }
    
    params['loading_params']['loader'] = 'pil'
    params['loading_params']['downsample_factor'] = 1
    
    return params

def apply_transformation_to_ome_tiff(original_ome_path: Union[str, pathlib.Path],
                                    transformation_info: dict,
                                    target_shape: tuple,
                                    output_path: pathlib.Path,
                                    all_channels: bool = True,
                                    preserve_bit_depth: bool = True) -> Optional[pathlib.Path]:
    """
    Apply transformation to OME-TIFF preserving all channels and bit depth
    
    Args:
        original_ome_path: Path to original OME-TIFF
        transformation_info: Dictionary containing transformation data
        target_shape: Shape of target image
        output_path: Output directory
        all_channels: Whether to transform all channels or just first
        preserve_bit_depth: If True, preserves original bit depth (e.g., 16-bit)
    
    Returns:
        Path to transformed OME-TIFF or None if failed
    """
    try:
        # Load original OME-TIFF
        with tifffile.TiffFile(original_ome_path) as tif:
            series = tif.series[0]
            original_data = series.asarray()
            axes = series.axes.upper()
            metadata = {'axes': axes}
            original_dtype = original_data.dtype
            
            # Get channel information
            if 'C' in axes:
                c_idx = axes.index('C')
                num_channels = original_data.shape[c_idx]
            else:
                num_channels = 1
                c_idx = None
            
            # Load displacement field
            disp_field_path = transformation_info['displacement_field']
            if disp_field_path and pathlib.Path(disp_field_path).exists():
                displacement_field = np.load(str(disp_field_path))
                
                # Create output array with original dtype
                if c_idx is not None:
                    output_shape = list(original_data.shape)
                    output_shape[axes.index('Y')] = target_shape[0]
                    output_shape[axes.index('X')] = target_shape[1]
                    warped_data = np.zeros(output_shape, dtype=original_dtype)
                else:
                    warped_data = np.zeros(target_shape[:2], dtype=original_dtype)
                
                # Apply transformation to each channel
                h, w = target_shape[:2]
                flow = displacement_field.transpose(1, 2, 0)
                
                # Create mesh grid
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + flow[:, :, 0]).astype(np.float32)
                map_y = (y + flow[:, :, 1]).astype(np.float32)
                
                if all_channels and num_channels > 1:
                    # Transform each channel
                    for ch in range(num_channels):
                        if c_idx is not None:
                            channel_data = np.take(original_data, ch, axis=c_idx)
                        else:
                            channel_data = original_data
                        
                        # Handle multi-dimensional data
                        if channel_data.ndim > 2:
                            # Take middle slice if Z-stack
                            if 'Z' in axes:
                                z_idx = axes.index('Z')
                                z_middle = channel_data.shape[z_idx] // 2
                                channel_data = np.take(channel_data, z_middle, axis=z_idx)
                            
                            # Remove time dimension if present
                            if 'T' in axes:
                                t_idx = axes.index('T')
                                channel_data = np.take(channel_data, 0, axis=t_idx)
                        
                        # Resize if needed
                        if channel_data.shape[:2] != target_shape[:2]:
                            # Use appropriate interpolation for bit depth
                            if original_dtype == np.uint16:
                                # For 16-bit, use INTER_LINEAR to preserve values better
                                channel_data = cv2.resize(channel_data, (target_shape[1], target_shape[0]), 
                                                        interpolation=cv2.INTER_LINEAR)
                            else:
                                channel_data = cv2.resize(channel_data, (target_shape[1], target_shape[0]))
                        
                        # Apply transformation
                        # For 16-bit data, we need to handle carefully
                        if original_dtype == np.uint16:
                            warped_channel = cv2.remap(channel_data.astype(np.float32), map_x, map_y, 
                                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            warped_channel = np.clip(warped_channel, 0, 65535).astype(np.uint16)
                        else:
                            warped_channel = cv2.remap(channel_data, map_x, map_y, cv2.INTER_LINEAR)
                        
                        # Store in output
                        if c_idx is not None:
                            # Build index tuple for assignment
                            idx = [slice(None)] * len(warped_data.shape)
                            idx[c_idx] = ch
                            warped_data[tuple(idx)] = warped_channel
                        else:
                            warped_data = warped_channel
                else:
                    # Transform only first channel
                    if c_idx is not None:
                        channel_data = np.take(original_data, 0, axis=c_idx)
                    else:
                        channel_data = original_data
                    
                    # Handle dimensions as above
                    if channel_data.ndim > 2:
                        while channel_data.ndim > 2:
                            channel_data = channel_data[channel_data.shape[0] // 2]
                    
                    # Resize if needed
                    if channel_data.shape[:2] != target_shape[:2]:
                        if original_dtype == np.uint16:
                            channel_data = cv2.resize(channel_data, (target_shape[1], target_shape[0]), 
                                                    interpolation=cv2.INTER_LINEAR)
                        else:
                            channel_data = cv2.resize(channel_data, (target_shape[1], target_shape[0]))
                    
                    # Apply transformation
                    if original_dtype == np.uint16:
                        warped_data = cv2.remap(channel_data.astype(np.float32), map_x, map_y, 
                                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        warped_data = np.clip(warped_data, 0, 65535).astype(np.uint16)
                    else:
                        warped_data = cv2.remap(channel_data, map_x, map_y, cv2.INTER_LINEAR)
                
                # Save as OME-TIFF with preserved bit depth
                output_file = output_path / "warped_source_ome.tiff"
                save_ome_tiff(warped_data, output_file, metadata=metadata, preserve_dtype=preserve_bit_depth)
                
                return output_file
                
    except Exception as e:
        print(f"Failed to apply transformation to OME-TIFF: {e}")
        return None

def perform_ome_tiff_registration(source_path: Union[str, pathlib.Path],
                                 target_path: Union[str, pathlib.Path],
                                 output_path: pathlib.Path,
                                 source_channel: Optional[int] = None,
                                 target_channel: Optional[int] = None,
                                 source_z: Optional[int] = None,
                                 target_z: Optional[int] = None,
                                 max_dimension: int = 8192) -> Optional[pathlib.Path]:
    """
    Complete registration pipeline for OME-TIFF files with error handling
    
    Args:
        source_path: Path to source OME-TIFF
        target_path: Path to target OME-TIFF  
        output_path: Output directory
        source_channel: Specific channel to use from source
        target_channel: Specific channel to use from target
        source_z: Specific z-slice to use from source
        target_z: Specific z-slice to use from target
        max_dimension: Maximum image dimension (will downsample if larger)
    
    Returns:
        Path to registered OME-TIFF or None if failed
    """
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert OME-TIFFs to standard TIFFs for DeepHistReg
        print("Converting OME-TIFF files for registration...")
        source_converted = convert_ome_to_standard_tiff(
            source_path, output_path, channel=source_channel, z_slice=source_z,
            max_dimension=max_dimension
        )
        target_converted = convert_ome_to_standard_tiff(
            target_path, output_path, channel=target_channel, z_slice=target_z,
            max_dimension=max_dimension
        )
        
        if source_converted is None or target_converted is None:
            print("Failed to convert OME-TIFF files")
            return None
        
        # Load converted images safely
        print("Loading converted images...")
        source = safe_load_image(source_converted)
        target = safe_load_image(target_converted)
        
        if source is None or target is None:
            print("Failed to load converted images")
            return None
        
        print(f"Source shape: {source.shape}")
        print(f"Target shape: {target.shape}")
        
        # Preprocess for registration
        print("Preprocessing images...")
        source_prep, target_prep = preprocess_images_advanced(source, target)
        
        # Save preprocessed images
        prep_source_path = output_path / "source_preprocessed.tiff"
        prep_target_path = output_path / "target_preprocessed.tiff"
        
        if not safe_save_image(source_prep, prep_source_path):
            print("Failed to save preprocessed source")
            return None
        if not safe_save_image(target_prep, prep_target_path):
            print("Failed to save preprocessed target")  
            return None
        
        # Run DeepHistReg
        print("Running DeepHistReg...")
        params = create_robust_registration_params()
        
        config = {
            'source_path': prep_source_path,
            'target_path': prep_target_path,
            'output_path': output_path,
            'registration_parameters': params,
            'case_name': 'registration',
            'save_displacement_field': True,
            'copy_target': True,
            'delete_temporary_results': False,
            'temporary_path': output_path / 'TEMP'
        }
        
        deeperhistreg.run_registration(**config)
        
        # Check for result
        warped_result = output_path / "warped_source.tiff"
        if warped_result.exists():
            print("Registration completed, applying to OME-TIFF...")
            
            # Apply transformation to original OME-TIFF
            transformation_info = {
                'displacement_field': output_path / 'TEMP' / 'displacement_field.npy'
            }
            
            ome_result = apply_transformation_to_ome_tiff(
                source_path,
                transformation_info,
                target.shape,
                output_path,
                all_channels=True
            )
            
            return ome_result if ome_result else warped_result
        else:
            print("Registration completed but no warped result found")
            return None
            
    except Exception as e:
        print(f"Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_ome_tiff_alignment(target_path: Union[str, pathlib.Path],
                               warped_path: Union[str, pathlib.Path],
                               channel: Optional[int] = None,
                               num_patches: int = 50,
                               patch_size: int = 256):
    """
    Evaluate alignment quality for OME-TIFF files
    """
    # Load images
    target_img, target_meta = load_ome_tiff(target_path, channel=channel)
    warped_img, warped_meta = load_ome_tiff(warped_path, channel=channel)
    
    print(f"Target shape: {target_img.shape}")
    print(f"Warped source shape: {warped_img.shape}")
    
    # Ensure same dimensions
    if warped_img.shape != target_img.shape:
        warped_img = resize_to_match(warped_img, target_img)
    
    # Convert to 8-bit for visualization
    if target_img.dtype != np.uint8:
        target_img = ((target_img - target_img.min()) / (target_img.max() - target_img.min()) * 255).astype(np.uint8)
    if warped_img.dtype != np.uint8:
        warped_img = ((warped_img - warped_img.min()) / (warped_img.max() - warped_img.min()) * 255).astype(np.uint8)
    
    # Evaluate patches
    min_height, min_width = target_img.shape[:2]
    ssim_scores = []
    good_patches = 0
    threshold_ssim = 0.5
    
    # Create figure for visualization
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    axes = axes.ravel()
    
    for i in range(num_patches):
        if min_height <= patch_size or min_width <= patch_size:
            print("Image too small for patch extraction")
            break
            
        # Random patch
        y = np.random.randint(0, min_height - patch_size)
        x = np.random.randint(0, min_width - patch_size)
        
        target_patch = target_img[y:y+patch_size, x:x+patch_size]
        warped_patch = warped_img[y:y+patch_size, x:x+patch_size]
        
        # Convert to grayscale for SSIM
        if target_patch.ndim == 3:
            target_gray = cv2.cvtColor(target_patch, cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target_patch
            
        if warped_patch.ndim == 3:
            warped_gray = cv2.cvtColor(warped_patch, cv2.COLOR_RGB2GRAY)
        else:
            warped_gray = warped_patch
        
        # Normalize for SSIM
        target_gray = target_gray.astype(np.float64) / 255.0
        warped_gray = warped_gray.astype(np.float64) / 255.0
        
        # Compute SSIM
        patch_ssim = ssim(target_gray, warped_gray, data_range=1.0)
        ssim_scores.append(patch_ssim)
        
        if patch_ssim > threshold_ssim:
            good_patches += 1
        
        # Display
        if i < 50:
            # Create overlay
            if target_patch.ndim == 2:
                target_patch = cv2.cvtColor(target_patch, cv2.COLOR_GRAY2RGB)
            if warped_patch.ndim == 2:
                warped_patch = cv2.cvtColor(warped_patch, cv2.COLOR_GRAY2RGB)
                
            overlay = cv2.addWeighted(target_patch, 0.5, warped_patch, 0.5, 0)
            axes[i].imshow(overlay)
            axes[i].set_title(f"SSIM: {patch_ssim:.3f}", fontsize=8)
            axes[i].axis('off')
    
    plt.suptitle(f"OME-TIFF Patch Alignment - Good patches: {good_patches}/{num_patches}")
    plt.tight_layout()
    plt.show()
    
    if ssim_scores:
        print(f"\nAlignment Statistics:")
        print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")
        print(f"Std SSIM: {np.std(ssim_scores):.4f}")
        print(f"Min SSIM: {np.min(ssim_scores):.4f}")
        print(f"Max SSIM: {np.max(ssim_scores):.4f}")
        print(f"Good patches (SSIM > {threshold_ssim}): {good_patches}/{num_patches}")
    
    return ssim_scores
                                 
# Main execution example
if __name__ == "__main__":
    # Set up paths - UPDATE THESE WITH YOUR ACTUAL PATHS
    source=input("Enter source path")
    target=input("Enter target path")
    output=input("Enter target path")
    source_path = pathlib.Path(source)  # Update this!
    target_path = pathlib.Path(target)  # Update this!
    output_path = pathlib.Path(output)               # Update this!
    
    # If you have model files in a specific directory (like Kaggle), set this path
    # model_dir = "/kaggle/input/deephistreg/pytorch/default/1"  # Update if needed
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set OpenCV environment variable for large images
    os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**63 - 1)
    
    try:
        # Perform registration
        print("\nPerforming OME-TIFF registration...")
        
        # You can specify which channels/z-slices to use for registration
        # For example, use channel 0 from source and channel 1 from target
        final_result = perform_ome_tiff_registration(
            source_path, 
            target_path, 
            output_path,
            source_channel=0,    # Specify channel or None for all
            target_channel=0,    # Specify channel or None for all
            source_z=None,       # Specify z-slice or None for middle
            target_z=None,       # Specify z-slice or None for middle
            max_dimension=4096   # Reduce this if you have memory issues
        )
        
        if final_result and final_result.exists():
            print(f"Registration completed: {final_result}")
            
            # Evaluate alignment
            print("\nEvaluating alignment...")
            scores = evaluate_ome_tiff_alignment(
                target_path, 
                final_result,
                channel=0  # Evaluate specific channel
            )
            
            # Create comparison visualization
            print("\nCreating visual comparison...")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Load images for visualization
            source_img, _ = load_ome_tiff(source_path, channel=0)
            target_img, _ = load_ome_tiff(target_path, channel=0)
            result_img, _ = load_ome_tiff(final_result, channel=0)
            
            # Normalize for display
            def normalize_for_display(img):
                if img.dtype != np.uint8:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                return img
            
            source_img = normalize_for_display(source_img)
            target_img = normalize_for_display(target_img)
            result_img = normalize_for_display(result_img)
            
            # Display
            axes[0].imshow(source_img)
            axes[0].set_title("Original Source (OME-TIFF)")
            axes[0].axis('off')
            
            axes[1].imshow(target_img)
            axes[1].set_title("Target (OME-TIFF)")
            axes[1].axis('off')
            
            axes[2].imshow(result_img)
            axes[2].set_title("Aligned Source (OME-TIFF)")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("Registration failed - no output produced")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
