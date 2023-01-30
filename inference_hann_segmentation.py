import itertools
from math import sqrt
from scipy.special import softmax
from typing import List, Union, Sequence

import torch
import torch.nn.functional as F
# import torch should be first. Unclear issue, mentionned here: https://github.com/pytorch/pytorch/issues/2083
import numpy as np
import time
import fiona  # keep this import. it sets GDAL_DATA to right value
import rasterio
import ttach as tta
from collections import OrderedDict
from fiona.crs import to_string
from tqdm import tqdm
from rasterio import features
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, open_dict
from omegaconf.listconfig import ListConfig

from dataset.aoi import aois_from_csv
from utils.logger import get_logger, set_tracker
from models.model_choice import define_model, read_checkpoint
from utils import augmentation
from utils.utils import generate_patch_list
from utils.utils import get_device_ids, get_key_def, \
add_metadata_from_raster_to_sample, set_device

# Set the logging file
logging = get_logger(__name__)
def _pad(arr, chunk_size):
    """ Pads img_arr """
    w_diff = chunk_size - arr.shape[0]
    h_diff = chunk_size - arr.shape[1]
    if len(arr.shape) > 2:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff), (0, 0)), mode="reflect")
    else:
        padded_arr = np.pad(arr, ((0, w_diff), (0, h_diff)), mode="reflect")

    return padded_arr


def ras2vec(raster_file, output_path):
    # Create a generic polygon schema for the output vector file
    i = 0
    feat_schema = {'geometry': 'Polygon',
                   'properties': OrderedDict([('value', 'int')])
                   }
    class_value_domain = set()
    out_features = []

    print("   - Processing raster file: {}".format(raster_file))
    with rasterio.open(raster_file, 'r') as src:
        raster = src.read(1)
    mask = raster != 0
    # Vectorize the polygons
    polygons = features.shapes(raster, mask, transform=src.transform)

    # Create shapely polygon features
    for polygon in polygons:
        feature = {'geometry': {
            'type': 'Polygon',
            'coordinates': None},
            'properties': OrderedDict([('value', 0)])}

        feature['geometry']['coordinates'] = polygon[0]['coordinates']
        value = int(polygon[1])  # Pixel value of the class (layer)
        class_value_domain.add(value)
        feature['properties']['value'] = value
        i += 1
        out_features.append(feature)

    print("   - Writing output vector file: {}".format(output_path))
    num_layers = list(class_value_domain)  # Number of unique pixel value
    for num_layer in num_layers:
        polygons = [feature for feature in out_features if feature['properties']['value'] == num_layer]
        layer_name = 'vector_' + str(num_layer).rjust(3, '0')
        print("   - Writing layer: {}".format(layer_name))

        with fiona.open(output_path, 'w',
                        crs=to_string(src.crs),
                        layer=layer_name,
                        schema=feat_schema,
                        driver='GPKG') as dest:
            for polygon in polygons:
                dest.write(polygon)
    print("")
    print("Number of features written: {}".format(i))


def gen_img_samples(src, patch_list, chunk_size, *band_order):
    """
    TODO
    Args:
        src: input image (rasterio object)
        patch_list: list of patches index
        chunk_size: preset image tile size
        *band_order: ignore

    Returns: generator object

    """
    for patch in patch_list:
        patch_x, patch_y, patch_width, patch_height, hann_window = patch
        window = Window.from_slices(slice(patch_y, patch_y + patch_height),
                                    slice(patch_x, patch_x + patch_width))
        if band_order:
            patch_array = reshape_as_image(src.read(band_order[0], window=window))
        else:
            patch_array = reshape_as_image(src.read(window=window))
        patch_array = _pad(patch_array, chunk_size)

        yield patch_array, (patch_y, patch_height), (patch_x, patch_width), hann_window

def sigmoid(x):
   return 1/(1+np.exp(-x))
# def softmax(x):
#     return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

@torch.no_grad()
def segmentation(param,
                 input_image,
                 num_classes: int,
                 model,
                 chunk_size: int,
                 device,
                 scale: List,
                 tp_mem,
                 debug=False,
                 ):
    """

    Args:
        param: parameter dict
        input_image: opened image (rasterio object)
        num_classes: number of classes
        model: model weights
        chunk_size: image tile size
        device: cuda/cpu device
        scale: scale range
        tp_mem: memory temp file for saving numpy array to disk
        debug: True/False

    Returns:

    """
    use_hanning = True
    threshold = 0.5
    sample = {'sat_img': None, 'map_img': None, 'metadata': None}
    start_seg = time.time()
    print_log = True if logging.level == 20 else False  # 20 is INFO
    model.eval()  # switch to evaluate mode
    # initialize test time augmentation
    transforms = tta.aliases.d4_transform()
    tf_len = len(transforms)
    h_padded, w_padded = input_image.height + chunk_size, input_image.width + chunk_size
    patch_list = generate_patch_list(w_padded, h_padded, chunk_size, use_hanning)
    # pred_img = np.zeros((tf_len, h_padded , w_padded, num_classes), dtype=np.float16)
    npy_save = tp_mem.parent / f"{tp_mem.stem}.npy"

    fp = np.memmap(tp_mem, dtype='float16', mode='w+', shape=(tf_len, h_padded, w_padded, num_classes))
    img_gen = gen_img_samples(src=input_image, patch_list=patch_list, chunk_size=chunk_size)
    single_class_mode = False if num_classes > 1 else True
    for patch, h_idxs, w_idxs, hann_win in tqdm(img_gen, position=1, leave=False,
                                                desc=f'Inferring on patches'):
        hann_win = np.expand_dims(hann_win, -1)
        image_metadata = add_metadata_from_raster_to_sample(sat_img_arr=patch,
                                                            raster_handle=input_image,
                                                            raster_info={})
        sample['metadata'] = image_metadata
        totensor_transform = augmentation.compose_transforms(param,
                                                             dataset='tst',
                                                             scale=scale,
                                                             aug_type='totensor',
                                                             print_log=print_log)
        sample['sat_img']=patch
        sample= totensor_transform(sample)
        inputs = sample['sat_img'].unsqueeze_(0)
        inputs = inputs.to(device)
        output_lst = []
        for transformer in transforms:
            # augment inputs
            augmented_input = transformer.augment_image(inputs)
            with torch.cuda.amp.autocast():
                augmented_output = model(augmented_input)
            if isinstance(augmented_output, OrderedDict) and 'out' in augmented_output.keys():
                augmented_output = augmented_output['out']
            logging.debug(f'Shape of augmented output: {augmented_output.shape}')
            # reverse augmentation for outputs
            deaugmented_output = transformer.deaugment_mask(augmented_output).squeeze(dim=0)
            output_lst.append(deaugmented_output)
        outputs = torch.stack(output_lst)
        outputs = outputs.permute(0, 2, 3, 1).squeeze(dim=0)
        outputs = outputs.cpu().numpy() * hann_win
        # pred_img[:, h_idxs[0]:h_idxs[0]+h_idxs[1], w_idxs[0]:w_idxs[0]+w_idxs[1], :] += outputs
        fp[:, h_idxs[0]:h_idxs[0]+h_idxs[1], w_idxs[0]:w_idxs[0]+w_idxs[1], :] += outputs
    fp.flush()
    del fp
    fp = np.memmap(tp_mem, dtype='float16', mode='r', shape=(tf_len, h_padded, w_padded, num_classes))
    pred_img = np.zeros((h_padded, w_padded), dtype=np.uint8)
    for row, col in tqdm(itertools.product(range(0, h_padded, chunk_size),
                                           range(0, w_padded, chunk_size)),leave=False, desc="Writing to array"):
        arr1 = (fp[:, row:row + chunk_size, col:col + chunk_size, :]).mean(axis=0)
        if single_class_mode:
            arr1 = sigmoid(arr1)
            np.save(npy_save, arr1) # TODO: Remove asap, used for testing/debugging
            arr1 = (arr1 > threshold)
            arr1 = np.squeeze(arr1, axis=2).astype(np.uint8)
        else:
            arr1 = softmax(arr1, axis=-1)
            arr1 = np.argmax(arr1, axis=-1).astype(np.uint8)
        pred_img[row:row + chunk_size, col:col + chunk_size] = arr1
    end_seg = time.time() - start_seg
    logging.info('Segmentation operation completed in {:.0f}m {:.0f}s'.format(end_seg // 60, end_seg % 60))
    if debug:
        logging.debug(f'Bin count of final output: {np.unique(pred_img, return_counts=True)}')
    input_image.close()
    return pred_img[:input_image.height, :input_image.width]


def calc_inference_chunk_size(gpu_devices_dict: dict, max_pix_per_mb_gpu: int = 200, default: int = 512) -> int:
    """
    Calculate maximum chunk_size that could fit on GPU during inference based on thumb rule with hardcoded
    "pixels per MB of GPU RAM" as threshold. Threshold based on inference with a large model (Deeplabv3_resnet101)
    :param gpu_devices_dict: dictionary containing info on GPU devices as returned by lst_device_ids (utils.py)
    :param max_pix_per_mb_gpu: Maximum number of pixels that can fit on each MB of GPU (better to underestimate)
    :return: returns a downgraded evaluation batch size if the original batch size is considered too high
    """
    if not gpu_devices_dict:
        return default
    # get max ram for smallest gpu
    smallest_gpu_ram = min(gpu_info['max_ram'] for _, gpu_info in gpu_devices_dict.items())
    # rule of thumb to determine max chunk size based on approximate max pixels a gpu can handle during inference
    max_chunk_size = sqrt(max_pix_per_mb_gpu * smallest_gpu_ram)
    max_chunk_size_rd = int(max_chunk_size - (max_chunk_size % 256))  # round to the closest multiple of 256
    logging.info(f'Data will be split into chunks of {max_chunk_size_rd}')
    return max_chunk_size_rd


def override_model_params_from_checkpoint(
        params: DictConfig,
        checkpoint_params):
    """
    Overrides model-architecture related parameters from provided checkpoint parameters
    @param params: Original parameters as inputted through hydra
    @param checkpoint_params: Checkpoint parameters as saved during checkpoint creation when training
    @return:
    """
    modalities = get_key_def('modalities', params['dataset'], expected_type=Sequence)
    classes = get_key_def('classes_dict', params['dataset'], expected_type=(dict, DictConfig))

    modalities_ckpt = get_key_def('modalities', checkpoint_params['dataset'], expected_type=Sequence)
    classes_ckpt = get_key_def('classes_dict', checkpoint_params['dataset'], expected_type=(dict, DictConfig))
    model_ckpt = get_key_def('model', checkpoint_params, expected_type=(dict, DictConfig))

    if model_ckpt != params.model or classes_ckpt != classes or modalities_ckpt != modalities:
        logging.warning(f"\nParameters from checkpoint will override inputted parameters."
                        f"\n\t\t\t Inputted | Overriden"
                        f"\nModel:\t\t {params.model} | {model_ckpt}"
                        f"\nInput bands:\t\t{modalities} | {modalities_ckpt}"
                        f"\nOutput classes:\t\t{classes} | {classes_ckpt}")
        with open_dict(params):
            OmegaConf.update(params, 'dataset.modalities', modalities_ckpt)
            OmegaConf.update(params, 'dataset.classes_dict', classes_ckpt)
            OmegaConf.update(params, 'model', model_ckpt)
    return params


def main(params: Union[DictConfig, dict]) -> None:
    """
    Function to manage details about the inference on segmentation task.
    1. Read the parameters from the config given.
    2. Read and load the state dict from the previous training or the given one.
    3. Make the inference on the data specified in the config.
    -------
    :param params: (dict) Parameters inputted during execution.
    """
    # SETTING OUTPUT DIRECTORY
    state_dict = get_key_def('state_dict_path', params['inference'], to_path=True,
                             validate_path_exists=True,
                             wildcard='*pth.tar')
    # Dataset params
    bands_requested = get_key_def('bands', params['dataset'], default=[1, 2, 3], expected_type=Sequence)
    classes_dict = get_key_def('classes_dict', params['dataset'], expected_type=DictConfig)
    num_classes = len(classes_dict)
    num_classes = num_classes + 1 if num_classes > 1 else num_classes  # multiclass account for background
    num_bands = len(bands_requested)

    working_folder = state_dict.parent.joinpath(f'inference_{num_bands}bands')
    logging.info("\nThe state dict path directory used '{}'".format(working_folder))
    Path.mkdir(working_folder, parents=True, exist_ok=True)
    logging.info(f'\nInferences will be saved to: {working_folder}\n\n')
    # Default input directory based on default output directory
    raw_data_csv = get_key_def('raw_data_csv', params['inference'], default=working_folder,
                                 expected_type=str, to_path=True, validate_path_exists=True)

    # LOGGING PARAMETERS
    exper_name = get_key_def('project_name', params['general'], default='gdl-training')
    run_name = get_key_def(['tracker', 'run_name'], params, default='gdl')
    tracker_uri = get_key_def(['tracker', 'uri'], params, default=None, expected_type=str, to_path=True)
    set_tracker(mode='inference', type='mlflow', task='segmentation', experiment_name=exper_name, run_name=run_name,
                tracker_uri=tracker_uri, params=params, keys2log=['inference', 'augmentation'])

    # OPTIONAL PARAMETERS
    num_devices = get_key_def('gpu', params['inference'], default=0, expected_type=(int, bool))
    if num_devices > 1:
        logging.warning(f"Inference is not yet implemented for multi-gpu use. Will request only 1 GPU.")
        num_devices = 1
    max_used_ram = get_key_def('max_used_ram', params['inference'], default=25, expected_type=int)
    if not (0 <= max_used_ram <= 100):
        raise ValueError(f'\nMax used ram parameter should be a percentage. Got {max_used_ram}.')
    max_used_perc = get_key_def('max_used_perc', params['inference'], default=25, expected_type=int)
    scale = get_key_def('scale_data', params['augmentation'], default=[0, 1], expected_type=ListConfig)
    raster_to_vec = get_key_def('ras2vec', params['inference'], default=False)
    debug = get_key_def('debug', params, default=False, expected_type=bool)
    if debug:
        logging.warning(f'\nDebug mode activated. Some debug features may mobilize extra disk space and '
                        f'cause delays in execution.')

    # list of GPU devices that are available and unused. If no GPUs, returns empty dict
    gpu_devices_dict = get_device_ids(num_devices, max_used_ram_perc=max_used_ram, max_used_perc=max_used_perc)
    max_pix_per_mb_gpu = get_key_def('max_pix_per_mb_gpu', params['inference'], default=25, expected_type=int)
    auto_chunk_size = calc_inference_chunk_size(gpu_devices_dict=gpu_devices_dict,
                                                max_pix_per_mb_gpu=max_pix_per_mb_gpu, default=512)
    chunk_size = get_key_def('chunk_size', params['inference'], default=auto_chunk_size, expected_type=int)
    device = set_device(gpu_devices_dict=gpu_devices_dict)
    # Read the concatenation point if requested model is deeplabv3 dualhead
    conc_point = get_key_def('conc_point', params['model'], None)

    model = define_model(
        net_params=params.model,
        in_channels=num_bands,
        out_classes=num_classes,
        main_device=device,
        devices=[list(gpu_devices_dict.keys())],
        state_dict_path=state_dict,
    )

    # GET LIST OF INPUT IMAGES FOR INFERENCE
    list_aois = aois_from_csv(csv_path=raw_data_csv, bands_requested=bands_requested)

    # LOOP THROUGH LIST OF INPUT IMAGES
    for aoi in tqdm(list_aois, desc='Inferring from images', position=0, leave=True):
        Path.mkdir(working_folder / aoi.raster_name.parent.name, parents=True, exist_ok=True)
        inference_image = working_folder / aoi.raster_name.parent.name / f"{aoi.raster_name.stem}_inference.tif"
        temp_file = working_folder / aoi.raster_name.parent.name / f"{aoi.raster_name.stem}.dat"
        logging.info(f'\nReading image: {aoi.raster_name.stem}')
        inf_meta = aoi.raster.meta

        pred = segmentation(param=params,
                            input_image=aoi.raster,
                            num_classes=num_classes,
                            model=model,
                            chunk_size=chunk_size,
                            device=device,
                            scale=scale,
                            tp_mem=temp_file,
                            debug=debug)

        pred = pred[np.newaxis, :, :].astype(np.uint8)
        inf_meta.update({"driver": "GTiff",
                         "height": pred.shape[1],
                         "width": pred.shape[2],
                         "count": pred.shape[0],
                         "dtype": 'uint8',
                         "compress": 'lzw'})
        logging.info(f'\nSuccessfully inferred on {aoi.raster_name}\nWriting to file: {inference_image}')
        with rasterio.open(inference_image, 'w+', **inf_meta) as dest:
            dest.write(pred)
        del pred
        try:
            temp_file.unlink()
        except OSError as e:
            logging.warning(f'File Error: {temp_file, e.strerror}')
        if raster_to_vec:
            start_vec = time.time()
            inference_vec = working_folder.joinpath(aoi.raster_name.parent.name,
                                                    f"{aoi.raster_name.stem}_inference.gpkg")
            ras2vec(inference_image, inference_vec)
            end_vec = time.time() - start_vec
            logging.info('Vectorization completed in {:.0f}m {:.0f}s'.format(end_vec // 60, end_vec % 60))
